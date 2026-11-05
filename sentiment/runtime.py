"""PyTorch model, sample-weighted evaluation, and portable inference artifacts."""
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

# Set before importing torch/initializing CUDA; does not change global OS settings.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from .data import PREPROCESSING, encode
from .metrics import BinaryMetrics


@dataclass(frozen=True)
class Config:
    seed: int = 42
    max_length: int = 200
    max_vocab: int = 20_000
    min_frequency: int = 2
    embedding_dim: int = 100
    hidden_dim: int = 128
    dropout: float = 0.5
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 10
    patience: int = 3

    def __post_init__(self):
        for key in ('max_length', 'min_frequency', 'embedding_dim', 'hidden_dim',
                    'batch_size', 'epochs', 'patience'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if self.max_vocab < 2 or not 0 <= self.dropout < 1 or self.learning_rate <= 0:
            raise ValueError('Invalid vocabulary size, dropout or learning rate')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim, 1)

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.fc(self.dropout(hidden[-1])).squeeze(-1)


def make_loader(rows, vocab, config, shuffle=False, batch_size=None):
    encoded = [encode(text, vocab, config.max_length) for text, _ in rows]
    if not encoded:
        raise ValueError('Cannot load an empty dataset')
    ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
    lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long)
    labels = torch.tensor([label for _, label in rows], dtype=torch.float32)
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(TensorDataset(ids, lengths, labels),
                      batch_size=batch_size or config.batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0, drop_last=False)


def evaluate(model, loader, device):
    model.eval()
    meter = BinaryMetrics()
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    with torch.inference_mode():
        for ids, lengths, labels in loader:
            labels = labels.to(device)
            logits = model(ids.to(device), lengths)
            loss_sum = criterion(logits, labels).item()
            meter.update(labels.cpu().tolist(), (logits >= 0).long().cpu().tolist(), loss_sum)
    return meter.result()


def fit(model, train_loader, validation_loader, config, device, on_epoch=None):
    """No test loader is accepted. Select checkpoint by lowest validation loss."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    history, best_state = [], None
    best_loss, best_epoch, stale = float('inf'), 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        meter = BinaryMetrics()
        for ids, lengths, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids.to(device), lengths)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            meter.update(labels.detach().cpu().tolist(),
                         (logits.detach() >= 0).long().cpu().tolist(),
                         loss.item() * labels.numel())
        validation = evaluate(model, validation_loader, device)
        history.append({'epoch': epoch, 'train': meter.result(), 'validation': validation})
        if validation['loss'] < best_loss:
            best_loss, best_epoch, stale = validation['loss'], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"Epoch {epoch}: train loss={history[-1]['train']['loss']:.4f}, "
              f"validation loss={validation['loss']:.4f}, accuracy={validation['accuracy']:.4f}")
        if on_epoch is not None:
            on_epoch(epoch, model, history, best_epoch)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return history, best_epoch


def save_artifact(model, vocab, config, directory, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights, metadata = directory / 'model.pt', directory / 'metadata.json'
    if weights.exists() or metadata.exists():
        raise FileExistsError('Refusing to overwrite existing model artifacts')
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights)
    payload = {
        'artifact_version': 1, 'architecture': 'unidirectional-single-layer-lstm',
        'config': asdict(config), 'vocabulary': vocab, 'preprocessing': PREPROCESSING,
        'labels': {'negative': 0, 'positive': 1}, 'threshold': 0.5,
        'model_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'provenance': provenance,
    }
    metadata.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')


class Predictor:
    def __init__(self, model, vocab, config, device='cpu'):
        self.model = model.to(device).eval()
        self.vocab, self.config, self.device = vocab, config, device

    @classmethod
    def load(cls, directory, device='cpu'):
        directory = Path(directory)
        meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        if (meta.get('artifact_version') != 1 or meta.get('preprocessing') != PREPROCESSING
                or meta.get('architecture') != 'unidirectional-single-layer-lstm'
                or meta.get('labels') != {'negative': 0, 'positive': 1}
                or meta.get('threshold') != 0.5):
            raise ValueError('Unsupported artifact format or preprocessing')
        vocab = meta['vocabulary']
        if (vocab.get('<PAD>') != 0 or vocab.get('<UNK>') != 1
                or sorted(vocab.values()) != list(range(len(vocab)))):
            raise ValueError('Invalid vocabulary indices')
        weights = directory / 'model.pt'
        if hashlib.sha256(weights.read_bytes()).hexdigest() != meta['model_sha256']:
            raise ValueError('Model checksum mismatch')
        config = Config(**meta['config'])
        model = LSTMClassifier(len(vocab), config)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        return cls(model, vocab, config, device)

    def predict(self, text):
        ids, length = encode(text, self.vocab, self.config.max_length)
        with torch.inference_mode():
            logit = self.model(torch.tensor([ids], device=self.device), torch.tensor([length]))
            probability = torch.sigmoid(logit).item()
        return {'sentiment': 'positive' if probability >= 0.5 else 'negative',
                'positive_probability': probability, 'tokens_used': length,
                'unknown_tokens': ids[:length].count(1)}
