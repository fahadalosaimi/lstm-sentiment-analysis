"""Synthetic smoke check: one optimizer step, no IMDb data and no persistent weights."""
import argparse
import json
import tempfile
from pathlib import Path


def run_smoke(device='cpu'):
    from .runtime import (Config, LSTMClassifier, Predictor, fit, make_loader,
                          save_artifact, seed_everything)
    import torch
    from .data import build_vocab
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: choose a GPU runtime before the GPU smoke check')
    seed_everything(42)
    config = Config(embedding_dim=8, hidden_dim=8, max_length=8, max_vocab=32,
                    min_frequency=1, batch_size=4, dropout=0.0, epochs=1, patience=1)
    training = [('good acting', 1), ('great story', 1), ('bad acting', 0), ('boring story', 0)]
    validation = [('good story', 1), ('bad story', 0)]
    vocab = build_vocab([text for text, _ in training], max_vocab=32)
    model = LSTMClassifier(len(vocab), config).to(device).eval()
    ids = torch.tensor([[vocab['good'], vocab['acting']]], device=device)
    lengths = torch.tensor([2])
    with torch.no_grad():
        short = model(ids, lengths)
        padded = model(torch.cat([ids, torch.zeros((1, 4), dtype=torch.long, device=device)], dim=1), lengths)
    assert tuple(short.shape) == (1,), 'Forward pass lost batch dimension'
    assert torch.isfinite(short).all(), 'Nonfinite forward result'
    torch.testing.assert_close(short, padded)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    train_loader = make_loader(training, vocab, config, shuffle=True)
    val_loader = make_loader(validation, vocab, config)
    assert len(train_loader) == 1
    history, best_epoch = fit(model, train_loader, val_loader, config, device)
    assert len(history) == best_epoch == 1
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items()), 'No weight update'
    assert torch.count_nonzero(model.embedding.weight[0]).item() == 0, 'PAD embedding changed'
    with tempfile.TemporaryDirectory(prefix='lstm_smoke_') as directory:
        path = Path(directory)
        save_artifact(model, vocab, config, path, {'purpose': 'synthetic one-step smoke; not a benchmark'})
        expected = Predictor(model, vocab, config, device).predict('good unknownword')['positive_probability']
        restored = Predictor.load(path, device).predict('good unknownword')['positive_probability']
        assert abs(expected - restored) < 1e-6, 'Artifact round trip changed prediction'
        # Also exercise portable GPU -> CPU restoration when checking CUDA.
        cpu_probability = Predictor.load(path, 'cpu').predict('good unknownword')['positive_probability']
        assert abs(expected - cpu_probability) < 1e-4, 'CPU restoration differs beyond smoke tolerance'
    return {'status': 'passed', 'device': str(device), 'optimizer_steps': 1,
            'forward': True, 'padding_invariant': True, 'weights_changed': True,
            'artifact_round_trip': True, 'cpu_restore': True,
            'fixture': 'synthetic', 'full_training': False, 'persistent_weights': False}


def main():
    parser = argparse.ArgumentParser(description='One synthetic optimizer step; no full training or dataset')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.device), indent=2))


if __name__ == '__main__':
    main()
