"""Dependency-free, deterministic preprocessing; the source CSV is never written."""
import csv
import hashlib
import html
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / 'Data' / 'IMDB Dataset.csv'
PREPROCESSING = {
    'version': 1, 'language': 'English', 'lowercase': True,
    'html': 'unescape then replace tags with spaces',
    'token_pattern': "[a-z]+(?:'[a-z]+)?", 'padding': 'right',
    'truncation': 'first max_length tokens', 'empty_input': 'raise ValueError',
    'pad_id': 0, 'unk_id': 1,
}


def tokenize(text):
    if not isinstance(text, str):
        raise ValueError('Review must be a string')
    text = html.unescape(text).lower().replace('’', "'")
    text = re.sub(r'<[^>]*>', ' ', text)
    return re.findall(PREPROCESSING['token_pattern'], text)


def load_reviews(path=DEFAULT_DATA_PATH):
    csv.field_size_limit(100_000_000)
    records = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['review', 'sentiment']:
            raise ValueError('Expected CSV columns: review,sentiment')
        for line, row in enumerate(reader, 2):
            text, label = row.get('review'), row.get('sentiment')
            if None in row or not text or label not in ('negative', 'positive'):
                raise ValueError(f'Invalid or missing CSV value at record line {line}')
            if not tokenize(text):
                raise ValueError(f'No supported English tokens at record line {line}')
            records.append((text, int(label == 'positive')))
    return records


def split_records(records, seed=42):
    """Deduplicate by normalized text, then split each class 70/15/15.

    This normalization is fixed, not fitted. Conflicting labels fail explicitly.
    Rounding remainders go to training. No review text is exported.
    """
    unique = {}
    for text, label in records:
        key = ' '.join(tokenize(text))
        if not key or label not in (0, 1):
            raise ValueError('Empty normalized review or invalid binary label')
        if key in unique and unique[key] != label:
            raise ValueError('Conflicting labels for a normalized duplicate review')
        unique[key] = label
    rng = random.Random(seed)
    splits = {'train': [], 'validation': [], 'test': []}
    for label in (0, 1):
        subset = [(text, y) for text, y in sorted(unique.items()) if y == label]
        n = int(len(subset) * 0.15)
        if n < 1:
            raise ValueError('Need at least 7 distinct reviews per class')
        rng.shuffle(subset)
        splits['validation'].extend(subset[:n])
        splits['test'].extend(subset[n:2*n])
        splits['train'].extend(subset[2*n:])
    for subset in splits.values():
        rng.shuffle(subset)
    return splits


def build_vocab(training_texts, max_vocab=20_000, min_frequency=1):
    if max_vocab < 2 or min_frequency < 1:
        raise ValueError('Vocabulary needs two special tokens and positive frequency')
    counts = Counter(token for text in training_texts for token in tokenize(text))
    if not counts:
        raise ValueError('Training texts must contain supported tokens')
    vocab = {'<PAD>': 0, '<UNK>': 1}
    words = sorted(counts, key=lambda word: (-counts[word], word))
    for word in words:
        if counts[word] >= min_frequency and len(vocab) < max_vocab:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_length):
    if max_length < 1:
        raise ValueError('max_length must be positive')
    tokens = tokenize(text)[:max_length]
    if not tokens:
        raise ValueError('Review has no supported English tokens after cleaning')
    ids = [vocab.get(token, 1) for token in tokens]
    return ids + [0] * (max_length - len(ids)), len(ids)


def prepare_data(path=DEFAULT_DATA_PATH, seed=42, max_vocab=20_000, min_frequency=2):
    records = load_reviews(path)
    splits = split_records(records, seed)
    vocab = build_vocab([text for text, _ in splits['train']], max_vocab, min_frequency)
    summary = {
        'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'source_records': len(records),
        'normalized_duplicates_removed': len(records) - sum(map(len, splits.values())),
        'split_protocol': 'normalized deduplication; stratified random 70/15/15; rounding to train',
        'seed': seed, 'vocabulary_size': len(vocab), 'splits': {},
    }
    for name, rows in splits.items():
        summary['splits'][name] = {
            'samples': len(rows), 'negative': sum(y == 0 for _, y in rows),
            'positive': sum(y == 1 for _, y in rows),
            'membership_sha256': hashlib.sha256(json.dumps(sorted(rows), ensure_ascii=True).encode()).hexdigest(),
        }
    return splits, vocab, summary


if __name__ == '__main__':
    # Read-only dataset/preprocessing check. No PyTorch import, training or file writes.
    _, _, report = prepare_data()
    print(json.dumps(report, indent=2))
