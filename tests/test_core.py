"""Fast regression tests; no downloads, optimizer steps, or real-data training."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from sentiment import data, metrics
except ModuleNotFoundError:
    data = metrics = None


class DataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(data, "Data implementation is missing")

    def test_html_and_punctuation_separate_words_and_keep_negation(self):
        self.assertEqual(data.tokenize("Good<br />bad/great &amp; isn't awful!"),
                         ["good", "bad", "great", "isn't", "awful"])

    def test_vocabulary_does_not_learn_held_out_tokens(self):
        records = [(f"review token{i} unique{'x' * i}", i % 2) for i in range(40)]
        splits = data.split_records(records, seed=42)
        vocab = data.build_vocab([text for text, _ in splits['train']])
        for text, _ in splits['test'] + splits['validation']:
            held_out = text.split()[-1]
            self.assertNotIn(held_out, vocab)
            ids, _ = data.encode(text, vocab, 6)
            self.assertIn(1, ids)

    def test_splits_are_disjoint_stratified_and_repeatable(self):
        rows = [(f"item {'x' * i}", i % 2) for i in range(40)]
        splits = data.split_records(rows, seed=5)
        self.assertEqual(splits, data.split_records(rows, seed=5))
        self.assertEqual([len(splits[k]) for k in ('train', 'validation', 'test')], [28, 6, 6])
        ids = [set(text for text, _ in splits[k]) for k in ('train', 'validation', 'test')]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        for subset in splits.values():
            self.assertEqual(sum(y for _, y in subset) * 2, len(subset))

    def test_duplicate_reviews_never_cross_splits(self):
        rows = [(f"item {'x' * i}", i % 2) for i in range(40)]
        splits = data.split_records(rows + [("ITEM x!", 1)], seed=5)
        self.assertEqual(sum(map(len, splits.values())), 40)

    def test_conflicting_duplicate_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            data.split_records([('good', 0), ('GOOD!', 1)])

    def test_padding_unknown_and_truncation(self):
        vocab = {'<PAD>': 0, '<UNK>': 1, 'good': 2}
        self.assertEqual(data.encode('good mystery', vocab, 4), ([2, 1, 0, 0], 2))
        self.assertEqual(data.encode('good good good', vocab, 2), ([2, 2], 2))

    def test_empty_and_unsupported_text_rejected(self):
        for text in ('', '  ', '<br />!!!', 'مرحبا'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                data.encode(text, {'<PAD>': 0, '<UNK>': 1}, 4)

    def test_vocab_limit_reserves_special_tokens_and_breaks_ties_stably(self):
        self.assertEqual(data.build_vocab(['zebra apple', 'zebra pear'], max_vocab=4),
                         {'<PAD>': 0, '<UNK>': 1, 'zebra': 2, 'apple': 3})

    def test_invalid_lengths_and_empty_training_are_rejected(self):
        with self.assertRaises(ValueError):
            data.encode('good', {'<PAD>': 0, '<UNK>': 1}, 0)
        with self.assertRaises(ValueError):
            data.build_vocab([])
        with self.assertRaises(ValueError):
            data.split_records([('good', 1), ('bad', 0)])

    def test_csv_schema_labels_and_missing_values_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.csv'
            path.write_text('review,sentiment\ngood,positive\nbad,negative\n', encoding='utf-8')
            self.assertEqual(data.load_reviews(path), [('good', 1), ('bad', 0)])
            for content in ('review,label\ngood,1\n', 'review,sentiment\n,positive\n',
                            'review,sentiment\ngood,maybe\n'):
                path.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    data.load_reviews(path)

    def test_default_data_path_is_independent_of_working_directory(self):
        self.assertEqual(data.DEFAULT_DATA_PATH, Path(__file__).resolve().parents[1] / 'Data' / 'IMDB Dataset.csv')

    def test_preparation_fits_training_only_and_preserves_source_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.csv'
            content = 'review,sentiment\n' + ''.join(
                f"review unique{'x' * i},{'positive' if i % 2 else 'negative'}\n"
                for i in range(40))
            path.write_text(content, encoding='utf-8')
            before = path.read_bytes()
            splits, vocab, summary = data.prepare_data(path, max_vocab=200, min_frequency=1)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(summary['source_records'], 40)
            self.assertEqual(summary['normalized_duplicates_removed'], 0)
            for text, _ in splits['validation'] + splits['test']:
                self.assertNotIn(text.split()[-1], vocab)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(metrics, "Metrics implementation is missing")

    def test_saved_confusion_matrix_reproduces_original_accuracy(self):
        result = metrics.from_confusion([[3188, 562], [466, 3284]])
        self.assertAlmostEqual(result['accuracy'], 6472 / 7500)
        self.assertAlmostEqual(result['macro_f1'], 0.8629108726507084)

    def test_uneven_batches_have_sample_weighted_loss(self):
        meter = metrics.BinaryMetrics()
        meter.update([0, 1], [0, 0], loss_sum=2.0)
        meter.update([1], [1], loss_sum=4.0)
        result = meter.result()
        self.assertEqual(result['loss'], 2.0)
        self.assertEqual(result['confusion_matrix'], [[1, 0], [1, 1]])
        self.assertAlmostEqual(result['accuracy'], 2 / 3)

    def test_empty_evaluation_and_invalid_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics.BinaryMetrics().result()
        with self.assertRaises(ValueError):
            metrics.BinaryMetrics().update([0], [2], 0.0)


class EntryPointTests(unittest.TestCase):
    def test_training_cli_requires_explicit_opt_in_without_importing_torch(self):
        result = subprocess.run([sys.executable, '-B', '-m', 'sentiment.train'],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('requires --train', result.stderr)

    def test_inference_help_does_not_require_pytorch_or_artifacts(self):
        result = subprocess.run([sys.executable, '-B', '-m', 'sentiment.predict', '--help'],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--artifact', result.stdout)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed; no installation authorized')
class TorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        from sentiment.runtime import Config, LSTMClassifier, seed_everything
        self.torch = torch
        self.Config, self.Model = Config, LSTMClassifier
        seed_everything(42)
        self.config = Config(embedding_dim=4, hidden_dim=5, max_length=6, max_vocab=10)
        self.vocab = {'<PAD>': 0, '<UNK>': 1, 'good': 2, 'bad': 3}
        self.model = LSTMClassifier(len(self.vocab), self.config).eval()

    def test_single_item_batch_keeps_batch_dimension(self):
        self.assertEqual(tuple(self.model(self.torch.tensor([[2, 0]]), self.torch.tensor([1])).shape), (1,))

    def test_extra_right_padding_does_not_change_logits(self):
        t = self.torch
        with t.no_grad():
            short = self.model(t.tensor([[2, 3]]), t.tensor([2]))
            padded = self.model(t.tensor([[2, 3, 0, 0, 0]]), t.tensor([2]))
        t.testing.assert_close(short, padded)

    def test_artifact_round_trip_preserves_prediction(self):
        from sentiment.runtime import save_artifact, Predictor
        t = self.torch
        with tempfile.TemporaryDirectory() as directory:
            save_artifact(self.model, self.vocab, self.config, Path(directory), {'purpose': 'untrained test fixture'})
            predictor = Predictor.load(Path(directory))
            actual = predictor.predict('good mystery')['positive_probability']
            ids, length = data.encode('good mystery', self.vocab, 6)
            with t.no_grad():
                expected = t.sigmoid(self.model(t.tensor([ids]), t.tensor([length]))).item()
            self.assertAlmostEqual(actual, expected)
            with self.assertRaises(ValueError):
                predictor.predict('')

    def test_seed_repeats_model_initialization(self):
        from sentiment.runtime import seed_everything
        seed_everything(42)
        other = self.Model(len(self.vocab), self.config)
        for left, right in zip(self.model.parameters(), other.parameters()):
            self.torch.testing.assert_close(left, right)

    def test_evaluation_includes_partial_batch_without_updating_weights(self):
        from sentiment.runtime import make_loader, evaluate
        rows = [('good', 1), ('bad', 0), ('good mystery', 1)]
        loader = make_loader(rows, self.vocab, self.config, shuffle=False, batch_size=2)
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        result = evaluate(self.model, loader, 'cpu')
        self.assertEqual(result['samples'], 3)
        for key, value in self.model.state_dict().items():
            self.torch.testing.assert_close(value, before[key])

    def test_corrupted_artifact_is_rejected(self):
        from sentiment.runtime import save_artifact, Predictor
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_artifact(self.model, self.vocab, self.config, path, {})
            with (path / 'model.pt').open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                Predictor.load(path)


if __name__ == '__main__':
    unittest.main()
