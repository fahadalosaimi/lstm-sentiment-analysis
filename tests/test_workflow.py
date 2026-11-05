"""Workflow safety and result packaging without ML dependencies or training."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    from sentiment import workflow
except ImportError:
    workflow = None


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(workflow, 'Workflow implementation missing')

    def test_training_refuses_missing_and_skipped_test_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            session = workflow.Workflow(Path(directory) / 'run')
            with self.assertRaises(RuntimeError):
                session.require_training_ready()
            with self.assertRaises(ValueError):
                session.gate('tests', {'tests_run': 28, 'skipped': 1, 'failures': 0, 'errors': 0, 'torch_tests': 6})
            self.assertNotIn('tests', session.gates)

    def test_dataset_change_after_checks_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.csv'
            path.write_text('review,sentiment\ngood,positive\nbad,negative\n', encoding='utf-8')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            summary = workflow.validate_dataset(path, digest, expected_rows=2)
            session = workflow.Workflow(Path(directory) / 'run')
            session.gate('environment', {'cuda_available': True, 'device': 'cuda'})
            session.gate('tests', {'tests_run': 28, 'skipped': 0, 'failures': 0, 'errors': 0, 'torch_tests': 6})
            session.gate('smoke', {'status': 'passed', 'device': 'cuda', 'optimizer_steps': 1})
            session.gate('dataset', summary)
            session.require_training_ready()
            path.write_text('changed', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                session.require_training_ready()

    def test_bad_fingerprint_and_schema_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.csv'
            path.write_text('wrong,label\ngood,positive\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                workflow.validate_dataset(path, '0' * 64, expected_rows=1)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                workflow.validate_dataset(path, digest, expected_rows=1)

    def test_failure_revokes_gates_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            session = workflow.Workflow(Path(directory) / 'run')
            session.gate('environment', {'cuda_available': True, 'device': 'cuda'})
            session.fail('smoke', RuntimeError('GPU unavailable'))
            self.assertEqual(json.loads((session.directory / 'status.json').read_text())['status'], 'failed')
            with self.assertRaises(RuntimeError):
                session.require_training_ready()

    def test_results_zip_excludes_dataset_and_unapproved_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            run.mkdir()
            (run / 'status.json').write_text('{"status":"failed"}', encoding='utf-8')
            (run / 'IMDB Dataset.csv').write_text('private fixture', encoding='utf-8')
            (run / '.env').write_text('private fixture', encoding='utf-8')
            output = workflow.export_results(run, root / 'results.zip')
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {'run/status.json', 'run/RESULTS_MANIFEST.json'})
                manifest = json.loads(archive.read('run/RESULTS_MANIFEST.json'))
                self.assertEqual(manifest['status'], 'failed')
                self.assertEqual(manifest['files']['status.json']['sha256'],
                                 hashlib.sha256((run / 'status.json').read_bytes()).hexdigest())

    def test_reporting_rejects_partial_training_without_plot_dependencies(self):
        from sentiment.reporting import generate_reports
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            (model / 'status.json').write_text('{"status":"running"}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'completed run'):
                generate_reports(model, root)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed; no local training permitted')
class TrainingPersistenceTests(unittest.TestCase):
    def test_tiny_run_saves_partial_checkpoint_and_final_status(self):
        from sentiment.runtime import Config
        from sentiment.train import run_training
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'fixture.csv'
            path.write_text('review,sentiment\n' + ''.join(
                f"film {'x' * (i + 1)},{'positive' if i % 2 else 'negative'}\n"
                for i in range(20)), encoding='utf-8')
            config = Config(embedding_dim=4, hidden_dim=5, max_length=4, max_vocab=32,
                            min_frequency=1, batch_size=64, epochs=1)
            output, report, history = run_training(path, root / 'model', config, device='cpu')
            self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'trained')
            self.assertTrue((output / 'partial/best_weights.pt').is_file())
            self.assertEqual(len(history), 1)
            self.assertEqual(report['best_epoch'], 1)


if __name__ == '__main__':
    unittest.main()
