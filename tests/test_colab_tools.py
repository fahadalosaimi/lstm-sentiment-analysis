"""Portable packaging and Colab entry-point checks; no dependency installs."""
import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ColabToolTests(unittest.TestCase):
    def test_smoke_help_needs_no_pytorch(self):
        result = subprocess.run([sys.executable, '-B', '-m', 'sentiment.smoke', '--help'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--device', result.stdout)

    def test_bundle_allowlist_excludes_data_weights_and_original_notebook(self):
        from tools.package_colab import bundle_files
        files = bundle_files(ROOT)
        names = {p.relative_to(ROOT).as_posix() for p in files}
        self.assertIn('sentiment/runtime.py', names)
        self.assertIn('notebooks/colab_gpu.ipynb', names)
        self.assertIn('Data/README.md', names)
        self.assertEqual(sum(p.endswith('.ipynb') for p in names), 1)
        self.assertFalse(any(p.endswith(('.csv', '.pt', '.pth', '.zip', '.pyc')) for p in names))
        self.assertNotIn('DLProject_ipynb2 (1).ipynb', names)
        self.assertFalse(any('.env' in p or '__pycache__' in p for p in names))

    def test_zip_round_trip_has_only_manifested_files_and_valid_hashes(self):
        from tools.package_colab import build_bundle
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bundle.zip'
            build_bundle(output)
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                prefix = 'lstm_sentiment_analysis/'
                manifest = json.loads(archive.read(prefix + 'MANIFEST.json'))
                self.assertEqual(set(archive.namelist()),
                                 {prefix + name for name in manifest['files']} | {prefix + 'MANIFEST.json'})
                for name, metadata in manifest['files'].items():
                    content = archive.read(prefix + name)
                    self.assertEqual(hashlib.sha256(content).hexdigest(), metadata['sha256'])
                    self.assertEqual(len(content), metadata['bytes'])

    def test_credential_signature_stops_packaging_before_archive_creation(self):
        from tools.package_colab import FILES, build_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture', encoding='utf-8')
            # Synthetic signature only; never a real credential.
            (root / 'README.md').write_text('ghp_' + 'x' * 30, encoding='utf-8')
            output = root / 'bundle.zip'
            with self.assertRaisesRegex(ValueError, 'Potential credential'):
                build_bundle(output, root)
            self.assertFalse(output.exists())


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class SmokeTests(unittest.TestCase):
    def test_synthetic_training_step_and_artifact_round_trip(self):
        from sentiment.smoke import run_smoke
        result = run_smoke('cpu')
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['optimizer_steps'], 1)
        self.assertFalse(result['full_training'])


if __name__ == '__main__':
    unittest.main()
