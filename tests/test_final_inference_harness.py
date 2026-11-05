"""Archive safety checks for the no-training final-artifact harness."""
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    import final_artifact_cpu_check as checker
except ImportError:
    checker = None


class FinalInferenceHarnessTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(checker, 'Final artifact test harness is missing')

    def archive(self, directory, extra=None, unsafe=False):
        payload = {'../escape.txt' if unsafe else 'safe.txt': b'fixture'}
        manifest = {'files': {name: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                              for name, raw in payload.items()}}
        path = Path(directory) / 'fixture.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, raw in payload.items():
                archive.writestr('fixture/' + name, raw)
            archive.writestr('fixture/MANIFEST.json', json.dumps(manifest))
            if extra:
                archive.writestr('fixture/extra.txt', b'unlisted')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_valid_manifest_returns_verified_payload_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self.archive(directory)
            payload = checker.read_manifest_zip(path, digest, 'fixture', 'MANIFEST.json')
            self.assertEqual(payload['safe.txt'], b'fixture')
            self.assertFalse((Path(directory) / 'safe.txt').exists())

    def test_wrong_archive_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.archive(directory)
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                checker.read_manifest_zip(path, '0' * 64, 'fixture', 'MANIFEST.json')

    def test_unlisted_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self.archive(directory, extra=True)
            with self.assertRaises(ValueError):
                checker.read_manifest_zip(path, digest, 'fixture', 'MANIFEST.json')

    def test_traversal_is_rejected_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self.archive(directory, unsafe=True)
            with self.assertRaises(ValueError):
                checker.read_manifest_zip(path, digest, 'fixture', 'MANIFEST.json')

    def test_help_does_not_import_pytorch_or_load_weights(self):
        script = Path(__file__).with_name('final_artifact_cpu_check.py')
        result = subprocess.run([sys.executable, '-B', str(script), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--artifact', result.stdout)


if __name__ == '__main__':
    unittest.main()
