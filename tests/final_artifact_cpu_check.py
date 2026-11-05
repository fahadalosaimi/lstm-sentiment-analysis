"""Explicit final trained-artifact CPU checks. No fit, backward, optimizer or dataset access.

Not auto-discovered by unittest; run explicitly or via final_inference_cpu.ipynb.
Archive preparation uses only the standard library. Missing PyTorch is NOT a pass.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import sys
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

RUN_ID = 'run_20261105T101047_013570Z'
SOURCE_ZIP_SHA = 'e49f872b0118bbd481c9838e17d60a9d3a894e62d39b043bbbf56e474071996d'
RESULTS_ZIP_SHA = 'bc16b5a4c13e21495b50d7433f906dc7554bcad149037c62ca08e4bfdd2e435e'
MODEL_SHA = '740572862066b6e39dcefd2ddafa4c77f2fbd8560590cd24a559851d65953718'
METADATA_SHA = 'f774d20762c5d1b8c9e18f2ceeedd7139a43a5de932984d3dd3c4487f0e27f7d'
SOURCE_HASHES = {
    'sentiment/__init__.py': 'e6017e636d8254480e3108f8a14ac84c39999423b5d6649c2a437f58a16fecd5',
    'sentiment/data.py': '0ef282da837cd6f5c72cc9fe9e847cec8fbc80694cb35d4880fbc8246ed23670',
    'sentiment/metrics.py': 'aa75019e1058a04d034a6fb982820b57f940da4841a2c8ac8e08e56280055f6c',
    'sentiment/runtime.py': 'aac5b2fca4788fd30c10397a53252f72b6c98af588a7f8f74db274f5de68efb4',
}
REVIEWS = [
    ('positive_1', 'positive', 'I loved this movie. The performances were excellent and the story was wonderful.'),
    ('positive_2', 'positive', 'This was a brilliant film with great acting and a beautiful ending.'),
    ('negative_1', 'negative', 'I hated this movie. The acting was terrible and the story was boring.'),
    ('negative_2', 'negative', 'This was an awful film, a complete waste of time with bad acting.'),
]
CONTEXT = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding='utf-8')
    temporary.replace(path)


def read_manifest_zip(path, expected_sha, prefix, manifest_name):
    raw = Path(path).read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError('Archive SHA-256 mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sum(i.file_size for i in archive.infolist()) > 100_000_000:
            raise ValueError('Duplicate members or oversized archive')
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if (name.is_absolute() or '..' in name.parts or not name.parts
                    or name.parts[0] != prefix or '\\' in info.filename
                    or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe archive path or symlink')
        manifest = json.loads(archive.read(prefix + '/' + manifest_name))
        expected = {prefix + '/' + name for name in manifest['files']} | {prefix + '/' + manifest_name}
        if set(names) != expected or archive.testzip() is not None:
            raise ValueError('Archive membership/CRC mismatch')
        payload = {}
        for name, item in manifest['files'].items():
            value = archive.read(prefix + '/' + name)
            if len(value) != item['bytes'] or sha(value) != item['sha256']:
                raise ValueError('Member fingerprint mismatch: ' + name)
            payload[name] = value
        payload[manifest_name] = archive.read(prefix + '/' + manifest_name)
    return payload


def prepare_inputs(source_zip, results_zip, workspace):
    source = read_manifest_zip(source_zip, SOURCE_ZIP_SHA, 'lstm_sentiment_analysis', 'MANIFEST.json')
    results = read_manifest_zip(results_zip, RESULTS_ZIP_SHA, RUN_ID, 'RESULTS_MANIFEST.json')
    if json.loads(source['MANIFEST.json'])['files'] != json.loads(results['source_manifest.json'])['files']:
        raise ValueError('Source ZIP is not the recorded experiment source')
    for name, digest in SOURCE_HASHES.items():
        if sha(source[name]) != digest:
            raise ValueError('Inference source mismatch: ' + name)
    if sha(results['model/model.pt']) != MODEL_SHA or sha(results['model/metadata.json']) != METADATA_SHA:
        raise ValueError('Not the pinned final trained model/metadata')
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=False)
    project, artifact = workspace / 'project', workspace / 'artifact'
    for name in SOURCE_HASHES:
        destination = project / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source[name])
    artifact.mkdir()
    for name in ('model.pt', 'metadata.json'):
        (artifact / name).write_bytes(results['model/' + name])
    # Only four inference source files and two final artifact files are extracted.
    # No training entry point, raw data, partial weights or prior logs are extracted.
    record = {'status': 'prepared_not_executed', 'source_zip_sha256': SOURCE_ZIP_SHA,
              'results_zip_sha256': RESULTS_ZIP_SHA, 'project': str(project),
              'artifact': str(artifact), 'inference_executed': False}
    write_report(workspace / 'preparation.json', record)
    return record


class FinalTrainedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from sentiment.runtime import Predictor, seed_everything
        seed_everything(42)
        cls.torch = torch
        cls.predictor = Predictor.load(CONTEXT['artifact'], device='cpu')
        cls.baseline = {key: tensor.detach().clone() for key, tensor in cls.predictor.model.state_dict().items()}

    def test_01_architecture_vocabulary_preprocessing(self):
        from dataclasses import asdict
        from sentiment.data import PREPROCESSING
        meta = CONTEXT['meta']
        p = self.predictor
        self.assertEqual(meta['architecture'], 'unidirectional-single-layer-lstm')
        self.assertEqual(asdict(p.config), meta['config'])
        self.assertEqual(meta['preprocessing'], PREPROCESSING)
        self.assertEqual(meta['labels'], {'negative': 0, 'positive': 1})
        self.assertEqual(meta['threshold'], 0.5)
        self.assertEqual(len(p.vocab), 20000)
        self.assertEqual(sorted(p.vocab.values()), list(range(20000)))
        self.assertEqual((p.vocab['<PAD>'], p.vocab['<UNK>']), (0, 1))
        expected = {'embedding.weight': (20000, 100), 'lstm.weight_ih_l0': (512, 100),
                    'lstm.weight_hh_l0': (512, 128), 'lstm.bias_ih_l0': (512,),
                    'lstm.bias_hh_l0': (512,), 'fc.weight': (1, 128), 'fc.bias': (1,)}
        state = p.model.state_dict()
        self.assertEqual({k: tuple(v.shape) for k, v in state.items()}, expected)
        self.assertEqual(sum(v.numel() for v in state.values()), 2117889)
        for tensor in state.values():
            self.assertEqual(tensor.device.type, 'cpu')
            self.assertEqual(tensor.dtype, self.torch.float32)
            self.assertTrue(self.torch.isfinite(tensor).all().item())
        self.assertEqual(self.torch.count_nonzero(state['embedding.weight'][0]).item(), 0)

    def check_examples(self, expected_label):
        for identifier, label, text in REVIEWS:
            if label != expected_label:
                continue
            with self.subTest(example=identifier):
                result = self.predictor.predict(text)
                CONTEXT['report']['examples'].append({'id': identifier, 'text': text,
                                                      'expected': label, 'actual': result})
                probability = result['positive_probability']
                self.assertTrue(math.isfinite(probability) and 0 <= probability <= 1)
                self.assertEqual(result['sentiment'], 'positive' if probability >= 0.5 else 'negative')
                self.assertEqual(result['sentiment'], label)

    def test_02_positive_reviews(self):
        self.check_examples('positive')

    def test_03_negative_reviews(self):
        self.check_examples('negative')

    def test_04_empty_and_unsupported_inputs(self):
        cases = [('empty', ''), ('whitespace', '   '), ('markup_only', '<br />!!!'),
                 ('arabic', 'مرحبا بالعالم'), ('chinese', '你好'), ('none', None), ('number', 42)]
        for name, value in cases:
            with self.subTest(case=name):
                rejected = False
                try:
                    self.predictor.predict(value)
                except ValueError:
                    rejected = True
                CONTEXT['report']['invalid_inputs'].append({'case': name, 'rejected': rejected})
                self.assertTrue(rejected)

    def test_05_unknown_english_words(self):
        result = self.predictor.predict('qzxvplm wurzqplx')
        self.assertEqual(result['unknown_tokens'], 2)
        self.assertEqual(result['tokens_used'], 2)
        self.assertTrue(math.isfinite(result['positive_probability']))
        CONTEXT['report']['unknown_word_result'] = result

    def test_06_predictions_stable_after_reload(self):
        from sentiment.runtime import Predictor
        restored = Predictor.load(CONTEXT['artifact'], device='cpu')
        for identifier, _, text in REVIEWS:
            with self.subTest(example=identifier):
                first = self.predictor.predict(text)
                repeated = self.predictor.predict(text)
                reloaded = restored.predict(text)
                delta = abs(first['positive_probability'] - reloaded['positive_probability'])
                CONTEXT['report']['reload_checks'].append({'id': identifier, 'before': first,
                    'repeated': repeated, 'after_reload': reloaded, 'absolute_delta': delta})
                self.assertEqual(first, repeated)
                self.assertEqual(first['sentiment'], reloaded['sentiment'])
                self.assertLessEqual(delta, 1e-7)

    def test_07_padding_and_single_batch(self):
        from sentiment.data import encode
        t = self.torch
        p = self.predictor
        ids, length = encode(REVIEWS[0][2], p.vocab, p.config.max_length)
        with t.inference_mode():
            original = p.model(t.tensor([ids]), t.tensor([length]))
            padded = p.model(t.tensor([ids + [0] * 8]), t.tensor([length]))
        CONTEXT['report']['forward_executed'] = True
        self.assertEqual(tuple(original.shape), (1,))
        t.testing.assert_close(original, padded, rtol=0, atol=1e-7)

    def test_99_no_weights_or_files_changed(self):
        self.assertFalse(self.predictor.model.training)
        for key, value in self.predictor.model.state_dict().items():
            self.assertTrue(self.torch.equal(value, self.baseline[key]), key)
        self.assertTrue(all(p.grad is None for p in self.predictor.model.parameters()))
        self.assertEqual(sha((Path(CONTEXT['artifact']) / 'model.pt').read_bytes()), MODEL_SHA)
        self.assertEqual(sha((Path(CONTEXT['artifact']) / 'metadata.json').read_bytes()), METADATA_SHA)


def run_checks(project, artifact, report_path):
    report = {'status': 'not_run', 'inference_executed': False, 'retraining': False,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'run_id': RUN_ID,
              'model_sha256': MODEL_SHA, 'metadata_sha256': METADATA_SHA,
              'source_sha256': SOURCE_HASHES, 'checker_sha256': sha(Path(__file__).read_bytes()),
              'examples': [], 'invalid_inputs': [], 'reload_checks': [], 'tests_run': 0}
    try:
        project, artifact = Path(project).resolve(), Path(artifact).resolve()
        for name, expected in SOURCE_HASHES.items():
            if sha((project / name).read_bytes()) != expected:
                raise ValueError('Pinned inference source mismatch: ' + name)
        if sha((artifact / 'model.pt').read_bytes()) != MODEL_SHA or sha((artifact / 'metadata.json').read_bytes()) != METADATA_SHA:
            raise ValueError('Final artifact fingerprint mismatch')
        report['artifact_integrity'] = 'passed'
        meta = json.loads((artifact / 'metadata.json').read_text())
        report.update(model_config=meta['config'], preprocessing=meta['preprocessing'],
                      vocabulary_size=len(meta['vocabulary']))
        if importlib.util.find_spec('torch') is None:
            report['reason'] = 'PyTorch unavailable; final inference NOT executed. Use Colab CPU.'
            write_report(report_path, report)
            print(report['reason'])
            return 2
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        sys.path.insert(0, str(project))
        import torch
        import numpy as np
        torch.set_num_threads(1)
        report['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
                                 'numpy': np.__version__, 'device': 'cpu', 'threads': 1}
        CONTEXT.update(artifact=artifact, meta=meta, report=report)
        suite = unittest.TestLoader().loadTestsFromTestCase(FinalTrainedArtifactTests)
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
        report.update(inference_executed=bool(report['examples'] or report.get('unknown_word_result')
                                            or report.get('forward_executed')), tests_run=result.testsRun,
                      failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                      failure_details=[{'test': str(test), 'traceback': trace} for test, trace in result.failures + result.errors])
        passed = result.wasSuccessful() and not result.skipped and result.testsRun == 8
        report['status'] = 'passed' if passed else 'failed'
        write_report(report_path, report)
        return 0 if passed else 1
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_report(report_path, report)
        print(type(error).__name__ + ': ' + str(error))
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--source-zip', type=Path)
    parser.add_argument('--results-zip', type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--artifact', type=Path)
    parser.add_argument('--report', type=Path, default=Path('outputs/final_inference/report.json'))
    args = parser.parse_args()
    if args.prepare:
        if not all((args.source_zip, args.results_zip, args.workspace)):
            parser.error('--prepare needs --source-zip, --results-zip and a new --workspace')
        print(json.dumps(prepare_inputs(args.source_zip, args.results_zip, args.workspace), indent=2))
        return 0
    if args.artifact is None:
        parser.error('--artifact is required for final inference checks')
    return run_checks(args.project, args.artifact, args.report)


if __name__ == '__main__':
    raise SystemExit(main())
