"""Dependency-free gates, durable status and allowlisted experiment exports."""
import hashlib
import json
import os
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .data import load_reviews

DATA_SHA256 = 'dfc447764f82be365fa9c2beef4e8df89d3919e3da95f5088004797d79695aa2'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding='utf-8')
    os.replace(temporary, path)


def validate_dataset(path, expected_hash=DATA_SHA256, expected_rows=50_000):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError('Dataset fingerprint mismatch; confirm the source file/version')
    rows = load_reviews(path)  # Strict schema, missing values, labels and supported text.
    if len(rows) != expected_rows:
        raise ValueError('Unexpected dataset record count')
    counts = Counter(label for _, label in rows)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError('Both sentiment classes are required')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Dataset changed during validation')
    return {'path': str(path), 'sha256': digest, 'records': len(rows),
            'columns': ['review', 'sentiment'], 'missing_values': 0,
            'negative': counts[0], 'positive': counts[1]}


class Workflow:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.gates = {}
        self.failed = False
        self.state = {'status': 'initialized', 'gates': {},
                      'created_utc': datetime.now(timezone.utc).isoformat()}
        self._save()

    def _save(self):
        self.state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(self.directory / 'status.json', self.state)

    def stage(self, name):
        if self.failed:
            raise RuntimeError('Workflow failed; start a fresh verified run')
        self.state.update(status='running', stage=name)
        self._save()

    def gate(self, name, result):
        self.gates.pop(name, None)
        self.state['gates'].pop(name, None)
        self._save()
        if self.failed:
            raise RuntimeError('Cannot reuse gates from a failed workflow')
        if name == 'environment':
            valid = result.get('cuda_available') is True and result.get('device') == 'cuda'
        elif name == 'tests':
            valid = (result.get('tests_run', 0) > 0 and result.get('torch_tests') == 6
                     and all(result.get(k) == 0 for k in ('skipped', 'failures', 'errors')))
        elif name == 'smoke':
            valid = (result.get('status') == 'passed' and result.get('device') == 'cuda'
                     and result.get('optimizer_steps') == 1)
        elif name == 'dataset':
            valid = bool(result.get('sha256')) and result.get('missing_values') == 0
        else:
            raise ValueError('Unknown gate')
        if not valid:
            raise ValueError(f'{name} gate failed; training is blocked')
        self.gates[name] = result
        self.state['gates'][name] = result
        write_json(self.directory / f'{name}.json', result)
        self._save()

    def require_training_ready(self):
        if self.failed or set(self.gates) != {'environment', 'tests', 'smoke', 'dataset'}:
            raise RuntimeError('Training blocked: all four current-session gates must pass')
        data = self.gates['dataset']
        if hashlib.sha256(Path(data['path']).read_bytes()).hexdigest() != data['sha256']:
            raise RuntimeError('Training blocked: dataset changed after validation')

    def fail(self, stage, error):
        self.failed = True
        self.state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          stage=stage, error_type=type(error).__name__, error=str(error))
        self._save()

    def complete(self):
        self.require_training_ready()
        self.state.update(status='complete', stage='export')
        self._save()

    def verified_only(self):
        self.require_training_ready()
        self.state.update(status='verified_not_trained', stage='export')
        self._save()


def export_results(directory, output):
    """Export generated outputs only. Never sweep arbitrary datasets or credentials."""
    directory, output = Path(directory), Path(output)
    names = [
        'status.json', 'config.json', 'environment.json', 'tests.json', 'tests.log',
        'smoke.json', 'smoke.log', 'dataset.json', 'training.log', 'source_manifest.json',
        'setup.log', 'workflow_error.log',
        'model/model.pt', 'model/metadata.json', 'model/metrics.json', 'model/history.json',
        'model/status.json', 'model/partial/best_weights.pt', 'model/partial/metadata.json',
        'model/partial/history.json', 'reports/results.md', 'reports/metrics.csv',
        'reports/provenance.json', 'figures/loss.png', 'figures/accuracy.png',
        'figures/confusion_matrix.png',
    ]
    payload = {name: (directory / name).read_bytes() for name in names
               if (directory / name).is_file() and not (directory / name).is_symlink()}
    status = json.loads(payload.get('status.json', b'{}')).get('status', 'unknown')
    manifest = {'status': status, 'files': {
        name: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
        for name, value in payload.items()}}
    payload['RESULTS_MANIFEST.json'] = json.dumps(manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in payload.items():
            archive.writestr(f'{directory.name}/{name}', value)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Results archive CRC failure')
        for name, value in payload.items():
            if archive.read(f'{directory.name}/{name}') != value:
                raise ValueError('Results archive content mismatch')
    os.replace(temporary, output)
    return output
