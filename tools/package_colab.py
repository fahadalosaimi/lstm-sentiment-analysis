"""Build a small allowlisted ZIP; no CSV, original notebook outputs or model weights."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'lstm_sentiment_analysis'
FILES = (
    '.gitignore', 'README.md', 'COLAB.md', 'FINAL_INFERENCE.md',
    'LICENSING.md', 'DEPENDENCIES_AND_RIGHTS.md', 'requirements.txt',
    'requirements-notebook.txt', 'requirements-colab.txt', 'Data/README.md',
    'notebooks/colab_gpu.ipynb',
    'sentiment/__init__.py', 'sentiment/data.py', 'sentiment/metrics.py',
    'sentiment/runtime.py', 'sentiment/train.py', 'sentiment/predict.py', 'sentiment/smoke.py',
    'sentiment/workflow.py', 'sentiment/reporting.py', 'sentiment/colab_workflow.py',
    'tests/test_core.py', 'tests/test_colab_tools.py', 'tests/test_workflow.py',
    'tests/final_artifact_cpu_check.py', 'tests/test_final_inference_harness.py',
    'tools/__init__.py', 'tools/package_colab.py', 'tools/run_tests.py',
    'reports/original_results.md',
)


def bundle_files(root=ROOT):
    paths = [Path(root) / name for name in FILES]
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or unsupported allowlisted file: {path.name}')
    return paths


def build_bundle(output, root=ROOT):
    root, output = Path(root), Path(output)
    payload = {}
    token_pattern = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
                               r'(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|'
                               r'github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}')
    for path in bundle_files(root):
        content = path.read_bytes()
        if token_pattern.search(content.decode('utf-8')):
            raise ValueError(f'Potential credential detected; packaging stopped: {path.name}')
        payload[path.relative_to(root).as_posix()] = content
    manifest = {'format_version': 1, 'files': {
        name: {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
        for name, content in payload.items()}}
    payload['MANIFEST.json'] = json.dumps(manifest, indent=2).encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(payload.items()):
            info = zipfile.ZipInfo(f'{PREFIX}/{name}', date_time=(2026, 11, 5, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        for name, content in payload.items():
            if archive.read(f'{PREFIX}/{name}') != content:
                raise ValueError('ZIP content mismatch')
    return {'archive': str(output.resolve()), 'entries': len(payload),
            'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist' / 'lstm_sentiment_source_current.zip')
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.output), indent=2))
