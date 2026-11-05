"""Colab-only controller. Importing this module never installs or trains anything."""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import traceback
from pathlib import Path

from .workflow import Workflow, export_results, validate_dataset, write_json


def verify_sources(root, manifest):
    for name, expected in manifest['files'].items():
        content = (Path(root) / name).read_bytes()
        if len(content) != expected['bytes'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
            raise ValueError('Project source changed after packaging: ' + name)


def run_colab_workflow(root, manifest, run_directory, config, train=True, existing_data_path=None):
    """Run remote stages sequentially, export diagnostics on caught failures."""
    from google.colab import files  # Intentionally unavailable outside a user-run Colab session.
    root = Path(root).resolve()
    wf = Workflow(run_directory)
    run_directory = wf.directory
    write_json(run_directory / 'config.json', config)
    write_json(run_directory / 'source_manifest.json', manifest)
    result_zip = run_directory.parent / (run_directory.name + '_results.zip')

    def command(arguments, log_name):
        print('\n>>>', ' '.join(str(x) for x in arguments), flush=True)
        with (run_directory / log_name).open('w', encoding='utf-8') as log:
            process = subprocess.Popen([str(x) for x in arguments], cwd=root,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1,
                                       env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                if process.wait() != 0:
                    raise subprocess.CalledProcessError(process.returncode, arguments)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise

    failure = None
    try:
        # 1: remote-only dependency setup; keep Colab's CUDA-enabled PyTorch build.
        verify_sources(root, manifest)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        wf.stage('environment_setup')
        command([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-colab.txt'], 'setup.log')
        import torch
        import numpy as np
        import matplotlib

        def version_pair(text):
            return tuple(int(x) for x in re.match(r'^(\d+)\.(\d+)', text).groups())

        if sys.version_info < (3, 11) or not (2, 6) <= version_pair(torch.__version__) < (3, 0):
            raise RuntimeError('Unsupported Python/PyTorch; choose a supported free GPU runtime')
        if not (1, 26) <= version_pair(np.__version__) < (3, 0) or not (3, 9) <= version_pair(matplotlib.__version__) < (4, 0):
            raise RuntimeError('Unsupported dependencies; restart a fresh runtime after setup')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Select a free GPU runtime; no CPU fallback.')
        wf.gate('environment', {
            'device': 'cuda', 'cuda_available': True, 'gpu': torch.cuda.get_device_name(0),
            'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
            'matplotlib': matplotlib.__version__, 'cuda_build': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        })

        # 2: all unit tests; every skip is a blocker, including missing model tests.
        wf.stage('unit_tests')
        command([sys.executable, '-B', '-m', 'tools.run_tests', '--report', run_directory / 'tests.json'], 'tests.log')
        wf.gate('tests', json.loads((run_directory / 'tests.json').read_text()))

        # 3: exactly one synthetic optimizer step through the production GPU path.
        wf.stage('gpu_smoke')
        script = "import sys; from sentiment.smoke import run_smoke; from sentiment.workflow import write_json; write_json(sys.argv[1], run_smoke('cuda'))"
        command([sys.executable, '-B', '-c', script, run_directory / 'smoke.json'], 'smoke.log')
        wf.gate('smoke', json.loads((run_directory / 'smoke.json').read_text()))

        # 4: upload a separate remote copy, then enforce schema and inspected hash.
        wf.stage('dataset')
        data_path = Path(existing_data_path).resolve() if existing_data_path else root / 'Data' / 'IMDB Dataset.csv'
        if existing_data_path and not data_path.is_file():
            raise FileNotFoundError('EXISTING_DATA_PATH is absent')
        if not data_path.exists():
            previous_cwd = Path.cwd()
            os.chdir(data_path.parent)
            try:
                uploaded = files.upload()
            finally:
                os.chdir(previous_cwd)
            if set(uploaded) != {'IMDB Dataset.csv'}:
                raise ValueError('Upload exactly IMDB Dataset.csv')
            del uploaded
        summary = validate_dataset(data_path)
        wf.gate('dataset', summary)
        print('Dataset verified:', {k: v for k, v in summary.items() if k != 'path'})

        if train:
            # 5: no training until all current-session gates and hashes agree.
            verify_sources(root, manifest)
            wf.require_training_ready()
            from .runtime import Config
            Config(**config)
            wf.stage('full_training')
            command([sys.executable, '-B', '-m', 'sentiment.train', '--train', '--device', 'cuda',
                     '--data', data_path, '--output', run_directory / 'model',
                     '--config', run_directory / 'config.json'], 'training.log')
            # run_training selects on validation only, restores best weights, tests once.
            wf.stage('reports_and_figures')
            from .reporting import generate_reports
            generate_reports(run_directory / 'model', run_directory)
            wf.complete()
            metrics = json.loads((run_directory / 'model' / 'metrics.json').read_text())
            print('NEW measured test results:', json.dumps(metrics['test'], indent=2))
        else:
            wf.verified_only()
            print('Verification complete; full training was explicitly disabled.')
    except BaseException as error:
        failure = error
        wf.fail(wf.state.get('stage', 'unknown'), error)
        (run_directory / 'workflow_error.log').write_text(traceback.format_exc(), encoding='utf-8')
        print('Workflow stopped:', type(error).__name__, str(error))
        print('Partial checkpoints are not final results. Inspect status.json and logs.')
    finally:
        # 6: export success or failure diagnostics; no input dataset is included.
        try:
            export_results(run_directory, result_zip)
        except Exception as export_error:
            wf.fail('export', export_error)
            failure = failure or export_error
            print('Export failed:', export_error)
            print('Files remain at:', run_directory)
        else:
            print('Run directory:', run_directory)
            print('Results ZIP:', result_zip)
            print('ZIP SHA-256:', hashlib.sha256(result_zip.read_bytes()).hexdigest())
            try:
                files.download(str(result_zip))
            except Exception as download_error:
                print('Automatic download failed:', download_error)
                print('Download this existing ZIP from the Colab Files panel:', result_zip)
    if failure is not None:
        raise RuntimeError('Workflow failed/interrupted; no successful experiment is claimed.') from failure
    return result_zip
