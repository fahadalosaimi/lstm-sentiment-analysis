"""Explicit training entry point. Never runs at import time."""
import argparse
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .data import DEFAULT_DATA_PATH, ROOT, prepare_data
from .workflow import write_json


def run_training(data_path=DEFAULT_DATA_PATH, output=None, config=None, device='cpu'):
    # Heavy dependencies are only needed for an explicitly requested training run.
    from .runtime import Config, LSTMClassifier, evaluate, fit, make_loader, save_artifact, seed_everything
    import numpy as np
    import torch
    config = config or Config()
    output = Path(output) if output else ROOT / 'outputs' / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing runs are preserved')
    output.mkdir(parents=True)
    write_json(output / 'status.json', {'status': 'running', 'stage': 'prepare'})
    try:
        seed_everything(config.seed)
        splits, vocab, summary = prepare_data(data_path, config.seed, config.max_vocab, config.min_frequency)
        train_loader = make_loader(splits['train'], vocab, config, shuffle=True)
        validation_loader = make_loader(splits['validation'], vocab, config)
        model = LSTMClassifier(len(vocab), config).to(device)
        partial = output / 'partial'
        partial.mkdir()
        from .data import PREPROCESSING
        write_json(partial / 'metadata.json', {
            'status': 'partial_not_a_completed_experiment', 'config': asdict(config),
            'vocabulary': vocab, 'preprocessing': PREPROCESSING, 'data': summary,
            'resume_supported': False,
        })

        def persist_epoch(epoch, current_model, history, best_epoch):
            if epoch == best_epoch:
                temporary = partial / 'best_weights.pt.tmp'
                torch.save({k: v.detach().cpu() for k, v in current_model.state_dict().items()}, temporary)
                temporary.replace(partial / 'best_weights.pt')
            write_json(partial / 'history.json', history)
            write_json(output / 'status.json', {
                'status': 'running', 'stage': 'training', 'completed_epochs': epoch,
                'best_epoch': best_epoch, 'test_evaluated': False,
            })

        started = time.perf_counter()
        history, best_epoch = fit(model, train_loader, validation_loader, config, device,
                                  on_epoch=persist_epoch)
        write_json(output / 'status.json', {'status': 'running', 'stage': 'final_test', 'best_epoch': best_epoch})
        # Test is evaluated once, after validation-only model selection is complete.
        test_metrics = evaluate(model, make_loader(splits['test'], vocab, config), device)
        report = {
            'status': 'trained', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'config': asdict(config), 'data': summary, 'best_epoch': best_epoch,
            'selection_metric': 'lowest validation loss', 'test': test_metrics,
            'elapsed_training_and_test_seconds': time.perf_counter() - started,
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'numpy': np.__version__, 'device': str(device),
                            'cuda_build': torch.version.cuda,
                            'gpu': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None},
        }
        save_artifact(model, vocab, config, output, report)
        write_json(output / 'metrics.json', report)
        write_json(output / 'history.json', history)
        write_json(output / 'status.json', {'status': 'trained', 'best_epoch': best_epoch, 'test_evaluated': True})
        print(f'Artifacts written to {output}')
        return output, report, history
    except BaseException as error:
        write_json(output / 'status.json', {
            'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            'error_type': type(error).__name__, 'error': str(error),
            'note': 'Partial checkpoints are diagnostic; exact optimizer/RNG resume is not implemented.',
        })
        raise


def main():
    parser = argparse.ArgumentParser(description='Train the improved IMDb LSTM (explicit opt-in)')
    parser.add_argument('--train', action='store_true', help='Required acknowledgement that this launches training')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--config', type=Path, help='Explicit Config values in a JSON file')
    args = parser.parse_args()
    if not args.train:
        parser.error('Training requires --train. Use python -m sentiment.data for a read-only data check.')
    config = None
    if args.config:
        from .runtime import Config
        config = Config(**json.loads(args.config.read_text(encoding='utf-8')))
    run_training(args.data, args.output, config=config, device=args.device)


if __name__ == '__main__':
    main()
