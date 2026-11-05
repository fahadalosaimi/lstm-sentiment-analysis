"""Generate figures/reports only from a completed improved training run."""
import csv
import json
from pathlib import Path

from .workflow import write_json


def generate_reports(model_directory, run_directory):
    model_directory, run_directory = Path(model_directory), Path(run_directory)
    state = json.loads((model_directory / 'status.json').read_text(encoding='utf-8'))
    if state.get('status') != 'trained' or not state.get('test_evaluated'):
        raise ValueError('Reports require a completed run with final test evaluation')
    report = json.loads((model_directory / 'metrics.json').read_text(encoding='utf-8'))
    history = json.loads((model_directory / 'history.json').read_text(encoding='utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figures, reports = run_directory / 'figures', run_directory / 'reports'
    figures.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    epochs = [row['epoch'] for row in history]
    for metric, title in [('loss', 'Sample-weighted BCE loss'), ('accuracy', 'Accuracy')]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for split in ('train', 'validation'):
            ax.plot(epochs, [row[split][metric] for row in history], label=split)
        ax.axvline(report['best_epoch'], color='gray', linestyle='--', label='selected epoch')
        ax.set(xlabel='Epoch (one-based)', ylabel=metric, title=title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f'{metric}.png', dpi=160)
        plt.close(fig)
    cm = report['test']['confusion_matrix']
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap='Blues')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'], xlabel='Predicted', ylabel='True',
           title='Improved model: final test')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                    color='white' if cm[i][j] > max(map(max, cm)) / 2 else 'black')
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    with (reports / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scope', 'metric', 'value'])
        for key in ('samples', 'loss', 'accuracy', 'macro_f1'):
            writer.writerow(['test', key, report['test'][key]])
        for label, values in report['test']['classes'].items():
            for key, value in values.items():
                writer.writerow([label, key, value])
    write_json(reports / 'provenance.json', {
        'config': report['config'], 'data': report['data'], 'environment': report['environment'],
        'selection_metric': report['selection_metric'], 'best_epoch': report['best_epoch'],
        'created_utc': report['created_utc'],
    })
    test = report['test']
    text = (
        '# Improved LSTM: completed experiment\n\n'
        f"Created UTC: {report['created_utc']}\n\n"
        f"Selection: lowest validation loss; epoch {report['best_epoch']}. "
        'Test was evaluated after selection, with a fixed 0.5 probability threshold.\n\n'
        '| Final test metric | Measured value |\n|---|---:|\n'
        f"| Samples | {test['samples']} |\n| Accuracy | {test['accuracy']:.6f} |\n"
        f"| Macro-F1 | {test['macro_f1']:.6f} |\n| BCE loss | {test['loss']:.6f} |\n\n"
        '![Loss](../figures/loss.png)\n\n![Accuracy](../figures/accuracy.png)\n\n'
        '![Confusion matrix](../figures/confusion_matrix.png)\n\n'
        'The original notebook stored 86.29% accuracy under a different protocol. '
        'It is historical and is not the score or a directly comparable baseline for this run.\n\n'
        'Limitations: English-only tokens, first-200-token truncation by default, '
        'one seed, custom random split, no movie-group separation, near-duplicate '
        'control, calibration or out-of-domain validation. No superiority claim is made.\n'
    )
    (reports / 'results.md').write_text(text, encoding='utf-8')
    return reports / 'results.md'
