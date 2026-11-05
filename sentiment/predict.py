"""Load an existing trusted artifact without training or downloading anything."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Predict English movie-review sentiment')
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    from .runtime import Predictor
    try:
        result = Predictor.load(args.artifact, args.device).predict(args.text)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
