"""Run the entire suite and fail closed on skips or missing original model tests."""
import argparse
import sys
import unittest
from pathlib import Path

from sentiment.workflow import write_json


class RecordedResult(unittest.TextTestResult):
    def startTest(self, test):
        if not hasattr(self, 'ids'):
            self.ids = []
        self.ids.append(test.id())
        super().startTest(test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite = unittest.TestLoader().discover(str(root / 'tests'))
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, resultclass=RecordedResult).run(suite)
    summary = {'tests_run': result.testsRun, 'skipped': len(result.skipped),
               'failures': len(result.failures), 'errors': len(result.errors),
               'torch_tests': sum('.TorchTests.' in name for name in getattr(result, 'ids', []))}
    summary['passed'] = (result.wasSuccessful() and not result.skipped and summary['torch_tests'] == 6)
    write_json(args.report, summary)
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
