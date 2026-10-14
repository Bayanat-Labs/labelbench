import json
import tempfile
import unittest
from pathlib import Path

from labelbench.datasets import Example
from labelbench.models import Prediction
from labelbench.runner import run_classifier


class FixedClassifier:
    name = "fixed"

    def fit(self, examples):
        del examples

    def predict(self, text):
        del text
        return Prediction("a", 0.9)


class RunnerTests(unittest.TestCase):
    def test_new_prediction_file_is_scored_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test_file = root / "test.jsonl"
            test_file.write_text(
                json.dumps(
                    {"id": "one", "text": "نص", "label": "a", "metadata": {}},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            examples = [Example("one", "نص", "a", {})]
            report = run_classifier(
                FixedClassifier(),
                [],
                examples,
                root / "run",
                None,
                test_file,
                "fixture",
                "classification",
            )
            self.assertEqual(report["metrics"]["accuracy"], 1.0)
            resumed = run_classifier(
                FixedClassifier(),
                [],
                examples,
                root / "run",
                None,
                test_file,
                "fixture",
                "classification",
            )
            self.assertEqual(resumed["timing"]["resumed_predictions"], 1)
            self.assertEqual(resumed["timing"]["new_predictions"], 0)


if __name__ == "__main__":
    unittest.main()
