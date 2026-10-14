import unittest

from labelbench.multilabel_metrics import (
    ScoredMultiLabelPrediction,
    multilabel_report,
    subset_accuracy,
)


class MultiLabelMetricTests(unittest.TestCase):
    def test_perfect_multilabel_report(self):
        rows = [
            ScoredMultiLabelPrediction(
                frozenset({"a", "b"}),
                frozenset({"a", "b"}),
                1.0,
            ),
            ScoredMultiLabelPrediction(
                frozenset({"not"}),
                frozenset({"not"}),
                1.0,
            ),
        ]
        report = multilabel_report(rows, labels=("a", "b", "not"))
        self.assertEqual(subset_accuracy(rows), 1.0)
        self.assertEqual(report["micro_f1"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["hamming_loss"], 0.0)

    def test_partial_prediction_is_not_exact(self):
        rows = [
            ScoredMultiLabelPrediction(
                frozenset({"a", "b"}),
                frozenset({"a"}),
                0.8,
            )
        ]
        report = multilabel_report(rows, labels=("a", "b"))
        self.assertEqual(report["subset_accuracy"], 0.0)
        self.assertGreater(report["micro_f1"], 0.0)
        self.assertLess(report["micro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
