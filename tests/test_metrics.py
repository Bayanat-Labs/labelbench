import unittest

from labelbench.metrics import (
    ScoredPrediction,
    accuracy,
    brier_score,
    calibrated_safe_automation_coverage,
    classification_report,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    safe_automation_coverage,
    wilson_lower_bound,
)


class MetricsTests(unittest.TestCase):
    def test_accuracy_and_macro_f1(self):
        rows = [
            ScoredPrediction("a", "a", 0.9),
            ScoredPrediction("a", "b", 0.8),
            ScoredPrediction("b", "b", 0.7),
            ScoredPrediction("b", "b", 0.6),
        ]
        self.assertEqual(accuracy(rows), 0.75)
        self.assertAlmostEqual(macro_f1(rows), (2 / 3 + 0.8) / 2)
        report = {item["label"]: item for item in classification_report(rows)}
        self.assertEqual(report["a"]["support"], 2)
        self.assertAlmostEqual(report["a"]["precision"], 1.0)
        self.assertAlmostEqual(report["b"]["recall"], 1.0)
        self.assertEqual(
            confusion_matrix(rows),
            {"labels": ["a", "b"], "matrix": [[1, 1], [0, 2]]},
        )

    def test_wilson_bound_is_conservative(self):
        self.assertLess(wilson_lower_bound(98, 100), 0.98)
        self.assertGreater(wilson_lower_bound(100, 100), 0.95)

    def test_calibration_metrics(self):
        perfect = [
            ScoredPrediction("a", "a", 1.0),
            ScoredPrediction("a", "b", 0.0),
        ]
        self.assertEqual(brier_score(perfect), 0.0)
        self.assertEqual(expected_calibration_error(perfect), 0.0)

    def test_safe_coverage_uses_confidence_order(self):
        rows = [
            ScoredPrediction("a", "a", 1 - index / 100)
            for index in range(30)
        ] + [
            ScoredPrediction("a", "b", 0.20 - index / 100)
            for index in range(10)
        ]
        result = safe_automation_coverage(rows, 0.50)
        self.assertEqual(result["accepted"], 40)
        self.assertEqual(result["coverage"], 1.0)

    def test_no_safe_coverage_returns_zero(self):
        rows = [
            ScoredPrediction("a", "b", 0.99),
            ScoredPrediction("a", "b", 0.98),
        ]
        result = safe_automation_coverage(rows, 0.95)
        self.assertEqual(result["accepted"], 0)
        self.assertEqual(result["coverage"], 0.0)

    def test_safe_coverage_never_splits_confidence_ties(self):
        rows = (
            [ScoredPrediction("a", "a", 0.9) for _ in range(50)]
            + [ScoredPrediction("a", "b", 0.9) for _ in range(50)]
        )
        result = safe_automation_coverage(rows, 0.60)
        self.assertEqual(result["accepted"], 0)

    def test_calibrated_safe_coverage_selects_on_calibration_split(self):
        calibration = (
            [ScoredPrediction("a", "a", 0.9) for _ in range(100)]
            + [ScoredPrediction("a", "b", 0.4) for _ in range(10)]
        )
        evaluation = (
            [ScoredPrediction("a", "a", 0.95) for _ in range(6)]
            + [ScoredPrediction("a", "b", 0.3) for _ in range(6)]
        )
        result = calibrated_safe_automation_coverage(
            calibration,
            evaluation,
            0.90,
        )
        self.assertEqual(result["calibration"]["confidence_threshold"], 0.9)
        self.assertEqual(result["evaluation"]["accepted"], 6)
        self.assertEqual(result["evaluation"]["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
