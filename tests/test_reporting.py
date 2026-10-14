import json
import tempfile
import unittest
from pathlib import Path

from labelbench.reporting import build_release


class ReportingTests(unittest.TestCase):
    def test_build_release_writes_machine_and_human_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "report.json"
            comparison_path = root / "comparison.json"
            report_path.write_text(
                json.dumps(
                    {
                        "dataset": "fixture",
                        "task": "dialect_region_identification",
                        "model": "model<&",
                        "model_metadata": {"digest": "abc"},
                        "metrics": {
                            "accuracy": 0.75,
                            "accuracy_bootstrap_95": {"low": 0.5, "high": 1.0},
                            "macro_f1": 0.7,
                            "sac_95": {"coverage": 0.0},
                            "confusion_matrix": {
                                "labels": ["EGY", "GLF"],
                                "matrix": [[1, 1], [0, 2]],
                            },
                        },
                        "artifacts": {
                            "normalized_test_sha256": "test",
                            "predictions_sha256": "predictions",
                        },
                        "run": {"test_examples": 4},
                    }
                ),
                encoding="utf-8",
            )
            comparison_path.write_text(
                json.dumps(
                    {
                        "mcnemar_exact_two_sided_p": 0.04,
                        "paired_examples": 4,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "release"
            result = build_release(
                [report_path],
                output,
                "fixture-v1",
                generated_at="2026-01-01T00:00:00+00:00",
                comparison_paths=[comparison_path],
            )
            self.assertEqual(result["result_count"], 1)
            self.assertEqual(
                result["paired_comparisons"][0]["mcnemar_holm_adjusted_p"],
                0.04,
            )
            for name in (
                "summary.json",
                "leaderboard.csv",
                "README.md",
                "task-performance.svg",
                "dialect-confusion.svg",
            ):
                self.assertTrue((output / name).exists())
            self.assertIn("model&lt;&amp;", (output / "task-performance.svg").read_text())


if __name__ == "__main__":
    unittest.main()
