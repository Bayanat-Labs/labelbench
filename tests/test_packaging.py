import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from labelbench.adjudication import default_review_template
from labelbench.cli import build_parser, main
from labelbench.datasets import MultiLabelExample, write_multilabel_examples
from labelbench.paths import CACHE_ENV_VAR, default_cache_dir


class PackagingTests(unittest.TestCase):
    def test_review_template_ships_with_the_package(self):
        template = default_review_template()
        self.assertIn("__TASK_MANIFEST_JSON__", template)
        self.assertIn("LabelBench by Bayanat Labs", template)

    def test_cache_dir_honours_environment_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {CACHE_ENV_VAR: directory}):
                self.assertEqual(default_cache_dir(), Path(directory) / "normalized")

    def test_cache_dir_defaults_outside_the_working_directory(self):
        environment = {
            key: value for key, value in os.environ.items() if key != CACHE_ENV_VAR
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            cache_dir = default_cache_dir()
        self.assertTrue(cache_dir.is_absolute())
        self.assertEqual(cache_dir.parts[-2:], ("labelbench", "normalized"))

    def test_explicit_cache_dir_flag_is_kept(self):
        args = build_parser().parse_args(
            ["pilot-aratox", "--output", "runs/x", "--cache-dir", "data/normalized"]
        )
        self.assertEqual(args.cache_dir, Path("data/normalized"))

    def test_make_review_bundle_uses_packaged_template_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples_path = root / "examples.jsonl"
            write_multilabel_examples(
                examples_path,
                [MultiLabelExample("task-1", "نص", ("cussing",), {})],
            )
            manifest_path = root / "bundles" / "manifest.json"
            bundle_path = root / "bundles" / "bundle.html"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "make-review-bundle",
                        "--examples",
                        str(examples_path),
                        "--manifest",
                        str(manifest_path),
                        "--bundle",
                        str(bundle_path),
                    ]
                )
            self.assertEqual(status, 0)
            bundle = bundle_path.read_text(encoding="utf-8")
            self.assertNotIn("__TASK_MANIFEST_JSON__", bundle)
            self.assertIn("task-1", bundle)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["tasks"]), 1)


if __name__ == "__main__":
    unittest.main()
