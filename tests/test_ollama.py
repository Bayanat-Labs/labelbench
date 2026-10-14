import unittest

from labelbench.ollama import (
    OllamaDialectRegionClassifier,
    OllamaMultiLabelToxicityClassifier,
    OllamaToxicityClassifier,
)


class OllamaTests(unittest.TestCase):
    def test_parse_valid_label(self):
        self.assertEqual(
            OllamaToxicityClassifier.parse_label('{"label":"toxic"}'), "toxic"
        )
        self.assertEqual(
            OllamaToxicityClassifier.parse_label_and_confidence(
                '{"label":"toxic","confidence":0.73}'
            ),
            ("toxic", 0.73),
        )
        self.assertEqual(
            OllamaToxicityClassifier.parse_label_and_confidence(
                '{"label":"toxic","confidence":7}'
            ),
            ("toxic", 1.0),
        )

    def test_parse_invalid_response(self):
        self.assertEqual(
            OllamaToxicityClassifier.parse_label("toxic"), "__invalid__"
        )
        self.assertEqual(
            OllamaToxicityClassifier.parse_label('{"label":"maybe"}'),
            "__invalid__",
        )

    def test_parse_multilabel_response(self):
        self.assertEqual(
            OllamaMultiLabelToxicityClassifier.parse_labels(
                '{"labels":["cussing","hatred"]}'
            ),
            ("cussing", "hatred"),
        )
        self.assertEqual(
            OllamaMultiLabelToxicityClassifier.parse_labels(
                '{"labels":["not","hatred"]}'
            ),
            ("__invalid__",),
        )
        self.assertEqual(
            OllamaMultiLabelToxicityClassifier.parse_labels_and_confidence(
                '{"labels":["cussing"],"confidence":0.41}'
            ),
            (("cussing",), 0.41),
        )

    def test_parse_dialect_region_response(self):
        self.assertEqual(
            OllamaDialectRegionClassifier.parse_label('{"label":"MGH"}'),
            "MGH",
        )
        self.assertEqual(
            OllamaDialectRegionClassifier.parse_label('{"label":"MSA"}'),
            "__invalid__",
        )


if __name__ == "__main__":
    unittest.main()
