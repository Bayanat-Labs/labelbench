import os
import unittest
from unittest.mock import patch

from labelbench.openai_api import OpenAIResponsesDialectClassifier


class OpenAIResponsesTests(unittest.TestCase):
    def test_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                OpenAIResponsesDialectClassifier(api_key=None)

    def test_payload_uses_strict_schema_and_disables_storage(self):
        classifier = OpenAIResponsesDialectClassifier(api_key="test-key")
        payload = classifier.build_payload("نص")
        self.assertFalse(payload["store"])
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertEqual(
            payload["text"]["format"]["schema"]["properties"]["label"]["enum"],
            list(classifier.labels),
        )
        self.assertIn(
            "confidence",
            payload["text"]["format"]["schema"]["properties"],
        )
        self.assertIn("confidence", payload["text"]["format"]["schema"]["required"])

    def test_extracts_and_parses_output_text(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"label":"EGY"}'}],
                }
            ]
        }
        content = OpenAIResponsesDialectClassifier._output_text(response)
        self.assertEqual(
            OpenAIResponsesDialectClassifier.parse_label(content),
            "EGY",
        )
        self.assertEqual(
            OpenAIResponsesDialectClassifier.parse_label_and_confidence(
                '{"label":"EGY","confidence":0.62}'
            ),
            ("EGY", 0.62),
        )


if __name__ == "__main__":
    unittest.main()
