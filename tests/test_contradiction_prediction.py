import unittest

from contradiction_detection import Claim, predict_contradiction_pairs


class ContradictionPredictionTests(unittest.TestCase):
    def test_detects_opposing_claims(self):
        pairs = [
            (
                Claim("Paper A", "Accuracy", "Accuracy improves by 18% over the baseline.", "finding"),
                Claim("Paper B", "Accuracy", "Accuracy drops by 12% relative to the baseline.", "finding"),
                0.95,
            )
        ]

        predictions = predict_contradiction_pairs(pairs)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["prediction"], "Likely Contradiction")
        self.assertGreaterEqual(predictions[0]["score"], 0.8)

    def test_ignores_non_conflicting_claims(self):
        pairs = [
            (
                Claim("Paper A", "Latency", "Latency decreases by 20% on the new benchmark.", "finding"),
                Claim("Paper B", "Latency", "Latency decreases by 18% on the new benchmark.", "finding"),
                0.9,
            )
        ]

        predictions = predict_contradiction_pairs(pairs)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["prediction"], "Low Risk")
        self.assertLess(predictions[0]["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
