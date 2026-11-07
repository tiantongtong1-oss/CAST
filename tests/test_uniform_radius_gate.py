import itertools
import unittest
from radius_gate import uniform_radius_candidates


class UniformGateTests(unittest.TestCase):
    def test_truth_table(self):
        for confidence, agreement, radius, enabled in itertools.product((0, 1), repeat=4):
            with self.subTest(confidence=confidence, agreement=agreement,
                              radius=radius, enabled=enabled):
                high, rescued, rejected = uniform_radius_candidates(
                    confidence, agreement, radius, enabled)
                self.assertEqual(bool(high), bool(confidence and agreement and (radius or not enabled)))
                self.assertEqual(bool(rescued), bool(enabled and agreement and not confidence and radius))
                self.assertEqual(bool(rejected), bool(enabled and confidence and agreement and not radius))
                expected = agreement and (radius if enabled else confidence)
                self.assertEqual(bool(high | rescued), bool(expected))
                self.assertFalse(bool(high & rescued))


if __name__ == '__main__':
    unittest.main()
