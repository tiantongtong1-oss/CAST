import unittest
import numpy as np
from source_class_weights import source_class_weights


class SourceClassWeightsTest(unittest.TestCase):
    def test_disabled_and_uniform(self):
        np.testing.assert_array_equal(source_class_weights([281, 4772]), [1, 1])
        np.testing.assert_array_equal(source_class_weights([10, 10], 'inverse_power'), [1, 1])

    def test_sqrt_ratio_and_normalization(self):
        weights = source_class_weights([1290, 281, 717, 4772, 1982, 705, 2524], 'inverse_power')
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(weights[1] / weights[3]), (4772 / 281) ** 0.5, places=5)
        self.assertGreater(weights[1], weights[2])

    def test_cap_and_missing_class(self):
        weights = source_class_weights([1, 1000000, 0], 'inverse_power', max_weight=1.5)
        self.assertEqual(weights[0], 1.5)
        self.assertEqual(weights[2], 0)
        self.assertTrue(np.isfinite(weights).all())

    def test_invalid(self):
        for counts in ([], [0, 0], [-1, 2], [np.nan, 2]):
            with self.assertRaises(ValueError):
                source_class_weights(counts)
        for kwargs in ({'power': -1}, {'power': np.nan}, {'max_weight': 0.5}, {'mode': 'bad'}):
            with self.assertRaises(ValueError):
                source_class_weights([1, 2], **kwargs)


if __name__ == '__main__':
    unittest.main()
