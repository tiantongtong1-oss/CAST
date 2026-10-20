import unittest
import numpy as np
from dual_domain_ds import DSConfig
from neighbor_support import MeanSupportBank, audit_mean_support, otsu_threshold

class MeanSupportTest(unittest.TestCase):
    def bank(self, n):
        return MeanSupportBank(np.tile([.2, np.sqrt(.96)], (n, 1)),
                               np.zeros(n, dtype=int), np.arange(n), np.ones(n), 2)

    def test_repeated_weak_support_does_not_amplify(self):
        for n in (3, 11):
            result = self.bank(n).evaluate([[1., 0.]], [99], DSConfig(k=11))
            self.assertAlmostEqual(result['mass'][0, 0], .2, places=6)
            self.assertAlmostEqual(result['uncertainty'][0], .8, places=6)
            self.assertFalse(audit_mean_support(result, [0], .6)['pass_mask'][0])

    def test_bpa_and_wrong_label(self):
        bank = MeanSupportBank([[1., 0.], [1., 0.], [-1., 0.]],
                               [0, 1, 0], [1, 2, 3], [1., 1., 1.], 2)
        result = bank.evaluate([[1., 0.], [0., 0.]], [9, 10], DSConfig(k=3))
        np.testing.assert_allclose(result['mass'][0], [1/3, 1/3])
        np.testing.assert_allclose(result['mass'].sum(1) + result['uncertainty'], 1)
        self.assertFalse(result['available'][1])

    def test_otsu_boundaries_and_degenerate_input(self):
        self.assertEqual(otsu_threshold([.1, .1, .2, .8, .9]), .8)
        for values in ([], [.2], [.2, .2], [np.nan]):
            self.assertTrue(np.isinf(otsu_threshold(values)))
        self.assertEqual(otsu_threshold([0, 1]), 1)

if __name__ == '__main__':
    unittest.main()
