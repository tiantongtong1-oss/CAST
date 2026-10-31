import unittest
import numpy as np
from balanced_radius import balanced_radius_mean_std


class BalancedMeanStdTests(unittest.TestCase):
    def test_equal_class_mass_and_between_class_variation(self):
        # Two point-mass classes: mean=.4, std=.3, irrespective of class sizes.
        radius, mean, std = balanced_radius_mean_std([[.1] * 100, [.7]], 2)
        np.testing.assert_allclose([radius, mean, std], [1., .4, .3])

    def test_matches_explicit_weighted_population_moments(self):
        classes = [np.array([.1, .3, .8]), np.array([.2, .5])]
        values = np.concatenate(classes)
        weights = np.array([1/6] * 3 + [1/4] * 2)
        mean = np.sum(weights * values)
        std = np.sqrt(np.sum(weights * (values - mean) ** 2))
        np.testing.assert_allclose(balanced_radius_mean_std(classes, .5),
                                   [mean + .5 * std, mean, std])
        replicated = [np.tile(classes[0], 50), classes[1]]
        np.testing.assert_allclose(balanced_radius_mean_std(classes),
                                   balanced_radius_mean_std(replicated))

    def test_empty_missing_constant_and_zero_scale(self):
        self.assertEqual(balanced_radius_mean_std([[], []]), (0., 0., 0.))
        np.testing.assert_allclose(balanced_radius_mean_std([[], [.2, .2]]), [.2, .2, 0])
        np.testing.assert_allclose(balanced_radius_mean_std([[0., 2.]], 0), [1., 1., 1.])

    def test_invalid_inputs(self):
        for scale in (-1, float('nan'), float('inf'), True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                balanced_radius_mean_std([[.1]], scale)
        for values in ([-1], [float('inf')], [float('nan')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                balanced_radius_mean_std([values])


if __name__ == '__main__':
    unittest.main()
