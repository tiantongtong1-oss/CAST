import unittest
import numpy as np
from discriminative_radius import select_discriminative_radius


class DiscriminativeRadiusTests(unittest.TestCase):
    def test_separable_and_smallest_optimum(self):
        r = select_discriminative_radius([[.1, .2], [.15]], [[.5, .6], [.4]])
        self.assertEqual(r['coverage'], 1.)
        self.assertEqual(r['intrusion'], 0.)
        self.assertEqual(r['radius'], float(np.nextafter(np.float32(.2), np.float32(np.inf))))

    def test_tied_events_and_no_positive_gain(self):
        r = select_discriminative_radius([[.1, .2]], [[.1, .2]])
        self.assertEqual(r['radius'], 0.)
        self.assertEqual(r['score'], 0.)
        r = select_discriminative_radius([[.2]], [[.1]])
        self.assertEqual(r['radius'], 0.)

    def test_class_replication_invariant(self):
        p, n = [[.1, .3], [.2, .4]], [[.2, .5], [.3, .6]]
        a = select_discriminative_radius(p, n)
        b = select_discriminative_radius([np.tile(p[0], 100), p[1]],
                                         [np.tile(n[0], 100), n[1]])
        self.assertEqual(a, b)

    def test_matches_brute_force_strict_float32_boundaries(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            p = [rng.integers(0, 20, size=k).astype(np.float32) / 10 for k in (2, 7, 19)]
            n = [rng.integers(0, 20, size=k).astype(np.float32) / 10 for k in (2, 7, 19)]
            candidates = np.r_[0., np.nextafter(np.unique(np.concatenate(p + n)), np.float32(np.inf))]
            scores = [np.mean([(a < r).mean() for a in p]) -
                      np.mean([(a < r).mean() for a in n]) for r in candidates]
            expected = next(float(r) for r, s in zip(candidates, scores) if s >= max(scores) - 1e-12)
            result = select_discriminative_radius(p, n)
            self.assertEqual(result['radius'], expected)
            self.assertAlmostEqual(result['score'], max(scores))

    def test_empty_missing_and_no_competitor(self):
        self.assertEqual(select_discriminative_radius([[]], [[]])['radius'], 0.)
        r = select_discriminative_radius([[], [.1]], [[], [np.inf]])
        self.assertEqual(r['class_coverage'], [None, 1.])
        self.assertEqual(r['intrusion'], 0.)

    def test_validation(self):
        for p, n in (([[.1]], []), ([[.1]], [[]]), ([[np.nan]], [[.2]]),
                     ([[-.1]], [[.2]]), ([[.1]], [[np.nan]]), ([[.1]], [[-np.inf]])):
            with self.subTest(p=p, n=n), self.assertRaises(ValueError):
                select_discriminative_radius(p, n)


if __name__ == '__main__':
    unittest.main()
