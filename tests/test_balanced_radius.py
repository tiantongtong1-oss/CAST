import io
import unittest
import numpy as np
from balanced_radius import balanced_radius_quantile

try:
    import torch
except ImportError:
    torch = None


class BalancedRadiusTests(unittest.TestCase):
    def test_equal_class_weight_changes_pooled_answer(self):
        # Majority class is tight: pooling returns .1, equal class mass returns .8.
        classes = [np.full(100, .1), np.array([.8, 1.0])]
        self.assertEqual(balanced_radius_quantile(classes, .7), .8)
        self.assertEqual(np.quantile(np.concatenate(classes), .7), .1)

    def test_replication_within_one_class_does_not_change_radius(self):
        a, b = np.array([.1, .2, .3]), np.array([.4, .5])
        for q in (.1, .5, .7, .9, 1.):
            with self.subTest(q=q):
                self.assertEqual(balanced_radius_quantile([a, b], q),
                                 balanced_radius_quantile([np.tile(a, 100), b], q))

    def test_matches_equal_class_empirical_cdf(self):
        rng = np.random.default_rng(7)
        classes = [rng.random(n) for n in (281, 717, 4772, 705, 1290, 1982, 2524)]
        radius = balanced_radius_quantile(classes, .9)
        self.assertGreaterEqual(np.mean([(a <= radius).mean() for a in classes]), .9)
        self.assertLess(np.mean([(a < radius).mean() for a in classes]), .9)

    def test_missing_empty_ties_and_endpoint(self):
        self.assertEqual(balanced_radius_quantile([[], [.1, .2]], .5), .1)
        self.assertEqual(balanced_radius_quantile([[], []], .9), 0.)
        self.assertEqual(balanced_radius_quantile([[.2, .2], [.2]], .9), .2)
        self.assertEqual(balanced_radius_quantile([[.1], [.8, 1.]], 1.), 1.)

    def test_invalid_inputs(self):
        for q in (0, -1, 1.1, float('nan'), True):
            with self.subTest(q=q), self.assertRaises(ValueError):
                balanced_radius_quantile([[.1]], q)
        for values in ([-.1], [float('inf')], [float('nan')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                balanced_radius_quantile([values], .9)


@unittest.skipIf(torch is None, 'PyTorch not installed')
class RadiusBankIntegrationTests(unittest.TestCase):
    def test_refresh_strict_gate_missing_class_and_checkpoint(self):
        from source_prototype_radius import SourcePrototypeRadiusBank
        bank = SourcePrototypeRadiusBank(3, 2, source_radius_quantile=.7)
        features = torch.tensor([[1., 0.]] * 20 + [[1., 0.], [0., 1.], [-1., 0.]])
        labels = torch.tensor([0] * 20 + [1] * 3)
        bank.begin_source_refresh()
        bank.accumulate_source(features[:7], labels[:7])
        bank.accumulate_source(features[7:], labels[7:])
        bank.finalize_source()
        self.assertEqual(bank.class_source_counts.tolist(), [20, 3, 0])
        self.assertEqual(bank.distribution_initialized.tolist(), [True, True, False])
        self.assertTrue(torch.all(bank.class_radii == bank.class_radii[0]))
        distances = bank.distribution_distance(features, labels)
        self.assertTrue(torch.equal(bank.in_distribution(features, labels),
                                    distances < bank.class_radii[labels]))
        self.assertFalse(bank.in_distribution(features[:1], torch.tensor([2])).item())
        self.assertEqual(bank.class_radius_coverage[2], None)
        payload = io.BytesIO()
        torch.save(bank.state_dict(), payload)
        payload.seek(0)
        restored = SourcePrototypeRadiusBank(3, 2)
        restored.load_state_dict(torch.load(payload, weights_only=True), strict=True)
        self.assertTrue(torch.equal(bank.class_radii, restored.class_radii))
        # All-empty refresh clears radii and keeps every class unavailable.
        bank.begin_source_refresh()
        bank.finalize_source()
        self.assertTrue(torch.all(bank.class_radii == 0))
        self.assertFalse(bank.distribution_initialized.any())


if __name__ == '__main__':
    unittest.main()
