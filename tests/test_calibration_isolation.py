import random
import unittest
import numpy as np
from neighbor_support import select_support_threshold


class ThresholdTests(unittest.TestCase):
    def test_constant_high_and_low_use_absolute_fallback(self):
        for score, accepted in ((.997, True), (.2, False)):
            result = select_support_threshold([score] * 200, 'otsu_guarded')
            self.assertEqual(result['mode'], 'fixed_fallback')
            self.assertEqual(score >= result['threshold'], accepted)

    def test_empty_small_and_narrow(self):
        for scores, reason in (([], 'no_candidates'),
                               ([.1, .9], 'too_few_candidates'),
                               (np.linspace(.995, .997, 200), 'narrow_distribution')):
            self.assertEqual(select_support_threshold(scores, 'otsu_guarded')['reason'], reason)

    def test_bimodal_and_original_modes(self):
        scores = [.2] * 100 + [.98] * 100
        result = select_support_threshold(scores, 'otsu_guarded')
        self.assertEqual(result['mode'], 'otsu')
        self.assertAlmostEqual(result['separation'], 1)
        self.assertEqual(select_support_threshold(scores, 'fixed')['threshold'], .92)
        self.assertTrue(np.isinf(select_support_threshold([.9], 'otsu')['threshold']))

    def test_invalid_guards(self):
        for kwargs in ({'fixed': float('nan')}, {'min_count': 1},
                       {'min_spread': -1}, {'min_separation': 2},
                       {'min_group_fraction': .6}):
            with self.assertRaises(ValueError):
                select_support_threshold([], **kwargs)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class IsolationTests(unittest.TestCase):
    def test_next_training_ids_and_augmentations_match(self):
        from calibration_rng import isolated_rng

        class AugmentedData(torch.utils.data.Dataset):
            def __len__(self):
                return 12

            def __getitem__(self, index):
                return index, random.random() + np.random.rand() + torch.rand(()).item()

        def run(scan):
            random.seed(3)
            np.random.seed(3)
            torch.manual_seed(3)
            train = torch.utils.data.DataLoader(AugmentedData(), batch_size=4,
                shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(3))
            if scan:
                calibration = torch.utils.data.DataLoader(AugmentedData(), batch_size=4,
                    num_workers=0, generator=torch.Generator().manual_seed(99))
                with isolated_rng(99):
                    list(calibration)
            return next(iter(train)), torch.rand(3)

        baseline, baseline_rng = run(False)
        calibrated, calibrated_rng = run(True)
        for expected, actual in zip(baseline, calibrated):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(baseline_rng, calibrated_rng))

    def test_exception_restores_states(self):
        from calibration_rng import isolated_rng
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        expected = (random.random(), np.random.rand(), torch.rand(1))
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        with self.assertRaises(RuntimeError):
            with isolated_rng(99):
                random.random()
                np.random.rand()
                torch.rand(3)
                raise RuntimeError('calibration failed')
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(1), expected[2]))
