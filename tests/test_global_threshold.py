import unittest
from global_threshold import lambda_for_epoch, GlobalConfidenceMean, smooth_global_threshold


class GlobalThresholdTests(unittest.TestCase):
    def test_schedule_endpoints_midpoint_and_fixed_override(self):
        self.assertAlmostEqual(lambda_for_epoch(0, 30), .9)
        self.assertAlmostEqual(lambda_for_epoch(29, 30), 1.)
        self.assertAlmostEqual(lambda_for_epoch(50, 30), 1.)
        self.assertAlmostEqual(lambda_for_epoch(1, 3), .95)
        self.assertAlmostEqual(lambda_for_epoch(0, 1), 1.)
        self.assertAlmostEqual(lambda_for_epoch(20, 30, fixed=.87), .87)

    def test_sample_weighted_mean_not_batch_mean(self):
        stats = GlobalConfidenceMean()
        stats.add_batch(1.6, 2)
        stats.add_batch(.2, 1)
        tau, mean = stats.threshold(.9)
        self.assertAlmostEqual(mean, .6)
        self.assertAlmostEqual(tau, .54)

    def test_empty_and_invalid_configuration(self):
        with self.assertRaises(ValueError):
            GlobalConfidenceMean().threshold(1.)
        for kwargs in ({'start':float('nan')}, {'end':.8}, {'ramp_epochs':0}, {'fixed':0}):
            config = dict(epoch=0, ramp_epochs=30)
            config.update(kwargs)
            with self.assertRaises(ValueError):
                lambda_for_epoch(**config)


class ThresholdEMATests(unittest.TestCase):
    def test_first_epoch_has_no_zero_initialization_bias(self):
        self.assertEqual(smooth_global_threshold(.7), .7)

    def test_previous_smoothed_threshold_is_reused(self):
        first = smooth_global_threshold(.6)
        second = smooth_global_threshold(.8, first)
        third = smooth_global_threshold(.5, second)
        self.assertAlmostEqual(second, .64)
        self.assertAlmostEqual(third, .612)

    def test_constant_threshold_and_invalid_inputs(self):
        self.assertAlmostEqual(smooth_global_threshold(.7, .7), .7)
        for raw, previous, momentum in ((float('nan'),None,.8),(.7,float('inf'),.8),(.7,.6,1.)):
            with self.assertRaises(ValueError):
                smooth_global_threshold(raw, previous, momentum)


if __name__ == '__main__':
    unittest.main()

