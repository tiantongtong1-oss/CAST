import ast
import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np

from evaluate_pseudo_labels import (parse_args, settings_from_checkpoint,
                                    select_weight_key, rescue_selection, summarize, write_csv)
from dual_domain_ds import DSConfig
from neighbor_vote import NeighborVoteBank


class OfflineMetricsTests(unittest.TestCase):
    def args(self, *extra):
        return parse_args(['--checkpoint', 'source.pth', '--output_dir', 'out', *extra])

    def test_saved_settings_cli_priority_and_missing_weights(self):
        checkpoint = {'model': {}, 'ema_teacher': {},
                      'args': {'threshold_std_scale': .3, 'backbone': 'resnet18'}}
        settings = settings_from_checkpoint(self.args(), checkpoint)
        self.assertEqual(settings.threshold_std_scale, .3)
        self.assertEqual(settings.backbone, 'resnet18')
        settings = settings_from_checkpoint(self.args('--threshold_std_scale', '.7'), checkpoint)
        self.assertEqual(settings.threshold_std_scale, .7)
        self.assertEqual(select_weight_key(checkpoint, 'auto'), 'ema_teacher')
        self.assertEqual(select_weight_key({'model': {}}, 'auto'), 'model')
        with self.assertRaises(KeyError):
            select_weight_key({'model': {}}, 'ema_teacher')

    def test_invalid_options_and_settings(self):
        with self.assertRaises(ValueError):
            settings_from_checkpoint(self.args('--source_radius_quantile', '0'), {})
        with self.assertRaises(ValueError):
            settings_from_checkpoint(self.args('--rescue_probability_margin', 'nan'), {})
        with self.assertRaises(SystemExit):
            self.args('--threshold', '.9', '--previous_tau', '.8')

    def test_rescue_same_gate_as_v27_and_warmup_disable(self):
        bank = NeighborVoteBank(np.array([[1., 0.]] * 11), np.zeros(11, dtype=int),
                                np.arange(11), np.ones(11), 7)
        config = DSConfig(k=11, alpha=1.)
        probs = np.tile([.7, .05, .05, .05, .05, .05, .05], (4, 1))
        z = np.tile([1., 0.], (4, 1))
        means = np.array([[1., 0.], [0., 1.]] + [[0., 0.]] * 5)
        ready = np.array([True, True] + [False] * 5)
        settings = settings_from_checkpoint(self.args(), {})
        inputs = (probs, probs, z, np.zeros(4, dtype=int), np.arange(4),
                  np.array([False, True, False, False]),
                  np.array([True, True, False, True]), np.zeros(4, dtype=bool),
                  bank, config, means, ready, settings)
        # Row 0 passes via vote even outside radius. Row 3 also passes.
        np.testing.assert_array_equal(rescue_selection(*inputs, True), [True, False, False, True])
        np.testing.assert_array_equal(rescue_selection(*inputs, False), [False] * 4)
        bad_probs = probs.copy()
        bad_probs[:, :2] = [.51, .49]
        bad_probs[:, 2:] = 0
        altered = (bad_probs, bad_probs, *inputs[2:])
        self.assertFalse(rescue_selection(*altered, True).any())
        # Even radius membership cannot bypass unavailable full K neighborhood.
        small_bank = NeighborVoteBank(np.array([[1., 0.]]), np.array([0]),
                                      np.array([0]), np.ones(1), 7)
        altered = (*inputs[:7], np.ones(4, dtype=bool), small_bank, *inputs[9:])
        self.assertFalse(rescue_selection(*altered, True).any())

    def test_summary_precision_empty_and_csv(self):
        rows = summarize(np.array([1, 1, 3]), np.array([1, 2, 3]),
                         {'confidence': np.array([True, True, False]),
                          'rescued': np.array([False, False, True]),
                          'final': np.ones(3, dtype=bool)})
        fear = next(r for r in rows if r['stage'] == 'confidence' and r['class_id'] == 1)
        self.assertEqual(fear['accepted_count'], 2)
        self.assertEqual(fear['pseudo_accuracy'], .5)
        self.assertIsNone(rows[0]['pseudo_accuracy'])
        self.assertAlmostEqual(rows[-1]['pseudo_accuracy'], 2/3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.csv'
            write_csv(path, rows)
            with path.open() as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 24)
            self.assertEqual(saved[0]['pseudo_accuracy'], '')

    def test_no_training_or_role_promotion(self):
        tree = ast.parse(Path('evaluate_pseudo_labels.py').read_text())
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse({'backward', 'step', 'observe', 'save'} & attributes)
        self.assertNotIn('optimizer', {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})


if __name__ == '__main__':
    unittest.main()
