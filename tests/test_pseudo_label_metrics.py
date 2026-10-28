import ast
import csv
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pseudo_label_metrics import PseudoLabelMetrics, STAGES, initialize_metrics_csv


def masks(values):
    return {s: np.asarray(values, dtype=bool) for s in STAGES}


class PseudoMetricsTests(unittest.TestCase):
    def test_shuffled_ids_unequal_batches_and_predicted_class_grouping(self):
        metric = PseudoLabelMetrics([1, 3, 1, 2, 3])
        metric.update(np.array([4, 0, 3]), np.array([3, 3, 2]), masks([1, 1, 0]))
        metric.update(np.array([2, 1]), np.array([1, 3]), masks([1, 1]))
        # Class 3 has three accepted predictions, two correct (not GT recall).
        self.assertEqual(metric.accepted['supervised'].tolist(), [0, 1, 0, 3, 0, 0, 0])
        self.assertEqual(metric.correct['supervised'].tolist(), [0, 1, 0, 2, 0, 0, 0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.csv'
            initialize_metrics_csv(path)
            with contextlib.redirect_stdout(io.StringIO()):
                metric.write_epoch(path, 0)
                metric.write_epoch(path, 1)
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 80)
            row = next(r for r in rows if r['stage'] == 'supervised' and r['class_id'] == '3')
            self.assertAlmostEqual(float(row['pseudo_accuracy']), 2/3)
            empty = next(r for r in rows if r['class_id'] == '0')
            self.assertEqual(empty['pseudo_accuracy'], '')
            total = next(r for r in rows if r['class_name'] == 'ALL')
            self.assertEqual(float(total['pseudo_accuracy']), .75)

    def test_masks_are_independent_and_empty_acceptance_is_undefined(self):
        metric = PseudoLabelMetrics([1, 2])
        stage_masks = masks([0, 0])
        stage_masks['rescued'] = np.array([True, True])
        stage_masks['final'] = np.array([True, True])
        stage_masks['anchors'] = np.array([False, True])
        stage_masks['supervised'] = np.array([False, True])
        metric.update(np.array([1, 0]), np.array([2, 2]), stage_masks)
        self.assertEqual(metric.accepted['rescued'].sum(), 2)
        self.assertEqual(metric.correct['rescued'].sum(), 1)
        self.assertEqual(metric.correct['supervised'].sum(), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.csv'
            initialize_metrics_csv(path)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                metric.write_epoch(path, 0)
            self.assertIn('class=ALL accepted=0 correct=0 accuracy=N/A', out.getvalue())

    def test_duplicate_missing_and_invalid_inputs_fail(self):
        metric = PseudoLabelMetrics([1, 2])
        with self.assertRaises(ValueError):
            metric.update(np.array([0, 0]), np.array([1, 1]), masks([1, 1]))
        with self.assertRaises(ValueError):
            metric.update(np.array([2]), np.array([1]), masks([1]))
        with self.assertRaises(ValueError):
            metric.update(np.array([0]), np.array([7]), masks([1]))
        metric.update(np.array([0]), np.array([1]), masks([1]))
        with self.assertRaises(ValueError):
            metric.update(np.array([0]), np.array([1]), masks([1]))
        with self.assertRaises(ValueError):
            metric.write_epoch('unused.csv', 0)

    def test_diagnostics_do_not_change_rng_or_inputs(self):
        ids, pred = np.array([1, 0]), np.array([2, 1])
        stage_masks = masks([1, 0])
        np.random.seed(123)
        before = np.random.get_state()
        metric = PseudoLabelMetrics([1, 2])
        metric.update(ids, pred, stage_masks)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(ids, [1, 0])
        np.testing.assert_array_equal(pred, [2, 1])
        np.testing.assert_array_equal(stage_masks['supervised'], [True, False])

    def test_training_hooks_use_actual_supervision_mask(self):
        source = Path('train.py').read_text()
        ast.parse(source)
        self.assertIn('PseudoLabelMetrics(target_train.label)', source)
        self.assertIn("'supervised': learn_mask", source)
        self.assertLess(source.index('learn_mask = supervision_mask('), source.index('pseudo_metrics.update('))
        self.assertLess(source.index('pseudo_metrics.update('), source.index('# CAST-equivalent CE:'))
        # Ground truth remains ignored by the training loop; diagnostics use IDs.
        self.assertIn('for weak1, weak2, strong, _, sample_ids in train_loader_target:', source)


if __name__ == '__main__':
    unittest.main()
