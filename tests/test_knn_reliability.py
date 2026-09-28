import math
import unittest

import torch

from knn_reliability import KNNReliabilityBank, combine_rescue_masks


def points(angles):
    return torch.tensor([[math.cos(a), math.sin(a)] for a in angles], dtype=torch.float32)


def ready_bank(k=2, **kwargs):
    bank = KNNReliabilityBank(2, 2, k=k, score_threshold=0.6, **kwargs)
    bank.begin_source_refresh(torch.tensor([[1., 0.], [-1., 0.]]), torch.tensor([True, True]))
    # Independent fixed radial scale for synthetic geometry tests.
    bank.global_var.fill_(0.04)
    bank.sigma_initialized.fill_(True)
    return bank


def fill(bank, angles, labels=None, conf=None, ids=None):
    n = len(angles)
    bank.set_target_memory(
        points(angles), torch.tensor(labels if labels is not None else [0] * n),
        torch.tensor(conf if conf is not None else [0.99] * n),
        torch.tensor(ids if ids is not None else list(range(n))),
    )


def query(bank, angle=0., label=0, sample_id=999, candidate=True):
    return bank.gate(points([angle]), torch.tensor([label]), torch.tensor([sample_id]),
                     torch.tensor([candidate]), torch.tensor([0.8, 0.8]))


class ReliabilityTests(unittest.TestCase):
    def test_dense_vs_sparse_with_equal_class_support(self):
        dense, sparse = ready_bank(), ready_bank()
        fill(dense, [0.02, -0.02])
        fill(sparse, [0.25, -0.25])
        a, b = query(dense), query(sparse)
        self.assertEqual(a['support_fraction'].item(), 1.)
        self.assertEqual(b['support_fraction'].item(), 1.)
        self.assertGreater(a['score'].item(), b['score'].item())
        self.assertTrue(a['pass_mask'].item())
        self.assertFalse(b['pass_mask'].item())
        self.assertTrue(a['dense_mask'].item())
        self.assertFalse(b['dense_mask'].item())

    def test_close_other_class_cannot_support_and_search_is_not_class_filtered(self):
        bank = ready_bank()
        fill(bank, [0.01, -0.01, 0.1, -0.1], labels=[1, 1, 0, 0])
        result = query(bank)
        self.assertGreater(result['density'].item(), 0.9)
        self.assertEqual(result['score'].item(), 0.)

    def test_low_confidence_neighbors_remain_in_search_but_cannot_support(self):
        bank = ready_bank()
        fill(bank, [0.01, -0.01, 0.1, -0.1], conf=[0.2, 0.2, 0.99, 0.99])
        result = query(bank)
        self.assertGreater(result['density'].item(), 0.9)
        self.assertEqual(result['score'].item(), 0.)

    def test_self_excluded_by_id_not_feature_equality(self):
        bank = ready_bank(k=1)
        fill(bank, [0., 0.02], labels=[0, 1], ids=[42, 43])
        self.assertEqual(query(bank, sample_id=42)['score'].item(), 0.)
        self.assertGreater(query(bank, sample_id=999)['score'].item(), 0.99)
        # Another sample may legitimately have the same feature as the query.
        fill(bank, [0., 0.], ids=[42, 43])
        self.assertGreater(query(bank, sample_id=42)['score'].item(), 0.99)

    def test_query_and_neighbors_must_be_in_assigned_prototype_region(self):
        bank = ready_bank()
        fill(bank, [0.7, 0.71])
        result = query(bank, angle=0.7)
        self.assertGreater(result['density'].item(), 0.9)
        self.assertEqual(result['score'].item(), 0.)
        result = query(bank)
        self.assertEqual(result['support_fraction'].item(), 0.)

    def test_interval_lambda_changes_region_membership(self):
        tight = ready_bank(interval_lambda=1.)
        wide = ready_bank(interval_lambda=2.)
        fill(tight, [0.25, -0.25])
        fill(wide, [0.25, -0.25])
        self.assertEqual(query(tight)['score'].item(), 0.)
        self.assertGreater(query(wide)['score'].item(), 0.)

    def test_missing_statistics_and_insufficient_neighbors_fail_closed(self):
        bank = ready_bank()
        self.assertFalse(query(bank)['checked_mask'].item())
        fill(bank, [0., 0.01], ids=[10, 11])
        self.assertFalse(query(bank, sample_id=10)['pass_mask'].item())
        self.assertFalse(query(bank, sample_id=10)['checked_mask'].item())
        bank.sigma_initialized.fill_(False)
        self.assertFalse(query(bank)['pass_mask'].item())
        bank.sigma_initialized.fill_(True)
        bank.prototype_initialized[0] = False
        self.assertFalse(query(bank)['pass_mask'].item())

    def test_nonfinite_memory_and_queries_do_not_produce_nan_scores(self):
        bank = ready_bank(k=1)
        bank.set_target_memory(torch.tensor([[float('nan'), 0.], [1., 0.]]),
                               torch.tensor([0, 0]), torch.tensor([0.99, 0.99]),
                               torch.tensor([0, 1]))
        self.assertEqual(bank.memory_ids.tolist(), [1])
        result = bank.gate(torch.tensor([[float('nan'), 0.], [0., 0.]]),
                           torch.tensor([0, 0]), torch.tensor([10, 11]),
                           torch.tensor([True, True]), torch.tensor([0.8, 0.8]))
        self.assertTrue(torch.isfinite(result['score']).all())
        self.assertFalse(result['pass_mask'].any())

    def test_duplicate_memory_ids_rejected(self):
        with self.assertRaises(ValueError):
            fill(ready_bank(), [0., 0.01], ids=[3, 3])

    def test_source_variance_is_global_sample_weighted_and_ema_updated(self):
        bank = KNNReliabilityBank(2, 2, sigma_momentum=0.9)
        centers = torch.tensor([[1., 0.], [-1., 0.]])
        initialized = torch.tensor([True, True])
        bank.begin_source_refresh(centers, initialized)
        bank.accumulate_source(torch.tensor([[1., 0.]]), torch.tensor([0]))
        bank.accumulate_source(torch.tensor([[0., 1.], [-1., 0.]]), torch.tensor([0, 0]))
        bank.finalize_source()
        self.assertAlmostEqual(bank.global_var.item(), 2., places=6)
        self.assertEqual(bank.source_count.item(), 3)
        bank.begin_source_refresh(centers, initialized)
        bank.accumulate_source(torch.tensor([[1., 0.], [-1., 0.]]), torch.tensor([0, 1]))
        bank.finalize_source()
        self.assertAlmostEqual(bank.global_var.item(), 1.8, places=6)

    def test_centers_are_snapshots_and_memory_is_not_checkpointed(self):
        bank = ready_bank()
        fill(bank, [0., 0.01])
        state = bank.state_dict()
        self.assertNotIn('memory_features', state)
        restored = KNNReliabilityBank(2, 2, k=2)
        restored.load_state_dict(state)
        self.assertAlmostEqual(restored.sigma.item(), 0.2, places=6)
        self.assertFalse(query(restored)['pass_mask'].item())
        centers = torch.tensor([[1., 0.], [-1., 0.]])
        restored.begin_source_refresh(centers, torch.tensor([True, True]))
        centers.zero_()
        self.assertEqual(restored.prototypes[0, 0].item(), 1.)

    def test_score_matches_direct_reference_and_chunking_does_not_change_results(self):
        bank = ready_bank(query_chunk_size=1)
        fill(bank, [0.02, -0.04])
        z = points([0., 0.01, -0.02]).requires_grad_()
        result = bank.gate(z, torch.zeros(3, dtype=torch.long), torch.tensor([11, 12, 13]),
                           torch.ones(3, dtype=torch.bool), torch.tensor([0.8, 0.8]))
        expected = torch.exp(-((z[:, None, :] - bank.memory_features[None, :, :]) ** 2)
                             .sum(dim=2) / (2 * 0.04)).mean(dim=1)
        self.assertTrue(torch.allclose(result['score'], expected.detach(), atol=1e-5))
        self.assertFalse(result['score'].requires_grad)
        bank.query_chunk_size = 100
        other = bank.gate(z, torch.zeros(3, dtype=torch.long), torch.tensor([11, 12, 13]),
                          torch.ones(3, dtype=torch.bool), torch.tensor([0.8, 0.8]))
        self.assertTrue(torch.allclose(result['score'], other['score']))

    def test_or_rescue_warmup_and_disagreement(self):
        confidence = torch.tensor([True, False, False, False])
        agree = torch.tensor([True, True, False, True])
        support = torch.tensor([False, True, True, False])
        final, rescue = combine_rescue_masks(confidence, agree, support, True)
        self.assertEqual(final.tolist(), [True, True, False, False])
        self.assertEqual(rescue.tolist(), [False, True, False, False])
        final, rescue = combine_rescue_masks(confidence, agree, support, False)
        self.assertEqual(final.tolist(), confidence.tolist())
        self.assertFalse(rescue.any())
        self.assertEqual(set(final.float().tolist()), {0., 1.})


if __name__ == '__main__':
    unittest.main()
