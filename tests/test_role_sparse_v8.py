import contextlib
import copy
import io
import math
import unittest

import torch
from torch import nn

from global_gaussian_knn_reliability import combine_rescue_masks
from sparse_reliable_knn import SparseReliableKNNBank
from reliability_roles import ReliabilityRoles, neighbor_soft_targets, soft_supervision_loss
from prototype_utils import PrototypeBank


def points(angles):
    return torch.tensor([[math.cos(a), math.sin(a)] for a in angles], dtype=torch.float32)


def bank_with_memory(angles, labels=None, sender_scores=None, score_mode='support', chunk=128):
    bank = SparseReliableKNNBank(2, 2, k=20, score_mode=score_mode, query_chunk_size=chunk)
    bank.class_means.copy_(torch.tensor([[1., 0.], [1., 0.]]))
    bank.global_var.fill_(0.04)
    bank.distribution_initialized.fill_(True)
    bank.sigma_initialized.fill_(True)
    n = len(angles)
    if labels is None:
        labels = torch.zeros(n, dtype=torch.long)
    bank.set_target_memory(points(angles), labels, torch.full((n,), 0.95), torch.arange(n))
    bank.set_sender_scores(torch.full((n,), 0.95) if sender_scores is None else sender_scores)
    return bank


def query(bank, angle=0., label=0, sample_id=99):
    return bank.gate(points([angle]), torch.tensor([label]), torch.tensor([sample_id]),
                     torch.tensor([True]), torch.tensor([0.8, 0.8]))


class RoleTests(unittest.TestCase):
    def test_rescued_sample_learns_immediately_but_updates_prototype_only_after_promotion(self):
        roles = ReliabilityRoles(1, promotion_epochs=3)
        prototypes = PrototypeBank(2, 2)
        ids, labels = torch.tensor([0]), torch.tensor([1])
        accepted, rescued = combine_rescue_masks(torch.tensor([False]), torch.tensor([True]),
                                                torch.tensor([True]), True)
        self.assertTrue(accepted.item())
        self.assertTrue(rescued.item())
        for epoch in range(3):
            anchor = roles.observe(ids, labels, accepted, torch.tensor([0.7]), epoch)
            prototypes.update_target(points([0.]), labels, anchor)
            self.assertEqual(anchor.item(), epoch == 2)
            self.assertEqual(prototypes.target_counts[1].item(), int(epoch == 2))
        self.assertGreater(roles.sender_scores(ids, labels, torch.tensor([0.75]), 3).item(), 0)

    def test_no_same_epoch_self_promotion_and_ineligible_revokes(self):
        roles = ReliabilityRoles(1, 2)
        ids, labels = torch.tensor([0]), torch.tensor([0])
        for _ in range(5):
            self.assertFalse(roles.observe(ids, labels, torch.tensor([True]), torch.tensor([0.9]), 0).item())
        self.assertEqual(roles.streak.item(), 1)
        self.assertTrue(roles.observe(ids, labels, torch.tensor([True]), torch.tensor([0.9]), 1).item())
        self.assertFalse(roles.observe(ids, labels, torch.tensor([False]), torch.tensor([0.9]), 1).item())
        self.assertEqual(roles.sender_scores(ids, labels, torch.tensor([0.9]), 2).item(), 0)

    def test_label_change_gap_and_snapshot_disagreement_reset_or_block(self):
        roles = ReliabilityRoles(1, 2)
        ids = torch.tensor([0])
        for epoch in range(2):
            roles.observe(ids, torch.tensor([0]), torch.tensor([True]), torch.tensor([0.9]), epoch)
        self.assertEqual(roles.sender_scores(ids, torch.tensor([1]), torch.tensor([0.99]), 2).item(), 0)
        self.assertEqual(roles.sender_scores(ids, torch.tensor([0]), torch.tensor([0.99]), 3).item(), 0)
        self.assertFalse(roles.observe(ids, torch.tensor([1]), torch.tensor([True]), torch.tensor([0.9]), 2).item())
        self.assertEqual(roles.streak.item(), 1)
        self.assertFalse(roles.observe(ids, torch.tensor([1]), torch.tensor([True]), torch.tensor([0.9]), 4).item())
        self.assertEqual(roles.streak.item(), 1)

    def test_history_roundtrip_and_invalid_ids(self):
        roles = ReliabilityRoles(2, 2)
        roles.observe(torch.tensor([1]), torch.tensor([0]), torch.tensor([True]), torch.tensor([0.8]), 0)
        restored = ReliabilityRoles(2, 2)
        restored.load_state_dict(copy.deepcopy(roles.state_dict()))
        self.assertTrue(restored.observe(torch.tensor([1]), torch.tensor([0]), torch.tensor([True]), torch.tensor([0.8]), 1).item())
        for ids in (torch.tensor([-1]), torch.tensor([2]), torch.tensor([0, 0])):
            with self.assertRaises(ValueError):
                restored.sender_scores(ids, torch.zeros_like(ids), torch.ones_like(ids).float(), 2)


class SparseSupportTests(unittest.TestCase):
    def test_default_sparse_coherent_support_can_pass(self):
        bank = bank_with_memory([0.44, 0.46, 0.48])
        result = query(bank)
        self.assertTrue(result['sparse_mask'].item())
        self.assertLess(result['density'].item(), 0.5)
        self.assertTrue(result['pass_mask'].item())
        self.assertAlmostEqual(result['score'].item(), 0.95, places=5)

    def test_v7_rejects_sparse_support_under_default_thresholds(self):
        bank = bank_with_memory([0.44] * 20, score_mode='v7')
        result = query(bank)
        self.assertTrue(result['sparse_mask'].item())
        self.assertFalse(result['pass_mask'].item())

    def test_dense_conflicting_neighbors_fail(self):
        bank = bank_with_memory([0.01, -0.01, 0.02, -0.02, 0.03, -0.03],
                                labels=torch.tensor([0, 1, 0, 1, 0, 1]))
        result = query(bank)
        self.assertTrue(result['dense_mask'].item())
        self.assertFalse(result['pass_mask'].item())
        self.assertFalse(result['neighbor_reliable'].item())

    def test_unreliable_senders_cannot_vote(self):
        bank = bank_with_memory([0.01, 0.02, 0.03], sender_scores=torch.tensor([0.95, 0., 0.]))
        result = query(bank)
        self.assertEqual(result['support_count'].item(), 1)
        self.assertFalse(result['pass_mask'].item())

    def test_effective_count_rejects_one_dominant_sender(self):
        bank = bank_with_memory([0.01, 0.02, 0.03], sender_scores=torch.tensor([0.99, 0.001, 0.001]))
        result = query(bank)
        self.assertEqual(result['support_count'].item(), 3)
        self.assertLess(result['effective_support'].item(), 2)
        self.assertFalse(result['pass_mask'].item())

    def test_far_coherent_neighbors_fail_distance_guard(self):
        bank = bank_with_memory([0.44, 0.46, 0.48])
        bank.radius_multiplier = 0.1
        self.assertFalse(query(bank)['pass_mask'].item())

    def test_self_exclusion_and_outside_source_region(self):
        bank = bank_with_memory([0.01, 0.02, 0.03])
        self.assertFalse(query(bank, sample_id=0)['pass_mask'].item())
        self.assertFalse(query(bank, angle=2.)['pass_mask'].item())

    def test_empty_memory_and_invalid_queries_fail_closed(self):
        bank = SparseReliableKNNBank(2, 2)
        self.assertFalse(query(bank)['pass_mask'].item())
        bank = bank_with_memory([0.01, 0.02, 0.03])
        result = bank.gate(torch.tensor([[float('nan'), 0.], [0., 0.]]),
                           torch.tensor([0, 99]), torch.tensor([99, 98]),
                           torch.tensor([True, True]), torch.tensor([0.8, 0.8]))
        self.assertFalse(result['pass_mask'].any())
        self.assertTrue(torch.isfinite(result['score']).all())

    def test_chunking_and_memory_order_do_not_change_results(self):
        bank = bank_with_memory([0.01, 0.02, 0.03, 0.04])
        args = (points([0., 0.1, 0.2]), torch.zeros(3, dtype=torch.long),
                torch.tensor([99, 98, 97]), torch.ones(3, dtype=torch.bool), torch.tensor([0.8, 0.8]))
        a = bank.gate(*args)
        bank.query_chunk_size = 1
        perm = torch.tensor([3, 1, 0, 2])
        for name in ('memory_features', 'memory_labels', 'memory_ids', 'memory_confidences', 'memory_sender_scores'):
            setattr(bank, name, getattr(bank, name)[perm])
        b = bank.gate(*args)
        for key in a:
            self.assertTrue(torch.allclose(a[key], b[key]), key)

    def test_refreshed_memory_does_not_inherit_old_sender_scores(self):
        bank = bank_with_memory([0.01, 0.02, 0.03])
        self.assertTrue(query(bank)['pass_mask'].item())
        bank.set_target_memory(points([0.01, 0.02, 0.03]), torch.zeros(3, dtype=torch.long),
                               torch.ones(3), torch.arange(3))
        self.assertFalse(query(bank)['pass_mask'].item())


class SoftRepairTests(unittest.TestCase):
    def test_conflict_is_soft_only_and_gradients_do_not_reach_teacher(self):
        bank = bank_with_memory([0.01, 0.02, 0.03], labels=torch.ones(3, dtype=torch.long))
        result = query(bank, label=0)
        self.assertFalse(result['pass_mask'].item())
        self.assertTrue(result['neighbor_reliable'].item())
        teacher = torch.tensor([[0.65, 0.35]], requires_grad=True)
        mask, targets = neighbor_soft_targets(teacher, result, torch.tensor([True]), torch.tensor([False]))
        self.assertTrue(mask.item())
        self.assertTrue(torch.allclose(targets, torch.tensor([[0.325, 0.675]])))
        logits = torch.zeros(1, 2, requires_grad=True)
        loss = soft_supervision_loss(logits, targets, mask)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertLess(logits.grad[0, 1].item(), 0)
        blocked, _ = neighbor_soft_targets(teacher, result, torch.tensor([True]), torch.tensor([True]))
        self.assertFalse(blocked.any())

    def test_empty_soft_loss_is_finite_and_differentiable(self):
        logits = torch.randn(2, 3, requires_grad=True)
        loss = soft_supervision_loss(logits, torch.zeros_like(logits), torch.zeros(2, dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
