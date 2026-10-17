"""Source-only audit checks without training data, torch or CUDA."""
import ast
from pathlib import Path
import unittest
import numpy as np
from dual_domain_ds import DSConfig, EvidenceBank, audit_source


class SourceOnlyDSTest(unittest.TestCase):
    def setUp(self):
        self.cfg = DSConfig(k=3, min_neighbors=3, alpha=1.0)

    def evidence(self, labels, quality=1.):
        n = len(labels)
        bank = EvidenceBank(np.tile([1., 0.], (n, 1)), labels,
                            np.arange(n), np.full(n, quality), 2)
        return bank.evaluate([[1., 0.]], [0], self.cfg)

    def test_source_alone_passes_without_target_anchors(self):
        result = audit_source(self.evidence([0, 0, 0]), [0], self.cfg)
        self.assertTrue(result['pass_mask'][0])
        self.assertGreater(result['support'][0], .99)

    def test_opposite_label_missing_neighbors_conflict_and_weak_evidence_fail(self):
        for labels, quality in (([1, 1, 1], 1.), ([0, 0], 1.),
                                ([0, 1, 1], 1.), ([0, 0, 0], .01)):
            with self.subTest(labels=labels, quality=quality):
                self.assertFalse(audit_source(self.evidence(labels, quality), [0], self.cfg)['pass_mask'][0])

    def test_each_gate_and_boundaries(self):
        source = dict(mass=np.array([[.7, .2]]), uncertainty=np.array([.1]),
                      conflict=np.array([.2]), available=np.array([True]))
        cfg = DSConfig(support=.7, margin=.49, uncertainty=.1, conflict=.2)
        self.assertTrue(audit_source(source, [0], cfg)['pass_mask'][0])
        for key, value in (('support', .71), ('margin', .51), ('uncertainty', .09), ('conflict', .19)):
            with self.subTest(gate=key):
                c = dict(support=.7, margin=.49, uncertainty=.1, conflict=.2)
                c[key] = value
                self.assertFalse(audit_source(source, [0], DSConfig(**c))['pass_mask'][0])
        source['available'][0] = False
        self.assertFalse(audit_source(source, [0], cfg)['pass_mask'][0])

    def test_empty_queries_and_invalid_labels(self):
        source = dict(mass=np.empty((0, 2)), uncertainty=np.empty(0),
                      conflict=np.empty(0), available=np.empty(0, dtype=bool))
        self.assertEqual(audit_source(source, [], self.cfg)['pass_mask'].size, 0)
        with self.assertRaises(ValueError):
            audit_source(self.evidence([0, 0, 0]), [2], self.cfg)



if __name__ == '__main__':
    unittest.main()

