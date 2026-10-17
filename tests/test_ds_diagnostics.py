import unittest
import numpy as np
from dual_domain_ds import DSConfig, EvidenceBank
from ds_diagnostics import diagnose_query
from diagnose_ds import parse_args, config_from_args


class DiagnosticTest(unittest.TestCase):
    def bank(self, x, y):
        return EvidenceBank(x,y,np.arange(len(y)),np.ones(len(y)),2)

    def test_same_class_saturation_sweep(self):
        b=self.bank(np.tile([1.,0.],(11,1)),[0]*11)
        s,n=diagnose_query(b,[1,0],0,0,DSConfig(k=11))
        self.assertEqual(len(n),11)  # source ID 0 must not be excluded
        self.assertAlmostEqual(s['support_k1'],.95)
        self.assertAlmostEqual(s['unknown_k3'],.05**3)
        self.assertAlmostEqual(s['unknown_k11'],.05**11)
        self.assertEqual(s['weighted_support'],1.)
        self.assertFalse(s['available_k1'])
        self.assertTrue(s['available_k3'])
        self.assertIsNone(s['d_other'])

    def test_other_distance_searches_beyond_selected_and_radius(self):
        b=self.bank([[1,0]]*11+[[0,1]],[0]*11+[1])
        s,n=diagnose_query(b,[1,0],0,44,DSConfig(k=11))
        self.assertEqual(len(n),11)
        self.assertAlmostEqual(s['d_other'],2**.5,places=6)
        self.assertAlmostEqual(s['distance_gap'],2**.5,places=6)

    def test_disagreement_and_empty_neighborhood(self):
        b=self.bank([[1,0]]*3,[0,1,1])
        s,_=diagnose_query(b,[1,0],0,4,DSConfig(k=3))
        self.assertGreater(s['conflict_k3'],.5)
        self.assertFalse(s['available_k3'])
        s,n=diagnose_query(b,[-1,0],0,4,DSConfig(k=3))
        self.assertEqual(n,[])
        self.assertEqual(s['unknown_k3'],1)
        self.assertIsNone(s['weighted_support'])

    def test_strength_formula_and_neighbor_cap(self):
        b=self.bank([[1,.1]]*15,[0]*15)
        cfg=DSConfig(k=15)
        s,n=diagnose_query(b,[1,0],0,3,cfg)
        self.assertEqual(len(n),15)
        self.assertEqual(s['count_k15'],15)
        self.assertAlmostEqual(n[0]['strength'],min(np.exp(-.5*(n[0]['distance']/cfg.bandwidth)**2),.95))

    def test_cli_no_torch_needed(self):
        a=parse_args(['--checkpoint','source.pth'])
        self.assertEqual(config_from_args(a).k,11)
        self.assertEqual(a.max_samples,500)

    def test_no_optimizer_or_backward(self):
        import ast
        from pathlib import Path
        tree=ast.parse(Path('diagnose_ds.py').read_text())
        attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        self.assertNotIn('backward',attrs)
        self.assertNotIn('step',attrs)
        self.assertNotIn('optim',attrs)


if __name__=='__main__':
    unittest.main()
