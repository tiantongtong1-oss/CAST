"""Target-only evidence, refresh gates, and high-confidence anchor lifecycle."""
import ast
from pathlib import Path
import unittest
import numpy as np
from dual_domain_ds import DSConfig, EvidenceBank, audit_target
from target_anchor_ds import anchor_reference_scores


class TargetAnchorTest(unittest.TestCase):
    def test_refresh_rejects_changed_label_low_confidence_outside_radius_and_invalid_quality(self):
        score = anchor_reference_scores(
            [0, 1, 0, 0, 0, 0, 0], [.9, .9, .79, .9, np.nan, .9, .9],
            [True, True, True, False, True, True, True],
            [0]*7, [.88, .88, .88, .88, .88, 0, np.nan], [.8, .8])
        np.testing.assert_allclose(score, [.88, 0, 0, 0, 0, 0, 0])

    def test_current_quality_is_capped_by_history(self):
        score = anchor_reference_scores([0,0], [.85,.99], [True,True], [0,0], [.95,.90], [.8])
        np.testing.assert_allclose(score,[.85,.9])

    def test_empty_reference_and_self_exclusion_prevent_rescue(self):
        cfg = DSConfig(k=3,min_neighbors=3)
        empty = EvidenceBank(np.empty((0,2)), [], [], [], 2)
        r=empty.evaluate([[1.,0.]], [0], cfg, exclude_self=True)
        self.assertFalse(audit_target(r,[0],cfg)['pass_mask'][0])
        bank=EvidenceBank(np.tile([1.,0.],(3,1)),[0]*3,[0,1,2],[.9]*3,2)
        r=bank.evaluate([[1.,0.]], [0], cfg, exclude_self=True)
        self.assertEqual(r['count'][0],2)
        self.assertFalse(audit_target(r,[0],cfg)['pass_mask'][0])
        r=bank.evaluate([[1.,0.]], [99], cfg, exclude_self=True)
        self.assertTrue(audit_target(r,[0],cfg)['pass_mask'][0])

    def test_reference_bank_is_snapshot(self):
        x=np.tile([1.,0.],(3,1));labels=np.zeros(3,dtype=int);quality=np.ones(3)
        b=EvidenceBank(x,labels,np.arange(3),quality,2)
        x[:]=0;labels[:]=1;quality[:]=0
        self.assertTrue((b.labels==0).all())
        self.assertTrue((b.quality==1).all())
        self.assertTrue((b.x[:,0]==1).all())

    def test_training_only_uses_target_bank_and_high_confidence_eligibility(self):
        s=Path('train.py').read_text();tree=ast.parse(s)
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertIn('audit_target',calls)
        self.assertNotIn('audit_source',calls)
        self.assertNotIn('combine_domains',calls)
        self.assertNotIn('refresh_ds_source_bank',calls)
        refresh=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_ds_target_bank')
        truth_reads=[n for n in ast.walk(refresh) if isinstance(n,ast.Name) and n.id=='_ignored_target_truth' and isinstance(n.ctx,ast.Load)]
        self.assertEqual(truth_reads,[])
        observe=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='observe')
        self.assertEqual(ast.unparse(observe.args[2]),'confidence_mask & region_ok')
        evaluate=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='evaluate')
        self.assertIn('exclude_self=True',ast.unparse(evaluate))
        self.assertLess(s.index('refresh_source_radius_bank(\n                teacher'),s.index('ds_target, revoked = refresh_ds_target_bank'))
        self.assertIn('role_tracker.streak[invalid_ids] = 0',s)
        self.assertIn('final_mask = confidence_mask | rescue_mask',s)


try:
    import torch
    from reliability_roles import ReliabilityRoles
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch unavailable; run lifecycle tests in training environment')
class AnchorLifecycleTest(unittest.TestCase):
    def test_three_high_confidence_epochs_and_rescue_cannot_promote(self):
        r=ReliabilityRoles(2,3);ids=torch.tensor([0,1]);labels=torch.tensor([0,0]);conf=torch.tensor([.9,.7])
        for epoch in range(3):
            result=r.observe(ids,labels,torch.tensor([True,False]),conf,epoch)
        self.assertEqual(result.tolist(),[True,False])
        self.assertEqual(r.streak.tolist(),[3,0])
        result=r.observe(ids,labels,torch.tensor([True,False]),conf,2)
        self.assertEqual(r.streak.tolist(),[3,0])

    def test_label_change_or_failed_observation_restarts_promotion(self):
        r=ReliabilityRoles(1,3);ids=torch.tensor([0]);conf=torch.tensor([.9])
        for epoch in range(3):
            r.observe(ids,torch.tensor([0]),torch.tensor([True]),conf,epoch)
        self.assertFalse(r.observe(ids,torch.tensor([1]),torch.tensor([True]),conf,3).item())
        self.assertEqual(r.streak.item(),1)
        r.observe(ids,torch.tensor([1]),torch.tensor([False]),conf,4)
        self.assertEqual(r.streak.item(),0)

    def test_refresh_revalidates_and_revokes_without_target_truth(self):
        tree=ast.parse(Path('train.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_ds_target_bank')
        namespace=dict(torch=torch,np=np,EvidenceBank=EvidenceBank,anchor_reference_scores=anchor_reference_scores)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<refresh>','exec'),namespace)
        class Hook:
            output=None
        hook=Hook()
        class Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            def forward(self,imgs,*args,**kwargs):
                hook.output=torch.nn.functional.normalize(imgs,dim=1)
                # The second coordinate encodes low confidence for sample 2.
                logits=torch.tensor([[0.,5.] if z[0]<0 else ([.1,0.] if z[1]>.5 else [5.,0.]) for z in imgs])
                return logits,None
        class Radius:
            feature_dim=2
            def in_distribution(self,features,labels):
                return features[:,1] < .3
        r=ReliabilityRoles(5,3);r.streak[:]=3;r.labels[:]=0;r.last_epoch[:]=2;r.quality[:]=.9
        dataset=[(torch.tensor(z),999,i) for i,z in enumerate([[1.,0.],[-1.,0.],[1.,1.],[1.,.4],[1.,0.]])]
        bank,revoked=namespace['refresh_ds_target_bank'](Teacher(),hook,dataset,r,Radius(),torch.tensor([.8,.8]),3,5,0,2)
        self.assertEqual(bank.ids.tolist(),[0,4])
        self.assertEqual(revoked,3)
        self.assertEqual(r.streak.tolist(),[3,0,0,0,3])
        np.testing.assert_allclose(bank.quality,[.9,.9])


if __name__=='__main__':
    unittest.main()
