import argparse
import ast
import contextlib
import io
from pathlib import Path
import unittest
import numpy as np
from dual_domain_ds import DSConfig
from source_class_weighting import make_source_class_weights, source_weighted_mean, mixed_source_target_ce


class SourceClassWeightTest(unittest.TestCase):
    def test_inverse_sqrt_ratio_and_normalization(self):
        w=make_source_class_weights([100,400],.5)
        self.assertAlmostEqual(float(w[0]/w[1]),2.)
        self.assertAlmostEqual(float(w.mean()),1.)
        np.testing.assert_allclose(make_source_class_weights([100,400],0),1.)

    def test_missing_classes_and_bad_inputs(self):
        w=make_source_class_weights([100,0,400],.5)
        self.assertEqual(w[1],0)
        self.assertAlmostEqual(float(w[[0,2]].mean()),1.)
        for counts,power in [([0,0],.5),([-1,3],.5),([1.5,2],.5),([1,np.nan],.5),([1,2],-1),([1,2],np.inf)]:
            with self.assertRaises(ValueError): make_source_class_weights(counts,power)

    def test_source_weighted_mean(self):
        loss=np.array([1.,3.]);labels=np.array([0,1]);w=np.array([2.,1.])
        self.assertAlmostEqual(source_weighted_mean(loss,labels,w),5/3)

    def test_mixed_target_unchanged_and_rejected_target_zero(self):
        loss=np.array([1.,3.,4.,999.]);labels=np.array([0,1,0,1]);mask=np.array([1.,1.,1.,0.]);w=np.array([2.,1.])
        result=mixed_source_target_ce(loss,labels,mask,2,w)
        self.assertAlmostEqual(result,(2*(5/3)+4)/3)
        delta=loss.copy();delta[2]+=1
        self.assertAlmostEqual(mixed_source_target_ce(delta,labels,mask,2,w)-result,1/3)
        delta=loss.copy();delta[3]+=100
        self.assertEqual(mixed_source_target_ce(delta,labels,mask,2,w),result)

    def test_uniform_disabled_and_no_accepted_targets(self):
        loss=np.array([1.,3.,4.]);labels=np.array([0,1,0]);mask=np.array([1.,1.,1.])
        for w in [None,np.ones(2)]:
            self.assertAlmostEqual(mixed_source_target_ce(loss,labels,mask,2,w),8/3)
        mask[-1]=0
        self.assertAlmostEqual(mixed_source_target_ce(loss,labels,mask,2,np.array([2.,1.])),5/3)

    def test_cli_and_no_balanced_sampling(self):
        text=Path('train.py').read_text();tree=ast.parse(text)
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['parse_args','ds_config_from_args']]
        env=dict(argparse=argparse,np=np,DSConfig=DSConfig)
        exec(compile(ast.Module(body=funcs,type_ignores=[]),'<args>','exec'),env)
        self.assertEqual(env['parse_args']([]).source_class_weight_power,0.)
        self.assertEqual(env['parse_args'](['--source_class_weight_power','.5']).source_class_weight_power,.5)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            env['parse_args'](['--source_class_weight_power','nan'])
        self.assertNotIn('WeightedRandomSampler',text)
        self.assertIn('source_train.label',text)
        self.assertIn('source_weighted_mean(criterion(output[0], targets)',text)
        self.assertIn('cls_loss = mixed_source_target_ce(',text)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch unavailable; run gradient check in training environment')
class GradientTest(unittest.TestCase):
    def test_target_gradient_unchanged(self):
        loss=torch.tensor([1.,3.,4.,9.],requires_grad=True)
        labels=torch.tensor([0,1,0,1]);mask=torch.tensor([1.,1.,1.,0.]);weights=torch.tensor([2.,1.])
        mixed_source_target_ce(loss,labels,mask,2,weights).backward()
        torch.testing.assert_close(loss.grad,torch.tensor([4/9,2/9,1/3,0.]))


if __name__=='__main__':
    unittest.main()
