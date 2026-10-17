import argparse
import ast
from pathlib import Path
import unittest
import numpy as np
from dual_domain_ds import DSConfig
from rescue_js import rescue_js_gate


class RescueJSTest(unittest.TestCase):
    def test_identical_and_different_views(self):
        js,passed=rescue_js_gate([[.9,.1],[1.,0.]],[[.9,.1],[0.,1.]],.01)
        np.testing.assert_allclose(js,[0,np.log(2)])
        self.assertEqual(passed.tolist(),[True,False])

    def test_inclusive_boundary(self):
        _,passed=rescue_js_gate([[1,0]],[[0,1]],np.log(2))
        self.assertTrue(passed[0])
        _,passed=rescue_js_gate([[.5,.5]],[[.5,.5]],0)
        self.assertTrue(passed[0])

    def test_invalid_threshold(self):
        for threshold in [-1,.7,np.nan,np.inf]:
            with self.assertRaises(ValueError):
                rescue_js_gate([[1,0]],[[1,0]],threshold)

    def parse(self,args):
        tree=ast.parse(Path('train.py').read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['parse_args','ds_config_from_args']]
        env=dict(np=np,argparse=argparse,DSConfig=DSConfig)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<args>','exec'),env)
        return env['parse_args'](args)

    def test_opt_in_and_cli_value(self):
        self.assertIsNone(self.parse([]).rescue_js_threshold)
        self.assertEqual(self.parse(['--rescue_js_threshold','0.01']).rescue_js_threshold,.01)

    def test_train_wiring_keeps_high_confidence_or_and_csv_shape(self):
        text=Path('train.py').read_text();tree=ast.parse(text)
        self.assertIn('rescue_mask &= js_mask',text)
        self.assertIn('final_mask = confidence_mask | rescue_mask',text)
        self.assertIn('ds_pass_tensor & js_mask[query_idx]',text)
        self.assertIn('sample_ids, pseudo_targets, final_mask & region_ok',text)
        cols=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='ds_columns' for t in n.targets))
        row=next(n.args[0] for n in ast.walk(tree) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
                 and n.func.value.id=='rows' and n.func.attr=='append')
        self.assertEqual(len(cols.elts),len(row.elts))


if __name__=='__main__':
    unittest.main()

