import argparse
import ast
import contextlib
import io
from pathlib import Path
import unittest
import numpy as np
from dual_domain_ds import DSConfig
from rescue_js import rescue_js_enabled


class JSScheduleTest(unittest.TestCase):
    def test_schedule_boundary_and_default(self):
        self.assertFalse(rescue_js_enabled(.01,0,20))
        self.assertFalse(rescue_js_enabled(.01,19,20))
        self.assertTrue(rescue_js_enabled(.01,20,20))
        self.assertTrue(rescue_js_enabled(.01,29,20))
        self.assertTrue(rescue_js_enabled(.01,0))
        self.assertTrue(rescue_js_enabled(0.,20,20))

    def test_omitted_threshold_remains_disabled(self):
        for epoch in [0,19,20,29]:
            self.assertFalse(rescue_js_enabled(None,epoch,20))

    def test_explicit_ablation_overrides_late_js(self):
        args = self.parse(['--rescue_js_threshold', '.01',
                           '--rescue_js_start_epoch', '20', '--no_rescue_js'])
        self.assertIsNone(args.rescue_js_threshold)
        for epoch in [0, 19, 20, 29]:
            self.assertFalse(rescue_js_enabled(args.rescue_js_threshold, epoch,
                                              args.rescue_js_start_epoch))

    def parse(self,argv):
        tree=ast.parse(Path('train.py').read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('parse_args','ds_config_from_args')]
        env=dict(argparse=argparse,np=np,DSConfig=DSConfig)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<args>','exec'),env)
        return env['parse_args'](argv)

    def test_cli_start_and_negative(self):
        self.assertEqual(self.parse([]).rescue_js_start_epoch,0)
        a=self.parse(['--rescue_js_threshold','.01','--rescue_js_start_epoch','20'])
        self.assertEqual(a.rescue_js_start_epoch,20)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse(['--rescue_js_start_epoch','-1'])

    def test_disabled_batch_skips_js_computation(self):
        # Execute the actual conditional block; no tensor operation should occur.
        tree=ast.parse(Path('train.py').read_text())
        nodes=list(ast.walk(tree))
        compute=next(n for n in nodes if isinstance(n,ast.If) and ast.unparse(n.test)=='enable_js'
                     and any(isinstance(z,ast.Call) and isinstance(z.func,ast.Name)
                             and z.func.id=='rescue_js_gate' for z in ast.walk(n)))
        def forbidden(*args):
            raise AssertionError('JS must not run before its start epoch')
        env=dict(enable_js=False,rescue_js_gate=forbidden)
        exec(compile(ast.Module(body=[compute],type_ignores=[]),'<batch>','exec'),env)
        self.assertNotIn('js_pass',env)

    def test_csv_divergence_uses_enabled_state(self):
        tree=ast.parse(Path('train.py').read_text())
        row=next(n.args[0] for n in ast.walk(tree) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
                 and n.func.value.id=='rows' and n.func.attr=='append')
        blanks=[n for n in row.elts if isinstance(n,ast.IfExp)]
        self.assertEqual(len(blanks),1)
        self.assertEqual(ast.unparse(blanks[0].test),'enable_js')
        self.assertIsNone(blanks[0].orelse.value)
        self.assertIn('enable_js',[ast.unparse(n) for n in row.elts])
        self.assertIn('final_mask = confidence_mask | rescue_mask',Path('train.py').read_text())


if __name__=='__main__':
    unittest.main()

