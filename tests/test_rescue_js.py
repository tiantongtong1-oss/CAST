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


if __name__ == '__main__':
    unittest.main()
