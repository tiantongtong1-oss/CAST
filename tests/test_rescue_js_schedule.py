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


if __name__ == '__main__':
    unittest.main()
