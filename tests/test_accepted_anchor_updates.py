import ast
from pathlib import Path
import unittest
import numpy as np


class AcceptedAnchorUpdateTests(unittest.TestCase):
    def test_live_training_mask_blocks_unaccepted_anchors(self):
        tree = ast.parse(Path('train.py').read_text())
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'prototype_update_mask'
                               for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        anchors = np.array([False, False, True, True])
        accepted = np.array([False, True, False, True])
        env = dict(anchor_mask=anchors.copy(), final_mask=accepted.copy())
        module = ast.Module(body=assignments, type_ignores=[])
        exec(compile(module, '<actual-update-mask>', 'exec'), env)
        np.testing.assert_array_equal(env['prototype_update_mask'], [False, False, False, True])
        np.testing.assert_array_equal(env['anchor_mask'], anchors)
        np.testing.assert_array_equal(env['final_mask'], accepted)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'update_target']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[2].id, 'prototype_update_mask')


if __name__ == '__main__':
    unittest.main()
