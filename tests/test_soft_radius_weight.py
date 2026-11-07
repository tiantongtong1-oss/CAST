import ast
import itertools
from pathlib import Path
import unittest


class SoftWeightTests(unittest.TestCase):
    def test_production_mask_and_weight_assignments(self):
        # Execute the actual production assignments with scalar tensor-mask analogues.
        source = (Path(__file__).resolve().parents[1] / 'train.py').read_text()
        tree = ast.parse(source)
        targets = {'radius_candidates', 'fusion_candidates', 'outside_high_confidence'}
        assignments = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                name = getattr(node.targets[0], 'id', None)
                if name in targets and isinstance(node.value, ast.BinOp):
                    assignments[name] = compile(ast.Expression(node.value), '<production>', 'eval')
        for confidence, agree, radius, final in itertools.product((0, 1), repeat=4):
            scope = dict(confidence_mask=confidence, agreement_mask=agree,
                         radius_pass_mask=radius, final_mask=final)
            scope['radius_candidates'] = eval(assignments['radius_candidates'], {}, scope)
            candidate = eval(assignments['fusion_candidates'], {}, scope)
            self.assertEqual(bool(candidate), bool(confidence or (agree and radius)))
            outside = eval(assignments['outside_high_confidence'], {}, scope)
            self.assertEqual(bool(outside), bool(final and confidence and not radius))
        self.assertIn('criterion(output[0], train_targets) * train_con_idx * cls_weights', source)
        self.assertIn('cls_loss = per_sample_loss.sum() / train_con_idx.sum().clamp_min(1.0)', source)
        self.assertIn('con_idx = final_mask.float()', source)
        self.assertIn('(source_imgs, strong.cpu())', source)

    def test_fixed_denominator_and_source_contribution(self):
        ce = [2., 3., 5., 7.]
        mask = [1., 1., 1., 0.]
        baseline = sum(x*m for x,m in zip(ce, mask)) / sum(mask)
        weighted = sum(x*m*w for x,m,w in zip(ce, mask, [1., 1., .5, 1.])) / sum(mask)
        self.assertAlmostEqual(baseline - weighted, 2.5 / 3)
        self.assertAlmostEqual(2. / sum(mask), 2. / 3)


if __name__ == '__main__':
    unittest.main()
