import ast
import contextlib
import io
from pathlib import Path
import unittest
import numpy as np
from batch_progress import BatchProgress


class PerformanceTests(unittest.TestCase):
    def test_progress_every_twenty_and_final_batch(self):
        now = [0.]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            progress = BatchProgress('Target', 2, 45, clock=lambda: now[0])
            for batch in range(1, 46):
                now[0] += 2
                progress.mark('data')
                now[0] += 3
                progress.mark('train')
                progress.report(batch)
        text = output.getvalue()
        self.assertIn('Batch 20/45', text)
        self.assertIn('Batch 40/45', text)
        self.assertIn('Batch 45/45', text)
        self.assertNotIn('Batch 19/45', text)
        self.assertIn('ETA(epoch): 125.0s', text)
        self.assertIn('ETA(epoch): 0.0s', text)
        self.assertIn('data=40.00s train=60.00s', text)

    def test_actual_csv_loop_uses_only_cpu_values(self):
        tree = ast.parse(Path('train.py').read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                    and isinstance(n.iter, ast.Call)
                    and isinstance(n.iter.func, ast.Name) and n.iter.func.id == 'enumerate'
                    and isinstance(n.iter.args[0], ast.Name) and n.iter.args[0].id == 'query_labels')
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ('item', 'cpu', 'to')
                             for n in ast.walk(loop)))
        labels = np.array([1, 0])
        source = dict(count=np.array([11, 11]), available=np.array([True, True]),
                      mass=np.array([[.2,.8],[.7,.3]]), uncertainty=np.zeros(2), conflict=np.zeros(2))
        result = dict(support=np.array([.8,.7]), margin=np.array([.6,.4]), uncertainty=np.zeros(2),
                      pass_mask=np.array([True,True]), vote_count=np.array([9,8]))
        comp = dict(probability_gap=np.array([.2,.3]), prototype_gap=np.array([.1,.2]),
                    vote_gap=np.array([7,5]), pass_mask=np.array([True,False]))
        env = dict(i=3, query_labels=labels, query_ids=np.array([8,19]), source_ds=source,
                   ds_result=result, competition=comp, accepted_cpu=np.array([True,False]),
                   radius_cpu=np.array([False,True]), vote_cpu=np.array([True,True]), rows=[])
        exec(compile(ast.Module(body=[loop], type_ignores=[]), '<csv-loop>', 'exec'),env)
        rows = env['rows']
        columns = next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                       and any(isinstance(t,ast.Name) and t.id == 'ds_columns' for t in n.targets))
        self.assertEqual(len(rows[0]), len(ast.literal_eval(columns)))
        self.assertEqual(rows[0][:3], [3,8,1])
        self.assertEqual(rows[0][12:17], [True,9,.8,False,True])
        self.assertFalse(rows[1][12])

    def test_training_concat_preserves_order_without_host_copy(self):
        tree = ast.parse(Path('train.py').read_text())
        names = {'train_imgs','train_targets','train_anchor_idx'}
        nodes = [n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
        self.assertEqual(len(nodes),3)
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('cpu','to')
                             for node in nodes for n in ast.walk(node)))
        class Torch:
            @staticmethod
            def cat(parts, dim=0): return np.concatenate(parts,axis=dim)
        env=dict(torch=Torch, source_imgs=np.ones((3,2)), strong=np.full((5,2),2),
                 source_targets=np.array([0,1,2]), pseudo_targets=np.array([3,4,5,6,0]),
                 source_con_idx=np.ones(3), anchor_idx=np.array([0,1,0,1,0]))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<concat>','exec'),env)
        self.assertEqual(env['train_imgs'].shape,(8,2))
        self.assertEqual(env['train_targets'].tolist(),[0,1,2,3,4,5,6,0])
        self.assertEqual(env['train_anchor_idx'].tolist(),[1,1,1,0,1,0,1,0])


if __name__ == '__main__': unittest.main()
