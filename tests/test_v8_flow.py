import ast
from pathlib import Path
import unittest


class V8FlowTests(unittest.TestCase):
    def test_source_loss_and_selection(self):
        source = Path('train.py').read_text()
        tree = ast.parse(source)
        run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_training')
        loop = next(n for n in run.body if isinstance(n,ast.For) and
                    ast.unparse(n.iter)=='range(args.pre_epochs)')
        text = ast.unparse(loop)
        self.assertIn('args.source_margin_weight * source_margin_loss',text)
        self.assertIn('val_loader_target',text)
        self.assertIn('if val_acc > best_source_val_acc',text)
        self.assertIn('prototype_loader_source',text)
        self.assertNotIn('stratified_split',source)
        self.assertLess(source.index('if args.source_only:'),source.index('teacher = create_ema_teacher'))
        self.assertIn("os.path.join('./new_models'",source)

    def test_prototype_refresh_restores_rng(self):
        source=Path('train.py').read_text()
        tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_source_margin_prototypes')
        text=ast.unparse(fn)
        self.assertIn('torch.set_rng_state',text)
        self.assertIn('torch.cuda.set_rng_state_all',text)
        self.assertIn('model.train(was_training)',text)
