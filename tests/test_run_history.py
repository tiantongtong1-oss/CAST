import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_history
from run_history import RunHistory, branch_slug


class RunHistoryTests(unittest.TestCase):
    def test_no_git_unique_runs_tee_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            root = Path(tmp) / 'model_history'
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(run_history, 'git', return_value=None), redirect_stdout(stdout), redirect_stderr(stderr):
                with RunHistory(root, SimpleNamespace(checkpoint=None), repo) as first:
                    print('training output')
                    print('warning output', file=sys.stderr)
                    (first.path / 'models' / 'source.pth').write_bytes(b'checkpoint')
                with RunHistory(root, SimpleNamespace(checkpoint=None), repo) as second:
                    pass
            self.assertNotEqual(first.path, second.path)
            log = (first.path / 'logs' / 'train.log').read_text()
            self.assertIn('training output', log)
            self.assertIn('warning output', log)
            self.assertIn('training output', stdout.getvalue())
            self.assertIn('warning output', stderr.getvalue())
            metadata = json.loads((first.path / 'run.json').read_text())
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(metadata['branch'], 'unknown-branch')
            self.assertTrue(metadata['started_at'].endswith('+08:00'))
            self.assertTrue((first.path / 'BRANCH_CHANGES.md').is_file())

    def test_failure_and_interrupt_preserve_traceback_and_restore_streams(self):
        for error, status in [(ValueError('test failure'), 'failed'), (KeyboardInterrupt(), 'interrupted')]:
            with tempfile.TemporaryDirectory() as tmp:
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    with self.assertRaises(type(error)):
                        with RunHistory(Path(tmp) / 'history', SimpleNamespace(), tmp) as run:
                            raise error
                    self.assertIs(sys.stdout, out)
                    self.assertIs(sys.stderr, err)
                metadata = json.loads((run.path / 'run.json').read_text())
                self.assertEqual(metadata['status'], status)
                self.assertIn(type(error).__name__, (run.path / 'logs' / 'train.log').read_text())

    def test_real_git_branch_committed_and_dirty_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            def git(*args):
                return subprocess.check_output(['git', '-C', tmp, *args], stderr=subprocess.DEVNULL, text=True).strip()
            git('init')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.invalid')
            (repo / '.gitignore').write_text('model_history/\n')
            (repo / 'code.py').write_text('x = 1\n')
            git('add', '.')
            git('commit', '-m', 'base')
            base = git('rev-parse', 'HEAD')
            git('checkout', '-b', 'experiment/archive-test')
            (repo / 'code.py').write_text('x = 2\n')
            git('commit', '-am', 'change code')
            (repo / 'code.py').write_text('x = 3\n')
            (repo / 'untracked.py').write_text('not copied\n')
            with patch.object(run_history, 'BASELINE_SHA', base), redirect_stdout(io.StringIO()):
                with RunHistory(repo / 'model_history', SimpleNamespace(), repo) as run:
                    pass
            self.assertTrue(run.path.name.startswith('experiment__archive-test__'))
            self.assertIn('+x = 2', (run.path / 'branch_changes.patch').read_text())
            self.assertIn('+x = 3', (run.path / 'working_tree.patch').read_text())
            self.assertIn('change code', (run.path / 'BRANCH_CHANGES.md').read_text())
            metadata = json.loads((run.path / 'run.json').read_text())
            self.assertEqual(metadata['untracked_files'], ['untracked.py'])

    def test_safe_directory_names(self):
        for branch in ('../../', 'experiment/test', 'a\\b', 'a:*?', ''):
            slug = branch_slug(branch)
            self.assertNotIn('/', slug)
            self.assertNotIn('\\', slug)
            self.assertNotIn(slug, ('', '.', '..'))

    def test_training_writes_only_into_run_and_keeps_base_training(self):
        source = Path('train.py').read_text()
        ast.parse(source)
        self.assertIn('with RunHistory(args.history_root, args) as history:', source)
        self.assertIn("model_path = os.path.join(run_directory, 'models')", source)
        self.assertIn("os.path.join(run_directory, 'logs', args.run_name", source)
        self.assertIn('shutil.copy2(args.checkpoint, source_best_path)', source)
        self.assertNotIn('PseudoLabelMetrics', source)
        self.assertIn('cls_loss = (', source)


if __name__ == '__main__':
    unittest.main()
