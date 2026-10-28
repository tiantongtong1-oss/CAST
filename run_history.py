"""Per-invocation experiment archives; stdlib only, no training decisions."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import traceback
import uuid


BASELINE_SHA = '37f0d8e55f56ce41de68f0884fa0d04ee4db4996'
BASELINE_BRANCH = 'experiment/global-mean-std-cast-ce-v27'


def now():
    return datetime.now(timezone(timedelta(hours=8)))


def git(repo, *args):
    try:
        result = subprocess.run(['git', '-C', str(repo), *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=30)
        return result.stdout.rstrip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def branch_slug(branch):
    value = re.sub(r'[^A-Za-z0-9._-]', '_', branch.replace('/', '__'))
    return value.strip('._-')[:160] or 'unknown-branch'


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


class RunHistory:
    def __init__(self, root, args, repo=None):
        self.root = Path(root).expanduser().resolve()
        self.args = vars(args).copy()
        self.repo = Path(repo or Path(__file__).parent).resolve()
        self.path = None

    def _save_metadata(self):
        temporary = self.path / 'run.json.tmp'
        temporary.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(self.path / 'run.json')

    def __enter__(self):
        branch = git(self.repo, 'symbolic-ref', '--short', 'HEAD')
        commit = git(self.repo, 'rev-parse', 'HEAD')
        branch = branch or ('detached-' + commit[:12] if commit else 'unknown-branch')
        started = now()
        self.root.mkdir(parents=True, exist_ok=True)
        # Atomic mkdir and random suffix also protect simultaneous invocations.
        for _ in range(10):
            self.path = self.root / (branch_slug(branch) + '__' +
                                    started.strftime('%Y%m%d-%H%M%S-%f') + '__' + uuid.uuid4().hex[:6])
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError('unable to allocate unique run directory')
        (self.path / 'logs').mkdir()
        (self.path / 'models').mkdir()
        baseline = git(self.repo, 'rev-parse', '--verify', BASELINE_SHA + '^{commit}')
        committed_diff = git(self.repo, 'diff', '--no-ext-diff', '--no-textconv', BASELINE_SHA, 'HEAD', '--') if baseline else None
        dirty_diff = git(self.repo, 'diff', '--no-ext-diff', '--no-textconv', 'HEAD', '--') if commit else None
        commits = git(self.repo, 'log', '--oneline', BASELINE_SHA + '..HEAD') if baseline else None
        stat = git(self.repo, 'diff', '--stat', BASELINE_SHA, 'HEAD', '--') if baseline else None
        # Only list untracked source files; never recursively inventory run data.
        untracked = git(self.repo, 'ls-files', '--others', '--exclude-standard')
        self.metadata = dict(branch=branch, commit=commit, baseline_branch=BASELINE_BRANCH,
                             baseline_commit=BASELINE_SHA if baseline else None,
                             started_at=started.isoformat(), status='running',
                             command=shlex.join([sys.executable] + sys.argv),
                             cwd=str(Path.cwd()), repository=str(self.repo),
                             args=self.args, run_directory=str(self.path),
                             input_checkpoint=self.args.get('checkpoint'),
                             untracked_files=(untracked or '').splitlines())
        self._save_metadata()
        (self.path / 'branch_changes.patch').write_text(
            committed_diff if committed_diff is not None else '# Baseline unavailable; no comparison produced.\n', encoding='utf-8')
        (self.path / 'working_tree.patch').write_text(dirty_diff or '', encoding='utf-8')
        report = ('# 本次运行与分支修改记录\n\n'
                  '- 分支：`%s`\n- 提交：`%s`\n- 开始时间：%s\n'
                  '- 比较基线：`%s`（`%s`）\n\n'
                  '## 本分支新增功能\n\n'
                  '每次训练创建独立目录；自动保存 stdout/stderr 日志、模型检查点、'
                  '运行参数、命令、Git 版本和修改记录。训练筛选与损失保持 v27 设置。\n\n'
                  '## 相对基线的提交\n\n```text\n%s\n```\n\n'
                  '## 文件变更统计\n\n```text\n%s\n```\n\n'
                  '完整已提交差异见 branch_changes.patch；未提交的已跟踪文件差异见 '
                  'working_tree.patch。未跟踪文件仅在 run.json 中列出，不保存其内容。\n'
                  % (branch, commit or 'unavailable', started.isoformat(), BASELINE_BRANCH,
                     baseline or 'unavailable', commits if commits is not None else 'Git/base unavailable',
                     stat if stat is not None else 'Git/base unavailable'))
        (self.path / 'BRANCH_CHANGES.md').write_text(report, encoding='utf-8')
        self._log = (self.path / 'logs' / 'train.log').open('a', encoding='utf-8', buffering=1)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(self._stdout, self._log), Tee(self._stderr, self._log)
        print('Run_History:', self.path, flush=True)
        print('Branch:', branch, 'Commit:', commit or 'unavailable', flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None:
                traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
            self.metadata.update(finished_at=now().isoformat(),
                                 status=('interrupted' if exc_type is KeyboardInterrupt else
                                         'failed' if exc_type is not None else 'completed'))
            if exc_type is not None:
                self.metadata['error'] = '%s: %s' % (exc_type.__name__, exc)
            self._save_metadata()
        finally:
            sys.stdout, sys.stderr = self._stdout, self._stderr
            self._log.close()
        return False
