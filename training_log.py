"""Mirror Python training output and tracebacks to a unique log per run."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import os
import shlex
import sys
import traceback


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def training_log():
    directory = Path(__file__).resolve().parent / 'logs'
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    path = directory / ('global_medoid_%s_%s.log' % (stamp, os.getpid()))
    stdout, stderr = sys.stdout, sys.stderr
    with path.open('x', encoding='utf-8', buffering=1) as logfile:
        sys.stdout, sys.stderr = Tee(stdout, logfile), Tee(stderr, logfile)
        try:
            print('Training log: %s' % path)
            print('Started: %s' % datetime.now().astimezone().isoformat())
            print('Command: %s' % shlex.join([sys.executable] + sys.argv))
            print('CUDA_VISIBLE_DEVICES=%s' % os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)'))
            yield path
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = stdout, stderr

