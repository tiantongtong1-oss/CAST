"""Mirror console streams to one experiment log."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
import sys


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        return len(text)

    def flush(self):
        self.console.flush(); self.file.flush()


@contextmanager
def training_log(directory, name):
    if directory is None:
        yield
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / (name + '.log')
    # Do not silently replace results from a prior experiment.
    with path.open('x', encoding='utf-8', buffering=1) as file:
        with redirect_stdout(Tee(sys.stdout, file)), redirect_stderr(Tee(sys.stderr, file)):
            yield
