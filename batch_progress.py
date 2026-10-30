"""Low-overhead wall-clock progress for training, without per-stage CUDA barriers."""
import time
from collections import defaultdict


class BatchProgress:
    def __init__(self, stage, epoch, total, interval=20, clock=None):
        self.stage, self.epoch, self.total = stage, epoch, total
        self.interval = interval
        self.clock = clock or time.perf_counter
        self.started = self.last_report = self.last_mark = self.clock()
        self.previous_batch = 0
        self.phases = defaultdict(float)
        print('[%s Epoch %d] Progress: 0/%d batches started' %
              (stage, epoch, total), flush=True)

    def mark(self, phase):
        now = self.clock()
        self.phases[phase] += now - self.last_mark
        self.last_mark = now

    def report(self, batch):
        if batch % self.interval and batch != self.total:
            return
        now = self.clock()
        elapsed = now - self.started
        window = now - self.last_report
        count = batch - self.previous_batch
        eta = elapsed / max(batch, 1) * max(self.total - batch, 0)
        phases = ' '.join('%s=%.2fs' % item for item in self.phases.items())
        print('[%s Epoch %d] Batch %d/%d Elapsed: %.1fs Window: %.1fs '
              'Avg: %.3fs/batch ETA(epoch): %.1fs PhaseWall: %s' %
              (self.stage, self.epoch, batch, self.total, elapsed, window,
               window / max(count, 1), eta, phases), flush=True)
        self.previous_batch = batch
        self.phases.clear()
        # Exclude printing itself from the next phase; elapsed includes it.
        self.last_mark = self.last_report = self.clock()
