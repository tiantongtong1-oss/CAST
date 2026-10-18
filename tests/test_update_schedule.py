from types import SimpleNamespace
import unittest
from update_schedule import batches_for_window, schedule_metadata


class Loader:
    def __init__(self, n):
        self.n, self.passes, self.reads = n, 0, 0

    def __iter__(self):
        self.passes += 1
        for i in range(self.n):
            self.reads += 1
            yield (i, self.reads)


class UpdateScheduleTests(unittest.TestCase):
    def test_small_dataset_restarts_and_resamples_without_cache(self):
        loader = Loader(2)
        rows = list(batches_for_window(loader, 5))
        self.assertEqual([r[0][0] for r in rows], [0, 1, 0, 1, 0])
        self.assertEqual([r[0][1] for r in rows], [1, 2, 3, 4, 5])
        self.assertEqual(loader.passes, 3)
        continuation = list(batches_for_window(loader, 2, rows[-1][1]))
        self.assertEqual([r[0][0] for r in continuation], [1, 0])

    def test_fixed_budget_identical_for_different_target_sizes(self):
        args = SimpleNamespace(target_steps_per_round=225, epochs=30,
                               rescue_js_start_epoch=20,
                               radius_rescue_warmup_epochs=3,
                               proto_warmup_epochs=3, proto_ramp_epochs=5,
                               source_refresh_interval=1)
        a, b = schedule_metadata(args, 2), schedule_metadata(args, 225)
        self.assertEqual(a, b)
        self.assertEqual(a['total_updates'], 6750)
        self.assertEqual(a['js_start_update'], 4500)
        self.assertEqual(a['prototype_start_update'], 675)
        self.assertEqual(a['prototype_ramp_updates'], 1125)
        args.target_steps_per_round = 0
        self.assertEqual(schedule_metadata(args, 2)['total_updates'], 60)

    def test_empty_loader_fails(self):
        with self.assertRaises(ValueError):
            list(batches_for_window(Loader(0), 1))


if __name__ == '__main__':
    unittest.main()
