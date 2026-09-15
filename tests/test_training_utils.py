import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from training_utils import (balance_source_losses, make_loader_generators,
                            path_order_sha256, reset_target_rng, seed_everything,
                            seed_worker, source_class_weights)


class RandomDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index, torch.rand(1).item(), random.random(), np.random.rand()


class ReproducibilityTests(unittest.TestCase):
    def test_target_rng_is_independent_of_pretraining_consumption(self):
        generators = make_loader_generators(1314)

        def sample():
            return (torch.rand(4), random.random(), np.random.rand(),
                    torch.randperm(8, generator=generators['source']),
                    torch.randperm(8, generator=generators['target']))

        reset_target_rng(1314, generators)
        expected = sample()
        for _ in range(13):
            sample()
        reset_target_rng(1314, generators)
        actual = sample()
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b)

    def test_validation_loader_does_not_consume_training_rng(self):
        generators = make_loader_generators(1314)
        # Evaluation transforms are deterministic, including when workers=0.
        loader = torch.utils.data.DataLoader(list(range(8)), batch_size=2,
                                             generator=generators['validation'])
        reset_target_rng(1314, generators)
        expected = torch.rand(4)
        expected_order = torch.randperm(8, generator=generators['target'])
        reset_target_rng(1314, generators)
        list(loader)
        self.assertTrue(torch.equal(torch.rand(4), expected))
        self.assertTrue(torch.equal(torch.randperm(8, generator=generators['target']), expected_order))

    def test_worker_augmentation_streams_repeat_with_same_seed(self):
        def run(seed):
            seed_everything(seed)
            generators = make_loader_generators(seed)
            loader = torch.utils.data.DataLoader(RandomDataset(), batch_size=2, shuffle=True,
                                                 num_workers=2, worker_init_fn=seed_worker,
                                                 generator=generators['target'],
                                                 collate_fn=list, timeout=10)
            # Return Python scalars from workers. This tests augmentation seeds
            # without requiring a local socket for PyTorch Tensor FD sharing.
            return [torch.tensor(batch, dtype=torch.float64) for batch in loader]

        first, second, different = run(1314), run(1314), run(1315)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(first, different)))

    def test_dataset_construction_does_not_reset_numpy_rng(self):
        from dataset import FER, RafDataSet
        with tempfile.TemporaryDirectory() as root:
            paths = [str(Path(root) / 'train' / '0' / 'a.jpg')]
            label_dir = Path(root) / 'EmoLabel'
            label_dir.mkdir()
            (label_dir / 'list_patition_label.txt').write_text(
                'train_00001.jpg 1\ntrain_00002.jpg 2\ntrain_00003.jpg 3\n')
            np.random.seed(77)
            expected = np.random.rand()
            np.random.seed(77)
            with patch('dataset.glob.glob', return_value=paths), redirect_stdout(io.StringIO()):
                FER(root, 'train')
                source = RafDataSet(root, 'train')
            self.assertEqual(np.random.rand(), expected)
            for path, label in zip(source.file_paths, source.label):
                self.assertEqual(int(Path(path).name.split('_')[1]), label + 1)

    def test_path_digest_is_root_independent_but_order_sensitive(self):
        first = path_order_sha256(['/old/a.jpg', '/old/b.jpg'], '/old')
        self.assertEqual(first, path_order_sha256(['/new/a.jpg', '/new/b.jpg'], '/new'))
        self.assertNotEqual(first, path_order_sha256(['/new/b.jpg', '/new/a.jpg'], '/new'))


class SourceBalanceTests(unittest.TestCase):
    def test_disabled_balance_is_exactly_original_ce(self):
        weights = source_class_weights([100, 2, 10], power=0)
        losses = torch.tensor([0.3, 1.8, 0.7])
        self.assertTrue(torch.equal(balance_source_losses(losses, torch.arange(3), weights), losses))

    def test_relative_weights_are_bounded_and_ignore_absent_classes(self):
        weights = source_class_weights([4772, 281, 705, 0], power=0.5, max_ratio=2)
        self.assertEqual(weights.tolist(), [1., 2., 2., 0.])
        for counts in ([], [0, 0], [-1, 3]):
            with self.assertRaises(ValueError):
                source_class_weights(counts)

    def test_rare_source_class_receives_more_gradient_without_target_reweighting(self):
        logits = torch.zeros(3, 2, requires_grad=True)
        labels = torch.tensor([0, 1, 0])
        ce = F.cross_entropy(logits, labels, reduction='none')
        source_ce = balance_source_losses(ce[:2], labels[:2], source_class_weights([100, 2], 0.5))
        loss = torch.cat((source_ce, ce[2:])).mean()
        loss.backward()
        self.assertAlmostEqual(logits.grad[1].abs().sum().item() /
                               logits.grad[0].abs().sum().item(), 2.0)
        self.assertAlmostEqual(logits.grad[2].abs().sum().item(), 1. / 3.)
        # Uniform losses retain their aggregate scale under balancing.
        self.assertAlmostEqual(source_ce.sum().item(), ce[:2].sum().item())

    def test_recovery_cli_defaults_and_legacy_alias(self):
        import train
        with patch('sys.argv', ['train.py']):
            args = train.parse_args()
        self.assertEqual(args.temporal_mode, 'observe')
        self.assertEqual(args.target_soft_weight, 0)
        self.assertEqual(args.source_balance_power, 0)
        with patch('sys.argv', ['train.py', '--disable_temporal']):
            self.assertEqual(train.parse_args().temporal_mode, 'off')
        with patch('sys.argv', ['train.py', '--source_balance_power', 'nan']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                train.parse_args()


if __name__ == '__main__':
    unittest.main()
