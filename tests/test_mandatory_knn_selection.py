"""Exercise the production selector truth table without requiring PyTorch."""
import ast
import itertools
from pathlib import Path
import unittest


class Mask:
    def __init__(self, value):
        self.value = bool(value)

    def bool(self):
        return self

    def __and__(self, other):
        return Mask(self.value and other.value)

    def __or__(self, other):
        return Mask(self.value or other.value)

    def __invert__(self):
        return Mask(not self.value)


class TorchMasks:
    @staticmethod
    def zeros_like(mask):
        return Mask(False)


source = (Path(__file__).resolve().parents[1] / 'raw_medoid_quantile.py').read_text()
tree = ast.parse(source)
bank = next(n for n in tree.body if isinstance(n, ast.ClassDef))
selector = next(n for n in bank.body if isinstance(n, ast.FunctionDef) and n.name == 'select')
selector.decorator_list = []
namespace = {'torch': TorchMasks}
exec(compile(ast.Module(body=[selector], type_ignores=[]), '<production selector>', 'exec'),
     namespace)


class FakeBank:
    select = namespace['select']

    def __init__(self, radius, vote, checked=True):
        self.radius = radius
        self.vote = vote
        self.checked = checked
        self.radius_called = False
        self.knn_candidate = None

    def accepts(self, features, labels):
        self.radius_called = True
        return Mask(self.radius)

    def neighbor_accepts(self, features, labels, candidates):
        self.knn_candidate = candidates.value
        return Mask(candidates.value and self.vote), Mask(candidates.value and self.checked)


class MandatoryKNNTests(unittest.TestCase):
    def test_all_combinations(self):
        for agreement, confidence, radius, vote, enabled in itertools.product(
                (False, True), repeat=5):
            with self.subTest(agreement=agreement, confidence=confidence,
                              radius=radius, vote=vote, enabled=enabled):
                bank = FakeBank(radius, vote)
                final, rescued, passed, checked = bank.select(
                    None, None, Mask(confidence), Mask(agreement), enabled)
                eligible = agreement and (confidence or (enabled and radius))
                expected = eligible and vote
                self.assertEqual(final.value, expected)
                self.assertEqual(passed.value, expected)
                self.assertEqual(checked.value, eligible)
                self.assertEqual(bank.knn_candidate, eligible)
                self.assertEqual(rescued.value, expected and not confidence)
                self.assertEqual(bank.radius_called, enabled)

    def test_unchecked_neighbors_cannot_accept(self):
        bank = FakeBank(True, True, checked=False)
        final, rescued, passed, checked = bank.select(
            None, None, Mask(True), Mask(True), True)
        self.assertFalse(final.value)
        self.assertFalse(rescued.value)
        self.assertFalse(passed.value)
        self.assertFalse(checked.value)


if __name__ == '__main__':
    unittest.main()
