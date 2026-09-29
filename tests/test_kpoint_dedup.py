import unittest

from wannierberri.grid.Kpoint import exclude_equiv_points


class FakePoint:
    comparisons = 0

    def __init__(self, label, factor=1.0):
        self.label = label
        self.factor = factor
        self.distGamma = 1.0

    def equiv(self, other):
        type(self).comparisons += 1
        return self.label == other.label

    def absorb(self, other):
        self.factor += other.factor


class TestIncrementalDeduplication(unittest.TestCase):
    def test_only_new_points_are_duplicate_candidates(self):
        old = [FakePoint(i) for i in range(2000)]
        new = [FakePoint(17), FakePoint(3000), FakePoint(3000)]
        points = old + new
        FakePoint.comparisons = 0

        exclude_equiv_points(points, new_points=len(new))

        self.assertEqual(len(points), len(old) + 1)
        self.assertEqual(points[17].factor, 2.0)
        self.assertEqual(points[-1].label, 3000)
        self.assertEqual(points[-1].factor, 2.0)
        # A quadratic old/old scan would make roughly four million calls.
        self.assertLessEqual(FakePoint.comparisons, len(old) * len(new) + 3)


if __name__ == '__main__':
    unittest.main()
