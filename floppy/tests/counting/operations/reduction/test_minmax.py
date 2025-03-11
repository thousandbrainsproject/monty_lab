import numpy as np

from floppy.counting.counter import FlopCounter


def test_min_basic():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3, 4])
        result = np.min(a)
        assert counter.flops == 3  # n-1 comparisons for n elements
        np.testing.assert_equal(result, 1)

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.min(a)
        assert counter.flops == 3
        np.testing.assert_equal(result, 1)


def test_max_basic():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3, 4])
        result = np.max(a)
        assert counter.flops == 3  # n-1 comparisons for n elements
        np.testing.assert_equal(result, 4)

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.max(a)
        assert counter.flops == 3
        np.testing.assert_equal(result, 4)