"""Run using python tests/test_minmax.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_min_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = np.min(a)
        assert counter.flops == 3  # n-1 comparisons for n elements

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.min(a)
        assert counter.flops == 3


def test_max_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = np.max(a)
        assert counter.flops == 3  # n-1 comparisons for n elements

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.max(a)
        assert counter.flops == 3


def test_minmax_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.min(a)
        assert counter.flops == 0

    counter.flops = 0
    with counter:
        _ = np.max(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_min_basic()
    test_max_basic()
    test_minmax_empty()
