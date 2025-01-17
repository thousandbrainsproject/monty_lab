"""Run using python tests/test_sum.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_sum_np_function():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = np.sum(a)
        assert counter.flops == 3  # n-1 additions for n elements


def test_sum_method():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = a.sum()
        assert counter.flops == 3


def test_sum_axis():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.sum(a, axis=0)  # Sum columns
        assert counter.flops == 2  # 1 addition per column * 3 columns

    counter.flops = 0
    with counter:
        _ = np.sum(a, axis=1)  # Sum rows
        assert counter.flops == 4  # 2 additions per row * 2 rows


def test_sum_keepdims():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.sum(a, keepdims=True)
        assert counter.flops == 5  # 6 elements - 1 addition


def test_sum_where():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        mask = np.array([True, False, True, False])
        _ = np.sum(a, where=mask)
        assert counter.flops == 1  # Only one addition needed for two True values


def test_sum_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.sum(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_sum_np_function()
    test_sum_method()
    test_sum_axis()
    test_sum_keepdims()
    test_sum_where()
    test_sum_empty()
