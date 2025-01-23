"""Run using python tests/test_argminmax.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_argmin_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([4, 2, 3, 1])
        _ = np.argmin(a)
        assert counter.flops == 3  # n-1 comparisons for n elements

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.argmin(a)
        assert counter.flops == 3  # 4 elements, 3 comparisons


def test_argmin_axis():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.argmin(a, axis=0)  # Compare elements in each column
        assert counter.flops == 3  # 1 comparison per column

    counter.flops = 0
    with counter:
        _ = np.argmin(a, axis=1)  # Compare elements in each row
        assert counter.flops == 4  # 2 comparisons per row


def test_argmax_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([4, 2, 3, 1])
        _ = np.argmax(a)
        assert counter.flops == 3  # n-1 comparisons for n elements

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.argmax(a)
        assert counter.flops == 3  # 4 elements, 3 comparisons


def test_argmax_axis():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.argmax(a, axis=0)  # Compare elements in each column
        assert counter.flops == 3  # 1 comparison per column

    counter.flops = 0
    with counter:
        _ = np.argmax(a, axis=1)  # Compare elements in each row
        assert counter.flops == 4  # 2 comparisons per row


def test_argminmax_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.argmin(a)
        assert counter.flops == 0

    counter.flops = 0
    with counter:
        _ = np.argmax(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_argmin_basic()
    test_argmin_axis()
    test_argmax_basic()
    test_argmax_axis()
    test_argminmax_empty()
