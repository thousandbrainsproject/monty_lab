"""Run using python tests/test_where.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_where_basic():
    counter = FlopCounter()
    with counter:
        condition = np.array([True, False, True])
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        _ = np.where(condition, x, y)
        assert counter.flops == 3  # 1 flop per element for selection


def test_where_broadcasting():
    counter = FlopCounter()
    with counter:
        condition = np.array([[True, False], [False, True]])
        x = 1
        y = 0
        _ = np.where(condition, x, y)
        assert counter.flops == 4  # 1 flop per element

    counter.flops = 0
    with counter:
        condition = True
        x = np.array([[1, 2], [3, 4]])
        y = 0
        _ = np.where(condition, x, y)
        assert counter.flops == 4


def test_where_empty():
    counter = FlopCounter()
    with counter:
        condition = np.array([])
        x = np.array([])
        y = np.array([])
        _ = np.where(condition, x, y)
        assert counter.flops == 0


if __name__ == "__main__":
    test_where_basic()
    test_where_broadcasting()
    test_where_empty()
