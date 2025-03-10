import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_where_basic():
    counter = FlopCounter(test_mode=True)
    with counter:
        condition = np.array([True, False, True])
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = np.where(condition, x, y)
        assert counter.flops == 3  # 1 flop per element for selection
        np.testing.assert_array_equal(result, np.array([1, 5, 3]))


def test_where_broadcasting():
    counter = FlopCounter(test_mode=True)
    with counter:
        condition = np.array([[True, False], [False, True]])
        x = 1
        y = 0
        result = np.where(condition, x, y)
        assert counter.flops == 4  # 1 flop per element
        np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]]))

    counter.flops = 0
    with counter:
        condition = True
        x = np.array([[1, 2], [3, 4]])
        y = 0
        result = np.where(condition, x, y)
        assert counter.flops == 4  # 1 flop per element in the result array
        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))


def test_where_empty():
    counter = FlopCounter(test_mode=True)
    with counter:
        condition = np.array([])
        x = np.array([])
        y = np.array([])
        _ = np.where(condition, x, y)
        assert counter.flops == 0
