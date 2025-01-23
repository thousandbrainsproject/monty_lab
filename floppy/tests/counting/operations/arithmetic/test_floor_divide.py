import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_floor_divide_operator_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a // b
        assert counter.flops == 6
        np.testing.assert_array_equal(result, np.array([0, 0, 0]))


def test_floor_divide_ufunc_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.floor_divide(a, b)
        assert counter.flops == 6
        np.testing.assert_array_equal(result, np.array([0, 0, 0]))


def test_floor_divide_augmented_assignment():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a //= b
        assert counter.flops == 6
        np.testing.assert_array_equal(a, np.array([0, 0, 0]))

def test_floor_divide_broadcasting():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a // b
        assert counter.flops == 6
        np.testing.assert_array_equal(result, np.array([0, 1, 1]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b // a
        assert counter.flops == 6
        np.testing.assert_array_equal(result, np.array([2, 1, 0]))
