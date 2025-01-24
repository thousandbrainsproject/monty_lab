import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_isnan_basic():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1.0, np.nan, 3.0])
        result = np.isnan(a)
        assert counter.flops == 3  # 1 flop per element
        np.testing.assert_array_equal(result, np.array([False, True, False]))


def test_isnan_broadcasting():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.nan
        result = np.isnan(a)
        assert counter.flops == 1
        np.testing.assert_array_equal(result, np.array([True]))

    counter.flops = 0
    with counter:
        a = np.array([[1.0, np.nan], [3.0, np.nan]])
        result = np.isnan(a)
        assert counter.flops == 4
        np.testing.assert_array_equal(result, np.array([[False, True], [False, True]]))


def test_isnan_empty():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([])
        result = np.isnan(a)
        assert counter.flops == 0
        np.testing.assert_array_equal(result, np.array([]))
