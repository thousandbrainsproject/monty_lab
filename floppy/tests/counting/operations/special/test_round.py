import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_round_basic():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1.4, 2.6, 3.5])
        result = np.round(a)
        assert counter.flops == 3  # 1 flop per element
        np.testing.assert_array_equal(result, np.array([1, 3, 4]))


def test_round_broadcasting():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = 2.5
        result = np.round(a)
        assert counter.flops == 1
        np.testing.assert_array_equal(result, 2)

    counter.flops = 0
    with counter:
        a = np.array([[1.4, 2.6], [3.5, 4.1]])
        result = np.round(a)
        assert counter.flops == 4
        np.testing.assert_array_equal(result, np.array([[1, 3], [4, 4]]))


def test_round_empty():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([])
        result = np.round(a)
        assert counter.flops == 0
        np.testing.assert_array_equal(result, np.array([]))
