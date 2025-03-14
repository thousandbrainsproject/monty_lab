import numpy as np
import pytest

from floppy.counting.core import FlopCounter


def test_subtract_operator_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a - b
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([-3, -3, -3]))


def test_subtract_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.subtract(a, b)
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([-3, -3, -3]))

def test_subtract_method_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a.subtract(b)
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([-3, -3, -3]))


def test_subtract_augmented_assignment():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a -= b
        assert counter.flops == 3
        np.testing.assert_allclose(a, np.array([-3, -3, -3]))

def test_subtract_broadcasting():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a - b
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([-1, 0, 1]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b - a
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([1, 0, -1]))
