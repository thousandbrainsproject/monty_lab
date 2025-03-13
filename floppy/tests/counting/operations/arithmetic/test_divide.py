import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_divide_operator_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a / b
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([0.25, 0.4, 0.5]))


def test_divide_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.divide(a, b)
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([0.25, 0.4, 0.5]))


def test_divide_method_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a.divide(b)
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([0.25, 0.4, 0.5]))


def test_divide_augmented_assignment():
    counter = FlopCounter()
    with counter:
        # dtype=np.float64 is required for in-place division since integers can't store decimal results
        a = np.array([1, 2, 3], dtype=np.float64)
        b = np.array([4, 5, 6], dtype=np.float64)
        a /= b
        assert counter.flops == 3
        np.testing.assert_allclose(a, np.array([0.25, 0.4, 0.5]))


def test_divide_broadcasting():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a / b
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([0.5, 1, 1.5]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b / a
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([2, 1, 0.6666666666666666]))
