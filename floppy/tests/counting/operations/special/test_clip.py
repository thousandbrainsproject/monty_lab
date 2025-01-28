import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_clip_direct_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        result = np.clip(a, 0, 2)
        assert counter.flops == 6
        np.testing.assert_array_equal(result, np.array([1, 2, 2]))


def test_clip_method_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        result = a.clip(0, 2)
        assert counter.flops == 6
        np.testing.assert_array_equal(result, np.array([1, 2, 2]))


def test_clip_broadcasting():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 5, 10], [-5, 3, 7]])
        min_vals = np.array([2, 0, 1])  # Broadcasting across rows
        max_vals = np.array([8, 6, 9])  # Broadcasting across rows
        result = np.clip(a, min_vals, max_vals)
        # 6 elements, 2 comparisons each = 12 FLOPs
        assert counter.flops == 12
        np.testing.assert_array_equal(result, np.array([[2, 5, 9], [2, 3, 7]]))


def test_clip_inplace():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 5, 10, -5, 3])
        np.clip(a, 2, 8, out=a)
        assert counter.flops == 10
        np.testing.assert_array_equal(a, np.array([2, 5, 8, 2, 3]))


def test_clip_scalar_bounds():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 5, 10, -5, 3])
        result = np.clip(a, 2.5, 7.5)
        assert counter.flops == 10
        np.testing.assert_array_equal(result, np.array([2.5, 5, 7.5, 2.5, 3]))


def test_clip_none_bounds():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 5, 10, -5, 3])
        # Only clip on one side (a_min)
        result1 = np.clip(a, 2, None)
        assert counter.flops == 5  # Only one comparison per element
        np.testing.assert_array_equal(result1, np.array([2, 5, 10, 2, 3]))

    counter.flops = 0
    with counter:
        # Only clip on other side (a_max)
        result2 = np.clip(a, None, 8)
        assert counter.flops == 5  # Only one comparison per element
        np.testing.assert_array_equal(result2, np.array([1, 5, 8, -5, 3]))
