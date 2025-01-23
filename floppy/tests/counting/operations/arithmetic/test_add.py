import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_add_operator_syntax():
    counter = FlopCounter(test_mode=True)

    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a + b
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([5, 7, 9]))

def test_add_ufunc_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.add(a, b)
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([5, 7, 9]))

# FIXME
@pytest.mark.xfail(reason="TrackedArray object has no attribute 'add'")
def test_add_method_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a.add(b)
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([5, 7, 9]))

def test_add_augmented_assignment():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a += b
        assert counter.flops == 3
        np.testing.assert_array_equal(a, np.array([5, 7, 9]))

def test_add_broadcasting():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a + b
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([3, 4, 5]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b + a
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([3, 4, 5]))
