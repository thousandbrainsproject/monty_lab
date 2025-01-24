import numpy as np
import pytest

from floppy.counting.counter import FlopCounter, TrackedArray

# TODO: Add test for fractional exponents
# TODO: Add test for negative exponents
# TODO: Add more tests for broadcasting
# TODO: Add more test for square (or separate in a new file)
# TODO: Find other "special" power cases
# - square - for power of 2
# - reciprocal - for power of -1
# - cbrt - for power of 1/3
# - sqrt - for power of 1/2

def test_power_operator_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a**b
        assert counter.flops == 120
        np.testing.assert_array_equal(result, np.array([1, 32, 729]))


def test_power_ufunc_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.power(a, b)
        assert counter.flops == 120
        np.testing.assert_array_equal(result, np.array([1, 32, 729]))


# FIXME
@pytest.mark.xfail(reason="TrackedArray object has no attribute 'power'")
def test_power_method_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a.power(b)
        assert counter.flops == 120
        np.testing.assert_array_equal(result, np.array([1, 32, 729]))


def test_power_augmented_assignment():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a **= b
        assert counter.flops == 120
        np.testing.assert_array_equal(a, np.array([1, 32, 729]))


def test_square():
    """Test that when exponent is 2, NumPy optimizes by using the square ufunc instead of power.

    This is an optimization in NumPy where a**2 triggers the square ufunc rather than the
    more general power ufunc, resulting in fewer floating point operations.
    """
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a**b
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([1, 4, 9]))

def test_power_broadcasting():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b**a
        assert counter.flops == 120
        np.testing.assert_array_equal(result, np.array([2, 4, 8]))