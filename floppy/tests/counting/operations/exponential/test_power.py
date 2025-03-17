# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np

from floppy.counting.base import FlopCounter


def test_power_operator_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a**b
        assert counter.flops == 120
        np.testing.assert_allclose(result, np.array([1, 32, 729]))


def test_power_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.power(a, b)
        assert counter.flops == 120
        np.testing.assert_allclose(result, np.array([1, 32, 729]))


def test_power_method_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a.power(b)
        assert counter.flops == 120
        np.testing.assert_allclose(result, np.array([1, 32, 729]))


def test_power_augmented_assignment():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a **= b
        assert counter.flops == 120
        np.testing.assert_allclose(a, np.array([1, 32, 729]))


def test_square():
    """Test that when exponent is 2, NumPy optimizes by using the square ufunc instead of power.

    This is an optimization in NumPy where a**2 triggers the square ufunc rather than the
    more general power ufunc, resulting in fewer floating point operations.
    """
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a**b
        assert counter.flops == 3  # One multiplication per element
        np.testing.assert_allclose(result, np.array([1, 4, 9]))


def test_square_2():
    # test np.square()
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.square(a)
        assert counter.flops == 3  # One multiplication per element
        np.testing.assert_allclose(result, np.array([1, 4, 9]))


def test_sqrt():
    """Test square root operation (power of 0.5)."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 4, 9])
        result = np.sqrt(a)
        assert counter.flops == 60  # 20 FLOPs per sqrt operation
        np.testing.assert_allclose(result, np.array([1, 2, 3]))


def test_cbrt():
    """Test cube root operation (power of 1/3)."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 8, 27])
        result = np.cbrt(a)
        assert counter.flops == 75  # 25 FLOPs per cbrt operation
        np.testing.assert_allclose(result, np.array([1, 2, 3]))


def test_reciprocal():
    """Test reciprocal operation (power of -1)."""
    counter = FlopCounter()
    with counter:
        a = np.array([1.0, 2.0, 4.0])
        result = np.reciprocal(a)
        assert counter.flops == 3  # 1 FLOP (division) per element
        np.testing.assert_allclose(result, np.array([1.0, 0.5, 0.25]))


def test_negative_integer_power():
    """Test negative integer powers."""
    counter = FlopCounter()
    with counter:
        a = np.array([1.0, 2.0, 3.0])
        result = a ** (-2)
        assert counter.flops == 6  # (2-1) multiplications + 1 division per element
        np.testing.assert_array_almost_equal(result, np.array([1, 0.25, 1 / 9]))


def test_fractional_power():
    """Test general fractional power."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = a ** (1.5)  # Neither sqrt nor cbrt
        assert counter.flops == 120  # 40 FLOPs per element for general fractional power
        np.testing.assert_array_almost_equal(
            result, np.array([1, 2.8284271247461903, 5.196152422706632])
        )


def test_power_broadcasting():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b**a
        assert counter.flops == 120
        np.testing.assert_allclose(result, np.array([2, 4, 8]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a**b
        assert counter.flops == 3  # Uses square optimization
        np.testing.assert_allclose(result, np.array([1, 4, 9]))


def test_power_empty_arrays():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        b = np.array([])
        result = a**b
        assert counter.flops == 0
        assert len(result) == 0
