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


def test_divide_operator_syntax() -> None:
    """Test division using operator syntax (a / b)."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a / b
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([0.25, 0.4, 0.5]))


def test_divide_ufunc_syntax() -> None:
    """Test division using numpy ufunc syntax (np.divide)."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.divide(a, b)
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([0.25, 0.4, 0.5]))


def test_divide_method_syntax() -> None:
    """Test division using method syntax (a.divide)."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a.divide(b)
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([0.25, 0.4, 0.5]))


def test_divide_augmented_assignment() -> None:
    """Test division using augmented assignment (a /= b)."""
    counter = FlopCounter()
    with counter:
        # dtype=np.float64 is required for in-place division since integers can't store decimal results
        a = np.array([1, 2, 3], dtype=np.float64)
        b = np.array([4, 5, 6], dtype=np.float64)
        a /= b
        assert counter.flops == 3
        np.testing.assert_allclose(a, np.array([0.25, 0.4, 0.5]))


def test_divide_broadcasting() -> None:
    """Test division with broadcasting between array and scalar."""
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
