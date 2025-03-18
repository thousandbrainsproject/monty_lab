# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from contextlib import suppress

import numpy as np

from floppy.counting.base import FlopCounter


def test_var_scalar() -> None:
    """Test var of scalar value."""
    counter = FlopCounter()
    with counter:
        x = np.array(5)
        result = np.var(x)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 0)


def test_var_scalar_python() -> None:
    """Test var of Python scalar."""
    counter = FlopCounter()
    with counter:
        result = np.var(5)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 0)


def test_var_1d() -> None:
    """Test var of 1D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        result = np.var(x)
        assert counter.flops == 20  # noqa: PLR2004
        np.testing.assert_allclose(result, 2.0)


def test_var_2d() -> None:
    """Test var of 2D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.var(x)
        assert counter.flops == 24  # noqa: PLR2004
        np.testing.assert_allclose(result, 2.9166666666666665)


def test_var_empty() -> None:
    """Test var of empty array."""
    counter = FlopCounter()
    with counter:
        x = np.array([])
        with suppress(RuntimeWarning):
            _ = np.var(x)
        assert counter.flops == 0  # empty arrays return 0 FLOPs


def test_var_single() -> None:
    """Test var of single element."""
    counter = FlopCounter()
    with counter:
        x = np.array([1])
        result = np.var(x)
        assert counter.flops == 4  # noqa: PLR2004
        np.testing.assert_allclose(result, 0)


def test_var_axis() -> None:
    """Test var with axis argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.var(x, axis=0)
        assert counter.flops == 24  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([2.25, 2.25, 2.25]))


def test_var_keepdims() -> None:
    """Test var with keepdims=True."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.var(x, keepdims=True)
        assert counter.flops == 24  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[2.9166666666666665]]))


def test_var_dtype() -> None:
    """Test var with dtype argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = np.var(x, dtype=np.float64)
        assert counter.flops == 24  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[2.9166666666666665]]))


def test_var_method() -> None:
    """Test array.var() method call."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4])
        result = x.var()
        assert counter.flops == 16  # noqa: PLR2004
        np.testing.assert_allclose(result, 1.25)


def test_var_broadcast() -> None:
    """Test var with broadcasting."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        result = np.var(x + y)
        assert counter.flops == 30  # noqa: PLR2004
        np.testing.assert_allclose(result, 4.916666666666667)


def test_var_multi_axis() -> None:
    """Test var with multiple axes."""
    counter = FlopCounter()
    with counter:
        x = np.ones((2, 3, 4))
        result = np.var(x, axis=(0, 2))
        assert counter.flops == 96  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([0.0, 0.0, 0.0]))
