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


def test_mean_scalar() -> None:
    """Test mean of scalar value."""
    counter = FlopCounter()
    with counter:
        x = np.array(5)
        result = np.mean(x)
        assert counter.flops == 0  # scalar inputs require no computation
        np.testing.assert_allclose(result, 5.0)


def test_mean_scalar_python() -> None:
    """Test mean of Python scalar."""
    counter = FlopCounter()
    with counter:
        result = np.mean(5)
        assert counter.flops == 0  # scalar inputs require no computation
        np.testing.assert_allclose(result, 5.0)


def test_mean_1d() -> None:
    """Test mean of 1D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        result = np.mean(x)
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, 3.0)


def test_mean_2d() -> None:
    """Test mean of 2D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.mean(x)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, 3.5)


def test_mean_3d() -> None:
    """Test mean of 3D array."""
    counter = FlopCounter()
    with counter:
        x = np.ones((2, 3, 4))
        result = np.mean(x)
        assert counter.flops == 24  # noqa: PLR2004
        np.testing.assert_allclose(result, 1.0)


def test_mean_empty() -> None:
    """Test mean of empty array."""
    counter = FlopCounter()
    with counter:
        x = np.array([])
        with suppress(RuntimeWarning):
            _ = np.mean(x)
        assert counter.flops == 0


def test_mean_single() -> None:
    """Test mean of single element."""
    counter = FlopCounter()
    with counter:
        x = np.array([1])
        result = np.mean(x)
        assert counter.flops == 1
        np.testing.assert_allclose(result, 1.0)


def test_mean_axis() -> None:
    """Test mean with axis argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.mean(x, axis=0)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([2.5, 3.5, 4.5]))


def test_mean_keepdims() -> None:
    """Test mean with keepdims=True."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.mean(x, keepdims=True)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[3.5]]))


def test_mean_dtype() -> None:
    """Test mean with dtype argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = np.mean(x, dtype=np.float64)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, 3.5)


def test_mean_method() -> None:
    """Test array.mean() method call."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4])
        result = x.mean()
        assert counter.flops == 4  # noqa: PLR2004
        np.testing.assert_allclose(result, 2.5)


def test_mean_broadcast() -> None:
    """Test mean with broadcasting."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        result = np.mean(x + y)
        assert counter.flops == 12  # noqa: PLR2004
        np.testing.assert_allclose(result, 5.5)


def test_mean_multi_axis() -> None:
    """Test mean with multiple axes."""
    counter = FlopCounter()
    with counter:
        x = np.ones((2, 3, 4))
        result = np.mean(x, axis=(0, 2))
        assert counter.flops == 24  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([1.0, 1.0, 1.0]))
