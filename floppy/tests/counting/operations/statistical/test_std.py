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


def test_std_scalar() -> None:
    """Test std of scalar value."""
    counter = FlopCounter()
    with counter:
        x = np.array(5)
        result = np.std(x)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 0)


def test_std_scalar_python() -> None:
    """Test std of Python scalar."""
    counter = FlopCounter()
    with counter:
        result = np.std(5)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 0)


def test_std_1d() -> None:
    """Test std of 1D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        result = np.std(x)
        assert counter.flops == 40  # noqa: PLR2004
        np.testing.assert_allclose(result, 1.41421356)


def test_std_2d() -> None:
    """Test std of 2D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.std(x)
        assert counter.flops == 44  # noqa: PLR2004
        np.testing.assert_allclose(result, 1.707825127659933)


def test_std_single() -> None:
    """Test std of single element."""
    counter = FlopCounter()
    with counter:
        x = np.array([1])
        result = np.std(x)
        assert counter.flops == 24  # noqa: PLR2004
        np.testing.assert_allclose(result, 0)


def test_std_axis() -> None:
    """Test std with axis argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.std(x, axis=0)
        assert counter.flops == 44  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([1.5, 1.5, 1.5]))


def test_std_keepdims() -> None:
    """Test std with keepdims=True."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.std(x, keepdims=True)
        assert counter.flops == 44  # noqa: PLR2004
        np.testing.assert_allclose(
            result,
            np.array([[1.707825127659933]]),
        )


def test_std_dtype() -> None:
    """Test std with dtype argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = np.std(x, dtype=np.float64)
        assert counter.flops == 44  # noqa: PLR2004
        np.testing.assert_allclose(
            result,
            np.array([[1.707825127659933, 1.707825127659933, 1.707825127659933]]),
        )


def test_std_method() -> None:
    """Test array.std() method call."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4])
        result = x.std()
        assert counter.flops == 36  # noqa: PLR2004
        np.testing.assert_allclose(result, 1.118033988749895)


def test_std_broadcast() -> None:
    """Test std with broadcasting."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        result = np.std(x + y)
        assert counter.flops == 50  # noqa: PLR2004
        np.testing.assert_allclose(result, 2.217355782608345)


def test_std_multi_axis() -> None:
    """Test std with multiple axes."""
    counter = FlopCounter()
    with counter:
        x = np.ones((2, 3, 4))
        result = np.std(x, axis=(0, 2))
        assert counter.flops == 116  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([0.0, 0.0, 0.0]))
