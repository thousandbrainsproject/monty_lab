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


def test_average_scalar() -> None:
    """Test average behavior and flop count for scalar input."""
    counter = FlopCounter()
    with counter:
        x = np.array(5)  # scalar array
        result = np.average(x)
        assert counter.flops == 0  # scalar inputs require no computation
        np.testing.assert_allclose(result, 5.0)


def test_average_scalar_python() -> None:
    """Test average behavior and flop count for scalar input using Python."""
    counter = FlopCounter()
    with counter:
        result = np.average(5)
        assert counter.flops == 0  # scalar inputs require no computation
        np.testing.assert_allclose(result, 5.0)


def test_average_1d() -> None:
    """Test average behavior and flop count for 1D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        result = np.average(x)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, 3.0)


def test_average_2d() -> None:
    """Test average behavior and flop count for 2D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.average(x)
        assert counter.flops == 7  # noqa: PLR2004
        np.testing.assert_allclose(result, 3.5)


def test_average_single() -> None:
    """Test average behavior and flop count for single element array."""
    counter = FlopCounter()
    with counter:
        x = np.array([1])
        result = np.average(x)
        assert counter.flops == 2  # noqa: PLR2004
        np.testing.assert_allclose(result, 1.0)


def test_average_weighted_1d() -> None:
    """Test average behavior and flop count for weighted 1D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        result = np.average(x, weights=weights)
        assert counter.flops == 11  # noqa: PLR2004
        np.testing.assert_allclose(result, 3.2)


def test_average_weighted_2d() -> None:
    """Test average behavior and flop count for weighted 2D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        weights = np.array([0.5, 0.5])
        result = np.average(x, weights=weights, axis=0)  # average along first axis
        assert counter.flops == 16  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([2.5, 3.5, 4.5]))


def test_average_axis() -> None:
    """Test average behavior and flop count for axis parameter."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.average(x, axis=0)  # average of each column
        assert counter.flops == 7  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([2.5, 3.5, 4.5]))


def test_average_broadcast() -> None:
    """Test average behavior and flop count for broadcast operation."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        result = np.average(x + y)
        assert counter.flops == 13  # noqa: PLR2004
        np.testing.assert_allclose(result, 5.5)


def test_average_weighted_broadcast() -> None:
    """Test average behavior and flop count for weighted broadcast operation."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        weights = np.ones_like(x)
        result = np.average(x + y, weights=weights)
        assert counter.flops == 19  # noqa: PLR2004
        np.testing.assert_allclose(result, 5.5)


def test_average_multi_axis() -> None:
    """Test average behavior and flop count for multiple axes."""
    counter = FlopCounter()
    with counter:
        x = np.ones((2, 3, 4))
        result = np.average(x, axis=(0, 2))
        assert counter.flops == 25  # noqa: PLR2004
        np.testing.assert_allclose(result, np.ones(3))


def test_masked_average() -> None:
    """Test average behavior and flop count for masked array."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        mask = np.array([True, False, True, False, True])
        masked_x = np.ma.array(x, mask=mask)
        result = np.ma.average(masked_x)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, 3.0)
