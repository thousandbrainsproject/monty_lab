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


def test_exponential_scalar() -> None:
    """Test exponential behavior and flop count with scalar input."""
    counter = FlopCounter()
    with counter:
        x = 2.0
        result = np.exp(x)
        assert counter.flops == 20  # noqa: PLR2004
        np.testing.assert_allclose(result, np.exp(2.0))


def test_exponential_array() -> None:
    """Test exponential behavior and flop count with 1D array input."""
    counter = FlopCounter()
    with counter:
        x = np.array([1.0, 2.0, 3.0])
        result = np.exp(x)
        assert counter.flops == 60  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([np.exp(1.0), np.exp(2.0), np.exp(3.0)])
        )


def test_exponential_2d_array() -> None:
    """Test exponential behavior and flop count with 2D array input."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = np.exp(x)
        assert counter.flops == 80  # noqa: PLR2004
        expected = np.array([[np.exp(1.0), np.exp(2.0)], [np.exp(3.0), np.exp(4.0)]])
        np.testing.assert_allclose(result, expected)


def test_exponential_empty_array() -> None:
    """Test exponential behavior and flop count with empty array input."""
    counter = FlopCounter()
    with counter:
        x = np.array([])
        result = np.exp(x)
        assert counter.flops == 0
        assert len(result) == 0


def test_exponential_broadcasting() -> None:
    """Test exponential behavior and flop count with broadcasting."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        result = np.exp(x + y)  # Broadcasting y to match x's shape
        assert counter.flops == 84  # noqa: PLR2004
        expected = np.array([[np.exp(2.0), np.exp(4.0)], [np.exp(4.0), np.exp(6.0)]])
        np.testing.assert_allclose(result, expected)


def test_exponential_negative_values() -> None:
    """Test exponential behavior and flop count with negative values."""
    counter = FlopCounter()
    with counter:
        x = np.array([-1.0, -2.0, -3.0])
        result = np.exp(x)
        assert counter.flops == 60  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([np.exp(-1.0), np.exp(-2.0), np.exp(-3.0)])
        )


def test_exponential_large_values() -> None:
    """Test exponential behavior and flop count with large values."""
    counter = FlopCounter()
    with counter:
        x = np.array([10.0, 20.0, 30.0])
        result = np.exp(x)
        assert counter.flops == 60  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([np.exp(10.0), np.exp(20.0), np.exp(30.0)])
        )


def test_exponential_small_values() -> None:
    """Test exponential behavior and flop count with small values."""
    counter = FlopCounter()
    with counter:
        x = np.array([1e-10, 1e-20, 1e-30])
        result = np.exp(x)
        assert counter.flops == 60  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([np.exp(1e-10), np.exp(1e-20), np.exp(1e-30)])
        )
