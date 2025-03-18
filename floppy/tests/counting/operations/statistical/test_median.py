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


def test_median_np_function() -> None:
    """Test median behavior and flop count using np.median."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4, 5])
        result = np.median(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 3)


def test_median_even_length() -> None:
    """Test median behavior and flop count for even length array."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        result = np.median(a)
        assert counter.flops == 2  # noqa: PLR2004
        np.testing.assert_allclose(result, 2.5)


def test_median_method() -> None:
    """Test median behavior and flop count using array.median method."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        result = np.median(a)
        assert counter.flops == 2  # noqa: PLR2004
        np.testing.assert_allclose(result, 2.5)


def test_median_axis() -> None:
    """Test median behavior and flop count for axis parameter."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.median(a, axis=0)
        assert counter.flops == 2  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([2.5, 3.5, 4.5]))

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.median(a, axis=1)
        assert counter.flops == 2  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([1.5, 3.5]))


def test_median_keepdims() -> None:
    """Test median behavior and flop count for keepdims parameter."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.median(a, keepdims=True)
        assert counter.flops == 2  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[3.5]]))


def test_median_empty() -> None:
    """Test median behavior and flop count for empty array."""
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.median(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.nan)


def test_median_single_element() -> None:
    """Test median behavior and flop count for single element array."""
    counter = FlopCounter()
    with counter:
        a = np.array([1])
        result = np.median(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 1)
