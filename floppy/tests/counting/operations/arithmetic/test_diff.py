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


def test_diff_basic() -> None:
    """Test np.diff behavior and flop count with basic array."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 4, 7, 0])
        result = np.diff(a)
        assert counter.flops == 4  # One subtraction per element in result
        np.testing.assert_allclose(result, np.array([1, 2, 3, -7]))


def test_diff_2d() -> None:
    """Test np.diff behavior and flop count with 2D arrays."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.diff(a, axis=0)  # Diff along rows
        assert counter.flops == 3  # One subtraction per element in result
        np.testing.assert_allclose(result, np.array([[3, 3, 3]]))

    counter.flops = 0
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.diff(a, axis=1)  # Diff along columns
        assert counter.flops == 4  # One subtraction per element in result
        np.testing.assert_allclose(result, np.array([[1, 1], [1, 1]]))


def test_diff_n() -> None:
    """Test np.diff behavior and flop count with nth difference."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 4, 7, 0])
        result = np.diff(a, n=2)  # Second difference
        assert counter.flops == 3  # One subtraction per element in result
        np.testing.assert_allclose(result, np.array([1, 1, -10]))


def test_diff_prepend() -> None:
    """Test np.diff behavior and flop count with prepended zeros."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 4, 7, 0])
        result = np.diff(a, prepend=0)
        assert counter.flops == 5  # One subtraction per element in result
        np.testing.assert_allclose(result, np.array([1, 1, 2, 3, -7]))


def test_diff_append() -> None:
    """Test np.diff behavior and flop count with appended zeros."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 4, 7, 0])
        result = np.diff(a, append=0)
        assert counter.flops == 5  # One subtraction per element in result
        np.testing.assert_allclose(result, np.array([1, 2, 3, -7, 0]))


def test_diff_empty() -> None:
    """Test np.diff behavior and flop count with empty array."""
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.diff(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]))


def test_diff_single_element() -> None:
    """Test np.diff behavior and flop count with single-element array."""
    counter = FlopCounter()
    with counter:
        a = np.array([1])
        result = np.diff(a)
        assert counter.flops == 0  # No elements to diff
        np.testing.assert_allclose(result, np.array([]))
