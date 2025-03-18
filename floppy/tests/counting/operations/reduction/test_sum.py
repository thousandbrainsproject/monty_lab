# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
from numpy import ma

from floppy.counting.base import FlopCounter


def test_sum_np_function() -> None:
    """Test sum behavior and flop count using np.sum."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        result = np.sum(a)
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, 10)


def test_sum_method() -> None:
    """Test sum behavior and flop count using array.sum method."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        result = a.sum()
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, 10)


def test_sum_axis() -> None:
    """Test sum behavior and flop count using axis parameter."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.sum(a, axis=0)  # Sum columns
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([5, 7, 9]))

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.sum(a, axis=1)  # Sum rows
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([3, 7]))


def test_sum_keepdims() -> None:
    """Test sum behavior and flop count using keepdims parameter."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.sum(a, keepdims=True)
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[21]]))


def test_sum_where() -> None:
    """Test sum behavior and flop count using where parameter."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        mask = np.array([True, False, True, False])
        result = np.sum(a, where=mask)
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, 4)


def test_sum_empty() -> None:
    """Test sum behavior and flop count for empty array."""
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.sum(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 0)


def test_nansum() -> None:
    """Test nansum behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, np.nan, 4])
        result = np.nansum(a)
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, 7)

    counter.flops = 0
    with counter:
        a = np.array([[1, 2, np.nan], [4, 5, 6]])
        result = np.nansum(a, axis=0)
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([5, 7, 6]))


def test_masked_sum() -> None:
    """Test masked sum behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = ma.array([1, 2, 3, 4], mask=[True, False, True, False])
        result = np.sum(a)
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, 6)

    counter.flops = 0
    with counter:
        a = ma.array(
            [[1, 2, 3], [4, 5, 6]], mask=[[True, False, True], [False, True, False]]
        )
        result = np.sum(a, axis=0)
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([4, 2, 6]))
