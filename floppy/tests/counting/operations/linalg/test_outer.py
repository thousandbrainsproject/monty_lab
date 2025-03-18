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


def test_outer_2x3() -> None:
    """Test outer product behavior and flop count for 2x3 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2])
        b = np.array([3, 4, 5])
        result = np.outer(a, b)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(
            result,
            np.array([[3, 4, 5], [6, 8, 10]]),
        )


def test_outer_3x2() -> None:
    """Test outer product behavior and flop count for 3x2 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        result = np.outer(a, b)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(
            result,
            np.array([[4, 5], [8, 10], [12, 15]]),
        )


def test_outer_1x1() -> None:
    """Test outer product behavior and flop count for 1x1 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([1])
        b = np.array([2])
        result = np.outer(a, b)
        assert counter.flops == 1
        np.testing.assert_allclose(result, np.array([[2]]))


def test_outer_empty() -> None:
    """Test outer product behavior and flop count for empty matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([])
        b = np.array([1, 2])
        result = np.outer(a, b)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]).reshape(0, 2))
