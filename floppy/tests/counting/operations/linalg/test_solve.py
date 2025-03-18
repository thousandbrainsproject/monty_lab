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


def test_solve_2x2() -> None:
    """Test solve behavior and flop count for 2x2 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([5, 6])
        result = np.linalg.solve(a, b)
        assert counter.flops == 13  # noqa: PLR2004
        np.testing.assert_allclose(
            result,
            np.array([-4, 4.5]),
        )


def test_solve_3x3() -> None:
    """Test solve behavior and flop count for 3x3 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([10, 11, 12])
        result = np.linalg.solve(a, b)
        assert counter.flops == 36  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([-25.33333333, 41.66666667, -16.0]))


def test_solve_1x1() -> None:
    """Test solve behavior and flop count for 1x1 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[2]])
        b = np.array([4])
        result = np.linalg.solve(a, b)
        assert counter.flops == 1
        np.testing.assert_allclose(result, np.array([2]))


def test_solve_zero() -> None:
    """Test solve behavior and flop count for zero matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 0], [0, 1]])
        b = np.array([0, 0])
        result = np.linalg.solve(a, b)
        assert counter.flops == 13  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([0, 0]))


def test_solve_identity() -> None:
    """Test solve behavior and flop count for identity matrix."""
    counter = FlopCounter()
    with counter:
        a = np.eye(3)
        b = np.array([1, 2, 3])
        result = np.linalg.solve(a, b)
        assert counter.flops == 36  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([1, 2, 3]))


def test_solve_batched() -> None:
    """Test solve behavior and flop count for batched matrices."""
    counter = FlopCounter()
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = np.array([[5, 6], [7, 8]])
        result = np.linalg.solve(a, b)
        assert counter.flops == 21  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[-4.0, 4.5], [-4.0, 4.5]]))

