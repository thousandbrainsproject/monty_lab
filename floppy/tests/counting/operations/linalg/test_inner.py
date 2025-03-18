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


def test_inner_1d() -> None:
    """Test inner product behavior and flop count for 1D arrays."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.inner(a, b)
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, 32)


def test_inner_2d() -> None:
    """Test inner product behavior and flop count for 2D arrays."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.inner(a, b)
        assert counter.flops == 12  # noqa: PLR2004
        np.testing.assert_allclose(
            result,
            np.array([[17, 23], [39, 53]]),
        )


def test_inner_1x1() -> None:
    """Test inner product behavior and flop count for 1x1 arrays."""
    counter = FlopCounter()
    with counter:
        a = np.array([1])
        b = np.array([2])
        result = np.inner(a, b)
        assert counter.flops == 1
        np.testing.assert_allclose(result, 2)


def test_inner_empty() -> None:
    """Test inner product behavior and flop count for empty arrays."""
    counter = FlopCounter()
    with counter:
        a = np.array([])
        b = np.array([])
        result = np.inner(a, b)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 0)


def test_inner_batched() -> None:
    """Test inner product behavior and flop count for batched arrays."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.inner(a, b)
        assert counter.flops == 12  # noqa: PLR2004
        np.testing.assert_allclose(
            result,
            np.array([[17, 23], [39, 53]]),
        )


def test_inner_3d() -> None:
    """Test inner product behavior and flop count for 3D arrays."""
    counter = FlopCounter()
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        result = np.inner(a, b)
        assert counter.flops == 48  # noqa: PLR2004
        expected = np.array(
            [
                [
                    [[29, 35], [41, 47]],  # First 2x2 block
                    [[67, 81], [95, 109]],
                ],
                [
                    [[105, 127], [149, 171]],  # Second 2x2 block
                    [[143, 173], [203, 233]],
                ],
            ]
        )
        np.testing.assert_allclose(result, expected)
