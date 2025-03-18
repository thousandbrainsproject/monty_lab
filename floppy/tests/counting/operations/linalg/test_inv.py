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


def test_inv_basic() -> None:
    """Test basic 2x2 matrix inversion behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 0], [0, 1]])
        result = np.linalg.inv(a)
        assert counter.flops == 13  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[1, 0], [0, 1]]))


def test_inv_3x3() -> None:
    """Test 3x3 matrix inversion behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 0, 2], [0, 1, 0], [2, 0, 1]])
        result = np.linalg.inv(a)
        assert counter.flops == 36  # noqa: PLR2004
        np.testing.assert_allclose(
            result,
            np.array(
                [
                    [-0.33333333, 0.0, 0.66666667],
                    [0.0, 1.0, 0.0],
                    [0.66666667, 0.0, -0.33333333],
                ]
            ),
        )


def test_inv_identity() -> None:
    """Test 3x3 identity matrix inversion behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.eye(3)
        result = np.linalg.inv(a)
        assert counter.flops == 36  # noqa: PLR2004
        np.testing.assert_allclose(result, np.eye(3))


def test_inv_1x1() -> None:
    """Test 1x1 matrix inversion behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.array([[4.0]])
        result = np.linalg.inv(a)
        assert counter.flops == 1
        np.testing.assert_allclose(result, np.array([[0.25]]))
