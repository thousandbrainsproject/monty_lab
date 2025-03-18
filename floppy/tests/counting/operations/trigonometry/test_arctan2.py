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


def test_arctan2_ufunc_syntax() -> None:
    """Test basic array conversion from tangent to arctangent."""
    counter = FlopCounter()
    with counter:
        y = np.array([1, 2, 3])
        x = np.array([1, 1, 1])
        result = np.arctan2(y, x)
        assert counter.flops == 120  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([0.78539816, 1.10714872, 1.24904577])
        )


def test_arctan2_broadcasting() -> None:
    """Test arctangent of 2D array."""
    counter = FlopCounter()
    with counter:
        y = 2
        x = 1
        _ = np.arctan2(y, x)
        assert counter.flops == 40  # noqa: PLR2004

    counter.flops = 0
    with counter:
        y = np.array([[1, 2], [3, 4]])
        x = np.array([[1, 1], [1, 1]])
        _ = np.arctan2(y, x)
        assert counter.flops == 160  # noqa: PLR2004


def test_arctan2_empty() -> None:
    """Test arctangent of empty array."""
    counter = FlopCounter()
    with counter:
        y = np.array([])
        x = np.array([])
        _ = np.arctan2(y, x)
        assert counter.flops == 0


def test_arctan2_quadrants() -> None:
    """Test all four quadrants."""
    counter = FlopCounter()
    with counter:
        y = np.array([1, 1, -1, -1])
        x = np.array([1, -1, 1, -1])
        result = np.arctan2(y, x)
        assert counter.flops == 160  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([0.78539816, 2.35619449, -0.78539816, -2.35619449])
        )


def test_arctan2_special_cases() -> None:
    """Test special cases."""
    counter = FlopCounter()
    with counter:
        # Test special cases: x=0, y=0, and both=0
        y = np.array([1, 0, 0])
        x = np.array([0, 1, 0])
        result = np.arctan2(y, x)
        assert counter.flops == 120  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([1.57079633, 0.0, 0.0]))
