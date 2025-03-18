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


def test_arcsin_ufunc_syntax() -> None:
    """Test basic array conversion from sine to arcsine."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 0.5, 0.25])
        result = np.arcsin(a)
        assert counter.flops == 99  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([1.57079633, 0.52359878, 0.25268026])
        )


def test_arcsin_broadcasting() -> None:
    """Test arcsine of 2D array."""
    counter = FlopCounter()
    with counter:
        a = -0.5
        result = np.arcsin(a)
        assert counter.flops == 33  # noqa: PLR2004
        np.testing.assert_allclose(result, -0.5235987755982988)

    counter.flops = 0
    with counter:
        a = np.array([[1, -1], [0, -0.25]])
        result = np.arcsin(a)
        assert counter.flops == 132  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([[1.57079633, -1.57079633], [0, -0.25268026]])
        )

def test_arcsin_empty() -> None:
    """Test arcsine of empty array."""
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.arcsin(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]))
