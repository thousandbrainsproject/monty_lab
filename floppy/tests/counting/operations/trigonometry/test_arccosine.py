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


def test_arccos_ufunc_syntax() -> None:
    """Test basic array conversion from cosine to arccosine."""
    counter = FlopCounter()
    with counter:
        a = np.array([-0.5, 0.0, 0.5])  # values within valid domain [-1, 1]
        result = np.arccos(a)
        assert counter.flops == 132  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([2.0943951, 1.57079633, 1.04719755])
        )


def test_arccos_scalar() -> None:
    """Test arccosine of scalar value."""
    counter = FlopCounter()
    with counter:
        a = 0.5
        result = np.arccos(a)
        assert counter.flops == 44  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([1.04719755]))


def test_arccos_broadcasting() -> None:
    """Test arccosine of 2D array."""
    counter = FlopCounter()
    with counter:
        a = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = np.arccos(a)
        assert counter.flops == 176  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([[1.47062894, 1.36943841], [1.26610367, 1.15927948]])
        )


def test_arccos_empty() -> None:
    """Test arccosine of empty array."""
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.arccos(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]))
