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


def test_tan_ufunc_syntax() -> None:
    """Test basic array conversion from tangent to arctangent."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.tan(a)
        assert counter.flops == 60  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([1.55740772, -2.18503986, -0.14254654])
        )


def test_tan_broadcasting() -> None:
    """Test tangent of 2D array."""
    counter = FlopCounter()
    with counter:
        a = 2
        _ = np.tan(a)
        assert counter.flops == 20  # noqa: PLR2004

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.tan(a)
        assert counter.flops == 80  # noqa: PLR2004


def test_tan_empty() -> None:
    """Test tangent of empty array."""
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.tan(a)
        assert counter.flops == 0
