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


def test_degrees_ufunc_syntax() -> None:
    """Test basic array conversion from radians to degrees."""
    counter = FlopCounter()
    with counter:
        rad = np.array([0, np.pi / 4, np.pi / 2, np.pi])
        result = np.degrees(rad)
        assert counter.flops == 4  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([0.0, 45.0, 90.0, 180.0]))


def test_radians_ufunc_syntax() -> None:
    """Test basic array conversion from degrees to radians."""
    counter = FlopCounter()
    with counter:
        deg = np.array([0, 45, 90, 180])
        result = np.radians(deg)
        assert counter.flops == 4  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([0.0, np.pi / 4, np.pi / 2, np.pi]))


def test_degrees_radians_broadcasting() -> None:
    """Test scalar and 2D array conversion from radians to degrees."""
    counter = FlopCounter()
    with counter:
        _ = np.degrees(np.pi / 4)
        assert counter.flops == 1

    counter.flops = 0
    with counter:
        rad = np.array([[0, np.pi / 4], [np.pi / 2, np.pi]])
        _ = np.degrees(rad)
        assert counter.flops == 4  # noqa: PLR2004

    counter.flops = 0
    with counter:
        _ = np.radians(45)
        assert counter.flops == 1

    counter.flops = 0
    with counter:
        deg = np.array([[0, 45], [90, 180]])
        _ = np.radians(deg)
        assert counter.flops == 4  # noqa: PLR2004


def test_degrees_radians_empty() -> None:
    """Test empty array conversion from radians to degrees."""
    counter = FlopCounter()
    with counter:
        _ = np.degrees(np.array([]))
        assert counter.flops == 0

    counter.flops = 0
    with counter:
        _ = np.radians(np.array([]))
        assert counter.flops == 0


def test_degrees_radians_roundtrip() -> None:
    """Test roundtrip conversion: degrees -> radians -> degrees."""
    counter = FlopCounter()
    with counter:
        deg = np.array([0, 45, 90, 180, 360])
        rad = np.radians(deg)
        deg_roundtrip = np.degrees(rad)
        assert counter.flops == 10  # noqa: PLR2004
        np.testing.assert_allclose(deg, deg_roundtrip)

