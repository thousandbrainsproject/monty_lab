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


def test_degrees_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        # Test basic array conversion from radians to degrees
        rad = np.array([0, np.pi / 4, np.pi / 2, np.pi])
        result = np.degrees(rad)
        assert counter.flops == 4  # 1 FLOP per element * 4 elements
        np.testing.assert_allclose(result, np.array([0.0, 45.0, 90.0, 180.0]))


def test_radians_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        # Test basic array conversion from degrees to radians
        deg = np.array([0, 45, 90, 180])
        result = np.radians(deg)
        assert counter.flops == 4  # 1 FLOP per element * 4 elements
        np.testing.assert_allclose(result, np.array([0.0, np.pi / 4, np.pi / 2, np.pi]))


def test_degrees_radians_broadcasting():
    counter = FlopCounter()
    with counter:
        # Test scalar conversion from radians to degrees
        _ = np.degrees(np.pi / 4)
        assert counter.flops == 1  # 1 FLOP for scalar operation

    counter.flops = 0
    with counter:
        # Test 2D array conversion from radians to degrees
        rad = np.array([[0, np.pi / 4], [np.pi / 2, np.pi]])
        _ = np.degrees(rad)
        assert counter.flops == 4  # 1 FLOP per element * 4 elements

    counter.flops = 0
    with counter:
        # Test scalar conversion from degrees to radians
        _ = np.radians(45)
        assert counter.flops == 1  # 1 FLOP for scalar operation

    counter.flops = 0
    with counter:
        # Test 2D array conversion from degrees to radians
        deg = np.array([[0, 45], [90, 180]])
        _ = np.radians(deg)
        assert counter.flops == 4  # 1 FLOP per element * 4 elements


def test_degrees_radians_empty():
    counter = FlopCounter()
    with counter:
        # Test empty array conversion from radians to degrees
        _ = np.degrees(np.array([]))
        assert counter.flops == 0

    counter.flops = 0
    with counter:
        # Test empty array conversion from degrees to radians
        _ = np.radians(np.array([]))
        assert counter.flops == 0


def test_degrees_radians_roundtrip():
    counter = FlopCounter()
    with counter:
        # Test roundtrip conversion: degrees -> radians -> degrees
        deg = np.array([0, 45, 90, 180, 360])
        rad = np.radians(deg)
        deg_roundtrip = np.degrees(rad)
        assert (
            counter.flops == 10
        )  # 1 FLOP per element * 5 elements for each conversion
        np.testing.assert_allclose(deg, deg_roundtrip)


def test_degrees_radians_special_cases():
    counter = FlopCounter()
    with counter:
        # Test special cases for degrees conversion
        rad = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        result = np.degrees(rad)
        assert counter.flops == 5  # 1 FLOP per element * 5 elements
        np.testing.assert_allclose(result, np.array([-180.0, -90.0, 0.0, 90.0, 180.0]))

    counter.flops = 0
    with counter:
        # Test special cases for radians conversion
        deg = np.array([-180, -90, 0, 90, 180])
        result = np.radians(deg)
        assert counter.flops == 5  # 1 FLOP per element * 5 elements
        np.testing.assert_allclose(
            result, np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
        )
