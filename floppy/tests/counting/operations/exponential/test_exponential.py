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


def test_exponential_scalar():
    counter = FlopCounter()
    with counter:
        x = 2.0
        result = np.exp(x)
        assert counter.flops == 20  # 20 FLOPs for scalar exponential
        np.testing.assert_allclose(result, np.exp(2.0))


def test_exponential_array():
    counter = FlopCounter()
    with counter:
        x = np.array([1.0, 2.0, 3.0])
        result = np.exp(x)
        assert counter.flops == 60  # 20 FLOPs per element
        np.testing.assert_allclose(
            result, np.array([np.exp(1.0), np.exp(2.0), np.exp(3.0)])
        )


def test_exponential_2d_array():
    counter = FlopCounter()
    with counter:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = np.exp(x)
        assert counter.flops == 80  # 20 FLOPs per element (4 elements)
        expected = np.array([[np.exp(1.0), np.exp(2.0)], [np.exp(3.0), np.exp(4.0)]])
        np.testing.assert_allclose(result, expected)


def test_exponential_empty_array():
    counter = FlopCounter()
    with counter:
        x = np.array([])
        result = np.exp(x)
        assert counter.flops == 0  # No FLOPs for empty array
        assert len(result) == 0


def test_exponential_broadcasting():
    counter = FlopCounter()
    with counter:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        result = np.exp(x + y)  # Broadcasting y to match x's shape
        assert counter.flops == 84  # 20 FLOPs per element (4 elements) and 4 additions
        expected = np.array([[np.exp(2.0), np.exp(4.0)], [np.exp(4.0), np.exp(6.0)]])
        np.testing.assert_allclose(result, expected)


def test_exponential_negative_values():
    counter = FlopCounter()
    with counter:
        x = np.array([-1.0, -2.0, -3.0])
        result = np.exp(x)
        assert counter.flops == 60  # 20 FLOPs per element
        np.testing.assert_allclose(
            result, np.array([np.exp(-1.0), np.exp(-2.0), np.exp(-3.0)])
        )


def test_exponential_large_values():
    counter = FlopCounter()
    with counter:
        x = np.array([10.0, 20.0, 30.0])
        result = np.exp(x)
        assert counter.flops == 60  # 20 FLOPs per element
        np.testing.assert_allclose(
            result, np.array([np.exp(10.0), np.exp(20.0), np.exp(30.0)])
        )


def test_exponential_small_values():
    counter = FlopCounter()
    with counter:
        x = np.array([1e-10, 1e-20, 1e-30])
        result = np.exp(x)
        assert counter.flops == 60  # 20 FLOPs per element
        np.testing.assert_allclose(
            result, np.array([np.exp(1e-10), np.exp(1e-20), np.exp(1e-30)])
        )
