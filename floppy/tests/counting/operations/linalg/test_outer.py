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


def test_outer_2x3():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2])
        b = np.array([3, 4, 5])
        result = np.outer(a, b)
        assert counter.flops == 6  # 2 * 3 = 6 multiplications
        np.testing.assert_allclose(
            result,
            np.array([[3, 4, 5], [6, 8, 10]]),
        )


def test_outer_3x2():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        result = np.outer(a, b)
        assert counter.flops == 6  # 3 * 2 = 6 multiplications
        np.testing.assert_allclose(
            result,
            np.array([[4, 5], [8, 10], [12, 15]]),
        )


def test_outer_1x1():
    counter = FlopCounter()
    with counter:
        a = np.array([1])
        b = np.array([2])
        result = np.outer(a, b)
        assert counter.flops == 1  # 1 * 1 = 1 multiplication
        np.testing.assert_allclose(result, np.array([[2]]))


def test_outer_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        b = np.array([1, 2])
        result = np.outer(a, b)
        assert counter.flops == 0  # Empty result
        np.testing.assert_allclose(result, np.array([]).reshape(0, 2))


def test_outer_zero():
    counter = FlopCounter()
    with counter:
        a = np.array([0, 1])
        b = np.array([0, 2])
        result = np.outer(a, b)
        assert counter.flops == 4  # 2 * 2 = 4 multiplications
        np.testing.assert_allclose(
            result,
            np.array([[0, 0], [0, 2]]),
        )


def test_outer_negative():
    counter = FlopCounter()
    with counter:
        a = np.array([-1, 2])
        b = np.array([-3, 4])
        result = np.outer(a, b)
        assert counter.flops == 4  # 2 * 2 = 4 multiplications
        np.testing.assert_allclose(
            result,
            np.array([[3, -4], [-6, 8]]),
        )


def test_outer_float():
    counter = FlopCounter()
    with counter:
        a = np.array([1.5, 2.5])
        b = np.array([3.5, 4.5])
        result = np.outer(a, b)
        assert counter.flops == 4  # 2 * 2 = 4 multiplications
        np.testing.assert_allclose(
            result,
            np.array([[5.25, 6.75], [8.75, 11.25]]),
        )


def test_outer_large():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([6, 7, 8, 9, 10])
        result = np.outer(a, b)
        assert counter.flops == 25  # 5 * 5 = 25 multiplications
        np.testing.assert_allclose(
            result,
            np.array(
                [
                    [6, 7, 8, 9, 10],
                    [12, 14, 16, 18, 20],
                    [18, 21, 24, 27, 30],
                    [24, 28, 32, 36, 40],
                    [30, 35, 40, 45, 50],
                ]
            ),
        )
