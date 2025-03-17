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


def test_inner_1d():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.inner(a, b)
        assert counter.flops == 5  # 3 multiplications + 2 additions
        np.testing.assert_allclose(result, 32)  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32


def test_inner_2d():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.inner(a, b)
        assert (
            counter.flops == 12
        )  # 4 inner products * (2 multiplications + 1 addition)
        np.testing.assert_allclose(
            result,
            np.array([[17, 23], [39, 53]]),  # [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
        )


def test_inner_1x1():
    counter = FlopCounter()
    with counter:
        a = np.array([1])
        b = np.array([2])
        result = np.inner(a, b)
        assert counter.flops == 1  # 1 multiplication + 0 additions
        np.testing.assert_allclose(result, 2)


def test_inner_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        b = np.array([])
        result = np.inner(a, b)
        assert counter.flops == 0  # Empty arrays
        np.testing.assert_allclose(result, 0)


def test_inner_batched():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])  # shape: (2, 2)
        b = np.array([[5, 6], [7, 8]])  # shape: (2, 2)
        result = np.inner(a, b)  # shape: (2, 2)
        assert (
            counter.flops == 12
        )  # 4 inner products * (2 multiplications + 1 addition)
        np.testing.assert_allclose(
            result,
            np.array([[17, 23], [39, 53]]),  # [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
        )


def test_inner_3d():
    counter = FlopCounter()
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # shape: (2, 2, 2)
        b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])  # shape: (2, 2, 2)
        result = np.inner(a, b)  # shape: (2, 2, 2, 2)
        assert (
            counter.flops == 48
        )  # 16 inner products * (2 multiplications + 1 addition)
        # The result is a 4D array with shape (2, 2, 2, 2)
        # Each element is the inner product of the corresponding 2D slices
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
