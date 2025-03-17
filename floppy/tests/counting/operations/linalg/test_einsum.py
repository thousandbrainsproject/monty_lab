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


def test_einsum_matrix_mult():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.einsum("ij,jk->ik", a, b)
        assert counter.flops == 12
        np.testing.assert_allclose(
            result,
            np.array([[19, 22], [43, 50]]),
        )


def test_einsum_trace():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.einsum("ii->", a)
        assert counter.flops == 1
        np.testing.assert_allclose(result, 5)


def test_einsum_dot_product():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.einsum("i,i->", a, b)
        assert counter.flops == 5  # 3 multiplications + 2 additions
        np.testing.assert_allclose(result, 32)  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32


def test_einsum_element_wise():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2])
        b = np.array([3, 4])
        result = np.einsum("i,i->i", a, b)
        assert counter.flops == 2  # 2 multiplications
        np.testing.assert_allclose(result, np.array([3, 8]))  # [1*3, 2*4]


def test_einsum_batched_matrix_mult():
    counter = FlopCounter()
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # shape: (2, 2, 2)
        b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])  # shape: (2, 2, 2)
        result = np.einsum("bij,bjk->bik", a, b)  # shape: (2, 2, 2)
        assert counter.flops == 24  # 2 * (2*2*2 multiplications + 2*2 additions)
        expected = np.array([[[31, 34], [71, 78]], [[155, 166], [211, 226]]])
        np.testing.assert_allclose(result, expected)


def test_einsum_sum():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.einsum("ij->", a)
        assert counter.flops == 3  # 3 additions (4-1)
        np.testing.assert_allclose(result, 10)  # 1 + 2 + 3 + 4 = 10


def test_einsum_zero():
    counter = FlopCounter()
    with counter:
        a = np.array([[0, 1], [2, 0]])
        b = np.array([[0, 2], [3, 0]])
        result = np.einsum("ij,jk->ik", a, b)
        assert counter.flops == 12
        np.testing.assert_allclose(
            result,
            np.array([[3, 0], [0, 4]]),
        )
