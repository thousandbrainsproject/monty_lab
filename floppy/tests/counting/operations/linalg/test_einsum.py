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


def test_einsum_matrix_mult() -> None:
    """Test matrix multiplication behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.einsum("ij,jk->ik", a, b)
        assert counter.flops == 12  # noqa: PLR2004
        np.testing.assert_allclose(
            result,
            np.array([[19, 22], [43, 50]]),
        )


def test_einsum_trace() -> None:
    """Test trace behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.einsum("ii->", a)
        assert counter.flops == 1
        np.testing.assert_allclose(result, 5)


def test_einsum_dot_product() -> None:
    """Test dot product behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.einsum("i,i->", a, b)
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, 32)


def test_einsum_element_wise() -> None:
    """Test element-wise behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2])
        b = np.array([3, 4])
        result = np.einsum("i,i->i", a, b)
        assert counter.flops == 2  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([3, 8]))


def test_einsum_batched_matrix_mult() -> None:
    """Test batched matrix multiplication behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        result = np.einsum("bij,bjk->bik", a, b)
        assert counter.flops == 24  # noqa: PLR2004
        expected = np.array([[[31, 34], [71, 78]], [[155, 166], [211, 226]]])
        np.testing.assert_allclose(result, expected)


def test_einsum_sum() -> None:
    """Test sum behavior and flop count."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.einsum("ij->", a)
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, 10)  # 1 + 2 + 3 + 4 = 10

