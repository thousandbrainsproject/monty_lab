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


def test_matmul_np_function():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.matmul(a, b)
        assert counter.flops == 12  # 4 * (2 muls + 1 add)
        np.testing.assert_allclose(result, np.array([[19, 22], [43, 50]]))


def test_matmul_operator():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = a @ b
        assert counter.flops == 12
        np.testing.assert_allclose(result, np.array([[19, 22], [43, 50]]))


def test_dot_function():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.dot(a, b)
        assert counter.flops == 12
        np.testing.assert_allclose(result, np.array([[19, 22], [43, 50]]))


def test_dot_method():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = a.dot(b)
        assert counter.flops == 12
        np.testing.assert_allclose(result, np.array([[19, 22], [43, 50]]))


def test_different_sizes():
    counter = FlopCounter()
    with counter:
        # (2x3) @ (3x2)
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8], [9, 10], [11, 12]])
        result = a @ b
        assert counter.flops == 20  # 4 * (3 muls + 2 adds)
        np.testing.assert_allclose(result, np.array([[58, 64], [139, 154]]))


def test_vector_matmul():
    counter = FlopCounter()
    with counter:
        # Matrix @ vector
        a = np.array([[1, 2], [3, 4]])
        b = np.array([5, 6])
        result = a @ b
        assert counter.flops == 6  # 2 * (2 muls + 1 add)
        np.testing.assert_allclose(result, np.array([17, 39]))

    counter.flops = 0
    with counter:
        # vector @ Matrix
        result = b @ a
        assert counter.flops == 6
        np.testing.assert_allclose(result, np.array([23, 34]))


def test_batch_matmul():
    counter = FlopCounter()
    with counter:
        # Batch matrix multiplication (2 batches of 2x2)
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        result = a @ b
        assert counter.flops == 24  # 2 batches * 12 flops
        np.testing.assert_allclose(
            result, np.array([[[31, 34], [71, 78]], [[155, 166], [211, 226]]])
        )


def test_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([]).reshape(0, 0)
        b = np.array([]).reshape(0, 0)
        result = a @ b
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]).reshape(0, 0))
