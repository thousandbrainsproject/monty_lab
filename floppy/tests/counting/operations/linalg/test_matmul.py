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


def test_matmul_np_function() -> None:
    """Test matrix multiplication using np.matmul and flop counting."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.matmul(a, b)
        assert counter.flops == 12  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[19, 22], [43, 50]]))


def test_matmul_operator() -> None:
    """Test matrix multiplication using the @ operator and flop counting."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = a @ b
        assert counter.flops == 12  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[19, 22], [43, 50]]))


def test_dot_function() -> None:
    """Test dot product using np.dot and flop counting."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.dot(a, b)
        assert counter.flops == 12  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[19, 22], [43, 50]]))


def test_dot_method() -> None:
    """Test dot product using the .dot method and flop counting."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = a.dot(b)
        assert counter.flops == 12  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[19, 22], [43, 50]]))


def test_different_sizes() -> None:
    """Test matrix multiplication with different sizes and flop counting."""
    counter = FlopCounter()
    with counter:
        # (2x3) @ (3x2)
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8], [9, 10], [11, 12]])
        result = a @ b
        assert counter.flops == 20  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([[58, 64], [139, 154]]))


def test_vector_matmul() -> None:
    """Test matrix multiplication with a vector and flop counting."""
    counter = FlopCounter()
    with counter:
        # Matrix @ vector
        a = np.array([[1, 2], [3, 4]])
        b = np.array([5, 6])
        result = a @ b
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([17, 39]))

    counter.flops = 0
    with counter:
        # vector @ Matrix
        result = b @ a
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([23, 34]))


def test_batch_matmul() -> None:
    """Test batch matrix multiplication and flop counting."""
    counter = FlopCounter()
    with counter:
        # Batch matrix multiplication (2 batches of 2x2)
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        result = a @ b
        assert counter.flops == 24  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([[[31, 34], [71, 78]], [[155, 166], [211, 226]]])
        )


def test_empty() -> None:
    """Test matrix multiplication with empty arrays and flop counting."""
    counter = FlopCounter()
    with counter:
        a = np.array([]).reshape(0, 0)
        b = np.array([]).reshape(0, 0)
        result = a @ b
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]).reshape(0, 0))
