"""Run using python tests/test_matmul.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_matmul_np_function():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        _ = np.matmul(a, b)
        assert counter.flops == 12  # 4 * (2 muls + 1 add)


def test_matmul_operator():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        _ = a @ b
        assert counter.flops == 12


def test_matmul_method():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        _ = a.matmul(b)
        assert counter.flops == 12


def test_dot_function():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        _ = np.dot(a, b)
        assert counter.flops == 12


def test_dot_method():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        _ = a.dot(b)
        assert counter.flops == 12


def test_different_sizes():
    counter = FlopCounter()
    with counter:
        # (2x3) @ (3x2)
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8], [9, 10], [11, 12]])
        _ = a @ b
        assert counter.flops == 20  # 4 * (3 muls + 2 adds)


def test_vector_matmul():
    counter = FlopCounter()
    with counter:
        # Matrix @ vector
        a = np.array([[1, 2], [3, 4]])
        b = np.array([5, 6])
        _ = a @ b
        assert counter.flops == 6  # 2 * (2 muls + 1 add)

    counter.flops = 0
    with counter:
        # vector @ Matrix
        _ = b @ a
        assert counter.flops == 6


def test_batch_matmul():
    counter = FlopCounter()
    with counter:
        # Batch matrix multiplication (2 batches of 2x2)
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        _ = a @ b
        assert counter.flops == 24  # 2 batches * 12 flops


def test_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([]).reshape(0, 0)
        b = np.array([]).reshape(0, 0)
        _ = a @ b
        assert counter.flops == 0


if __name__ == "__main__":
    test_matmul_np_function()
    test_matmul_operator()
    test_matmul_method()
    test_dot_function()
    test_dot_method()
    test_different_sizes()
    test_vector_matmul()
    test_batch_matmul()
    test_empty()
