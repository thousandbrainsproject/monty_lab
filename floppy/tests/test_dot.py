"""Run using python tests/test_dot.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_dot_np_function():
    counter = FlopCounter()
    with counter:
        # Vector dot product
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        _ = np.dot(a, b)
        assert counter.flops == 5  # 3 multiplications + 2 additions


def test_dot_method():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        _ = a.dot(b)
        assert counter.flops == 5


def test_dot_matrix_vector():
    counter = FlopCounter()
    with counter:
        # (2x3) @ (3,)
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([7, 8, 9])
        _ = np.dot(a, b)
        assert counter.flops == 10  # 2 rows * (3 muls + 2 adds)


def test_dot_matrix_matrix():
    counter = FlopCounter()
    with counter:
        # (2x2) @ (2x2)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        _ = np.dot(a, b)
        assert counter.flops == 12  # 4 elements * (2 muls + 1 add)


def test_dot_different_sizes():
    counter = FlopCounter()
    with counter:
        # (2x3) @ (3x2)
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8], [9, 10], [11, 12]])
        _ = np.dot(a, b)
        assert counter.flops == 20  # 4 elements * (3 muls + 2 adds)


def test_dot_1d_scalar():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        _ = np.dot(a, b)
        assert counter.flops == 3  # 3 multiplications, no additions


def test_dot_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        b = np.array([])
        _ = np.dot(a, b)
        assert counter.flops == 0


if __name__ == "__main__":
    test_dot_np_function()
    test_dot_method()
    test_dot_matrix_vector()
    test_dot_matrix_matrix()
    test_dot_different_sizes()
    test_dot_1d_scalar()
    test_dot_empty()
