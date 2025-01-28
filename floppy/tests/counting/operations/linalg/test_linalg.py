"""Run using python tests/test_linalg.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_norm_vector():
    counter = FlopCounter()
    with counter:
        # Vector norm (Frobenius/L2 norm)
        a = np.array([1, 2, 3, 4])
        _ = np.linalg.norm(a)
        assert counter.flops == 8  # 2n FLOPs for n elements


def test_norm_matrix():
    counter = FlopCounter()
    with counter:
        # Matrix Frobenius norm
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.norm(a)
        assert counter.flops == 8  # 2n FLOPs for n total elements


def test_norm_different_ord():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = np.linalg.norm(a, ord=2)  # L2 norm
        assert counter.flops == 8

    counter.flops = 0
    with counter:
        _ = np.linalg.norm(a, ord=np.inf)  # Max norm
        assert counter.flops == 8  # Should still count as 2n FLOPs


def test_cond_basic():
    counter = FlopCounter()
    with counter:
        # 2x2 matrix condition number
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.cond(a)
        assert counter.flops == 14 * 2**3 + 1  # 14n³ + 1 FLOPs for nxn matrix


def test_cond_different_norm():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.cond(a, p="fro")
        assert counter.flops == 14 * 2**3 + 1


def test_inv_basic():
    counter = FlopCounter()
    with counter:
        # 2x2 matrix inversion
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.inv(a)
        assert counter.flops == (2 * 2**3) // 3 + 2 * 2**2  # 2/3 n³ + 2n² FLOPs


def test_inv_method():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.matrix(a).I  # Matrix inverse using matrix class
        assert counter.flops == (2 * 2**3) // 3 + 2 * 2**2


def test_eig_basic():
    counter = FlopCounter()
    with counter:
        # 2x2 eigendecomposition
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.eig(a)
        assert counter.flops == 22 * 2**3  # 22n³ FLOPs


def test_eig_symmetric():
    counter = FlopCounter()
    with counter:
        # Symmetric 2x2 matrix
        a = np.array([[1, 2], [2, 4]])
        _ = np.linalg.eigh(a)  # Using eigh for symmetric matrices
        assert counter.flops == 22 * 2**3  # Should count same as regular eig


def test_larger_matrices():
    n = 4
    counter = FlopCounter()

    # Test norm
    with counter:
        a = np.random.rand(n, n)
        _ = np.linalg.norm(a)
        assert counter.flops == 2 * n * n  # 2n FLOPs for n² elements

    # Test cond
    counter.flops = 0
    with counter:
        _ = np.linalg.cond(a)
        assert counter.flops == 14 * n**3 + 1

    # Test inv
    counter.flops = 0
    with counter:
        _ = np.linalg.inv(a)
        assert counter.flops == (2 * n**3) // 3 + 2 * n**2

    # Test eig
    counter.flops = 0
    with counter:
        _ = np.linalg.eig(a)
        assert counter.flops == 22 * n**3


def test_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([]).reshape(0, 0)
        try:
            _ = np.linalg.norm(a)
        except:
            pass
        assert counter.flops == 0


if __name__ == "__main__":
    test_norm_vector()
    test_norm_matrix()
    test_norm_different_ord()
    test_cond_basic()
    test_cond_different_norm()
    test_inv_basic()
    test_inv_method()
    test_eig_basic()
    test_eig_symmetric()
    test_larger_matrices()
    test_empty()
