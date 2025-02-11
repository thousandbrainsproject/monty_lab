import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_norm_basic():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.norm(a)  # Frobenius norm by default
        assert counter.flops == 8


def test_norm_1d():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        _ = np.linalg.norm(a)  # L2 norm for vector
        assert counter.flops == 16


def test_norm_rectangular():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.linalg.norm(a)
        assert counter.flops == 12


def test_norm_3d():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        _ = np.linalg.norm(a)
        assert counter.flops == 26


def test_norm_empty():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[]])
        _ = np.linalg.norm(a)
        assert counter.flops == 0


def test_norm_l1():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        _ = np.linalg.norm(a, ord=1)  # L1 norm
        assert counter.flops == 5


def test_norm_l2():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        _ = np.linalg.norm(a, ord=2)  # L2 norm
        assert counter.flops == 16


def test_norm_max():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        _ = np.linalg.norm(a, ord=np.inf)  # Max norm
        assert counter.flops == 5


def test_matrix_norm_l1():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.norm(a, ord=1)  # Maximum column sum
        assert counter.flops == 5


def test_matrix_norm_inf():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.norm(a, ord=np.inf)  # Maximum row sum
        assert counter.flops == 5


def test_norm_nuclear():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.norm(a, ord="nuc")  # Nuclear norm
        assert counter.flops == 114


def test_norm_spectral():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.norm(a, ord=2)  # Spectral norm
        assert counter.flops == 112
