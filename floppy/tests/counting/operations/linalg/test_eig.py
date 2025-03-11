import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_eig_2x2():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _, _ = np.linalg.eig(a)
        assert counter.flops == 240  # 30 * 2^3


def test_eig_3x3():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        _, _ = np.linalg.eig(a)
        assert counter.flops == 810  # 30 * 3^3


def test_eig_4x4():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        _, _ = np.linalg.eig(a)
        assert counter.flops == 1920  # 30 * 4^3


def test_eig_1x1():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[5]])
        _, _ = np.linalg.eig(a)
        assert counter.flops == 30  # 30 * 1^3


def test_eig_symmetric():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2], [2, 4]])  # Symmetric matrix
        _, _ = np.linalg.eig(a)
        assert counter.flops == 240  # Same as non-symmetric case


def test_eig_diagonal():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 0], [0, 4]])  # Diagonal matrix
        _, _ = np.linalg.eig(a)
        assert counter.flops == 240  # Same complexity as non-diagonal


def test_eig_identity():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.eye(3)  # 3x3 identity matrix
        _, _ = np.linalg.eig(a)
        assert counter.flops == 810  # 30 * 3^3


def test_eig_zero():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.zeros((2, 2))  # Zero matrix
        _, _ = np.linalg.eig(a)
        assert counter.flops == 240  # 30 * 2^3
