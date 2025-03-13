import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_inv_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 0], [0, 1]])
        _ = np.linalg.inv(a)
        assert counter.flops == 13  # Basic 2x2 matrix inversion


def test_inv_3x3():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 0, 2], [0, 1, 0], [2, 0, 1]])
        _ = np.linalg.inv(a)
        assert counter.flops == 36  # 3x3 matrix inversion


def test_inv_4x4():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [2, 0, 0, 1]])
        _ = np.linalg.inv(a)
        assert counter.flops == 74  # 4x4 matrix inversion


def test_inv_identity():
    counter = FlopCounter()
    with counter:
        a = np.eye(3)  # 3x3 identity matrix
        _ = np.linalg.inv(a)
        assert counter.flops == 36  # Same as regular 3x3


def test_inv_symmetric():
    counter = FlopCounter()
    with counter:
        a = np.array([[2, 1], [1, 2]])  # Symmetric matrix
        _ = np.linalg.inv(a)
        assert counter.flops == 13  # Same as regular 2x2


def test_inv_1x1():
    counter = FlopCounter()
    with counter:
        a = np.array([[4.0]])
        _ = np.linalg.inv(a)
        assert counter.flops == 1  # Simple reciprocal for 1x1
