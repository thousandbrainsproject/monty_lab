"""Run using python tests/test_cross.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_cross_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        _ = np.cross(a, b)
        assert counter.flops == 9  # 9 flops for one cross product


def test_cross_multiple():
    counter = FlopCounter()
    with counter:
        # Multiple cross products at once
        a = np.array([[1, 0, 0], [2, 0, 0]])
        b = np.array([[0, 1, 0], [0, 2, 0]])
        _ = np.cross(a, b)
        assert counter.flops == 18  # 9 flops * 2 cross products


def test_cross_broadcasting():
    counter = FlopCounter()
    with counter:
        # Broadcasting a single vector against multiple vectors
        a = np.array([1, 0, 0])
        b = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]])
        _ = np.cross(a, b)
        assert counter.flops == 27  # 9 flops * 3 cross products


def test_cross_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([]).reshape(0, 3)
        b = np.array([]).reshape(0, 3)
        _ = np.cross(a, b)
        assert counter.flops == 0


if __name__ == "__main__":
    test_cross_basic()
    test_cross_multiple()
    test_cross_broadcasting()
    test_cross_empty()
