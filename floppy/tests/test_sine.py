"""Run using python tests/test_sin.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_sin_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        _ = np.sin(a)
        assert counter.flops == 24


def test_sin_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        _ = np.sin(a)
        assert counter.flops == 8

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.sin(a)
        assert counter.flops == 32


def test_sin_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.sin(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_sin_ufunc_syntax()
    test_sin_broadcasting()
    test_sin_empty()
