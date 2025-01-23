"""Run using python tests/test_arcsine.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_arcsin_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        _ = np.arcsin(a)
        assert counter.flops == 30  # 10 flops * 3 elements


def test_arcsin_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        _ = np.arcsin(a)
        assert counter.flops == 10  # 10 flops * 1 element

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.arcsin(a)
        assert counter.flops == 40  # 10 flops * 4 elements


def test_arcsin_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.arcsin(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_arcsin_ufunc_syntax()
    test_arcsin_broadcasting()
    test_arcsin_empty()
