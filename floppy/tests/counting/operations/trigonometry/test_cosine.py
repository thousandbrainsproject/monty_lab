"""Run using python tests/test_cosine.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_cos_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        _ = np.cos(a)
        assert counter.flops == 24  # 8 flops * 3 elements


def test_cos_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        _ = np.cos(a)
        assert counter.flops == 8  # 8 flops * 1 element

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.cos(a)
        assert counter.flops == 32  # 8 flops * 4 elements


def test_cos_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.cos(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_cos_ufunc_syntax()
    test_cos_broadcasting()
    test_cos_empty()
