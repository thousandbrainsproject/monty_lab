"""Run using python tests/test_arctangent.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_arctan_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        _ = np.arctan(a)
        assert counter.flops == 30  # 10 flops * 3 elements


def test_arctan_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        _ = np.arctan(a)
        assert counter.flops == 10  # 10 flops * 1 element

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.arctan(a)
        assert counter.flops == 40  # 10 flops * 4 elements


def test_arctan_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.arctan(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_arctan_ufunc_syntax()
    test_arctan_broadcasting()
    test_arctan_empty()
