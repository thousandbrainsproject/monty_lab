"""Run using python tests/test_tangent.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_tan_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        _ = np.tan(a)
        assert counter.flops == 51  # 17 flops * 3 elements


def test_tan_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        _ = np.tan(a)
        assert counter.flops == 17  # 17 flops * 1 element

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.tan(a)
        assert counter.flops == 68  # 17 flops * 4 elements


def test_tan_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.tan(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_tan_ufunc_syntax()
    test_tan_broadcasting()
    test_tan_empty()
