"""Run using python tests/test_isnan.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_isnan_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1.0, np.nan, 3.0])
        _ = np.isnan(a)
        assert counter.flops == 3  # 1 flop per element


def test_isnan_broadcasting():
    counter = FlopCounter()
    with counter:
        a = np.nan
        _ = np.isnan(a)
        assert counter.flops == 1

    counter.flops = 0
    with counter:
        a = np.array([[1.0, np.nan], [3.0, np.nan]])
        _ = np.isnan(a)
        assert counter.flops == 4


def test_isnan_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.isnan(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_isnan_basic()
    test_isnan_broadcasting()
    test_isnan_empty()
