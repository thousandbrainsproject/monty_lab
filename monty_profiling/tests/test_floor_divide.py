"""Run using python tests/test_floor_divide.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_floor_divide_operator_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        _ = a // b
        assert counter.flops == 3


def test_floor_divide_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        _ = np.floor_divide(a, b)
        assert counter.flops == 3


def test_floor_divide_augmented_assignment():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a //= b
        assert counter.flops == 3


def test_floor_divide_broadcasting():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        _ = a // b
        assert counter.flops == 3

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        _ = b // a
        assert counter.flops == 3


if __name__ == "__main__":
    test_floor_divide_operator_syntax()
    test_floor_divide_ufunc_syntax()
    test_floor_divide_augmented_assignment()
    test_floor_divide_broadcasting()