"""Run using python tests/test_bitwise_or.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_bitwise_or_operator_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        _ = a | b
        assert counter.flops == 3


def test_bitwise_or_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        _ = np.bitwise_or(a, b)
        assert counter.flops == 3


def test_bitwise_or_augmented_assignment():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a |= b
        assert counter.flops == 3


def test_bitwise_or_broadcasting():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        _ = a | b
        assert counter.flops == 3

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        _ = b | a
        assert counter.flops == 3


if __name__ == "__main__":
    test_bitwise_or_operator_syntax()
    test_bitwise_or_ufunc_syntax()
    test_bitwise_or_augmented_assignment()
    test_bitwise_or_broadcasting()
