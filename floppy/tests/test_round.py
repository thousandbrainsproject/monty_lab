"""Run using python tests/test_round.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_round_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1.4, 2.6, 3.5])
        _ = np.round(a)
        assert counter.flops == 3  # 1 flop per element


def test_round_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2.5
        _ = np.round(a)
        assert counter.flops == 1

    counter.flops = 0
    with counter:
        a = np.array([[1.4, 2.6], [3.5, 4.1]])
        _ = np.round(a)
        assert counter.flops == 4


def test_round_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.round(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_round_basic()
    test_round_broadcasting()
    test_round_empty()
