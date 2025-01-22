"""Run using python tests/test_log.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_log_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        _ = np.log(a)
        assert counter.flops == 15  # Assuming 5 flops per log operation


def test_log_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        _ = np.log(a)
        assert counter.flops == 5

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.log(a)
        assert counter.flops == 20  # 5 flops * 4 elements


def test_log_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.log(a)
        assert counter.flops == 0


if __name__ == "__main__":
    test_log_basic()
    test_log_broadcasting()
    test_log_empty()
