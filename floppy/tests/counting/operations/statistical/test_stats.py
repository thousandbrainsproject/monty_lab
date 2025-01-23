"""Run using python tests/test_stats.py. Do not use pytest."""

import numpy as np
from floppy.flop_counting.counter import FlopCounter


def test_mean_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = np.mean(a)
        assert counter.flops == 4  # n-1 adds + 1 division


def test_mean_method():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = a.mean()
        assert counter.flops == 4


def test_mean_axis():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.mean(a, axis=0)  # Column means
        assert counter.flops == 6  # 3 columns * (1 add + 1 div)

    counter.flops = 0
    with counter:
        _ = np.mean(a, axis=1)  # Row means
        assert counter.flops == 8  # 2 rows * (2 adds + 1 div)


def test_std_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = np.std(a)
        # For n elements:
        # - n-1 adds + 1 div for mean
        # - n subtractions from mean
        # - n multiplications for square
        # - n-1 adds for sum
        # - 1 division by (n-1)
        # - 1 sqrt
        assert counter.flops == 19  # 4 + 4 + 4 + 3 + 1 + 3


def test_std_method():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = a.std()
        assert counter.flops == 19


def test_var_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = np.var(a)
        # Same as std but without sqrt
        assert counter.flops == 16  # 4 + 4 + 4 + 3 + 1


def test_var_method():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = a.var()
        assert counter.flops == 16


def test_average_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        _ = np.average(a)
        assert counter.flops == 4  # Same as mean


def test_average_weighted():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        weights = np.array([1, 1, 1, 1])
        _ = np.average(a, weights=weights)
        # For n elements:
        # - n multiplications for weights
        # - n-1 adds for numerator
        # - n-1 adds for denominator
        # - 1 division
        assert counter.flops == 11  # 4 + 3 + 3 + 1


def test_trace_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.trace(a)
        assert counter.flops == 1  # n-1 adds for n diagonal elements


def test_trace_method():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = a.trace()
        assert counter.flops == 1


def test_trace_offset():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        _ = np.trace(a, offset=1)  # Sum of [2, 6]
        assert counter.flops == 1


def test_stats_axis_params():
    counter = FlopCounter()
    a = np.array([[1, 2, 3], [4, 5, 6]])

    # Test std along axis
    with counter:
        _ = np.std(a, axis=0)
        assert counter.flops == 27  # 3 columns * 9 flops per std

    counter.flops = 0
    with counter:
        _ = np.std(a, axis=1)
        assert counter.flops == 38  # 2 rows * 19 flops per std

    # Test var along axis
    counter.flops = 0
    with counter:
        _ = np.var(a, axis=0)
        assert counter.flops == 24  # 3 columns * 8 flops per var


def test_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.mean(a)
        assert counter.flops == 0

        _ = np.std(a)
        assert counter.flops == 0

        _ = np.var(a)
        assert counter.flops == 0

        _ = np.average(a)
        assert counter.flops == 0

        b = np.array([]).reshape(0, 0)
        _ = np.trace(b)
        assert counter.flops == 0


if __name__ == "__main__":
    test_mean_basic()
    test_mean_method()
    test_mean_axis()
    test_std_basic()
    test_std_method()
    test_var_basic()
    test_var_method()
    test_average_basic()
    test_average_weighted()
    test_trace_basic()
    test_trace_method()
    test_trace_offset()
    test_stats_axis_params()
    test_empty()
