import numpy as np

from floppy.counting.counter import FlopCounter


def test_diff_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 4, 7, 0])
        result = np.diff(a)
        assert counter.flops == 4  # One subtraction per element in result
        np.testing.assert_equal(result, np.array([1, 2, 3, -7]))


def test_diff_2d():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.diff(a, axis=0)  # Diff along rows
        assert counter.flops == 3  # One subtraction per element in result
        np.testing.assert_equal(result, np.array([[3, 3, 3]]))

    counter.flops = 0
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.diff(a, axis=1)  # Diff along columns
        assert counter.flops == 4  # One subtraction per element in result
        np.testing.assert_equal(result, np.array([[1, 1], [1, 1]]))


def test_diff_n():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 4, 7, 0])
        result = np.diff(a, n=2)  # Second difference
        assert counter.flops == 3  # One subtraction per element in result
        np.testing.assert_equal(result, np.array([1, 1, -10]))


def test_diff_prepend():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 4, 7, 0])
        result = np.diff(a, prepend=0)
        assert counter.flops == 5  # One subtraction per element in result
        np.testing.assert_equal(result, np.array([1, 1, 2, 3, -7]))


def test_diff_append():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 4, 7, 0])
        result = np.diff(a, append=0)
        assert counter.flops == 5  # One subtraction per element in result
        np.testing.assert_equal(result, np.array([1, 2, 3, -7, 0]))


def test_diff_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.diff(a)
        assert counter.flops == 0
        np.testing.assert_equal(result, np.array([]))


def test_diff_single_element():
    counter = FlopCounter()
    with counter:
        a = np.array([1])
        result = np.diff(a)
        assert counter.flops == 0  # No elements to diff
        np.testing.assert_equal(result, np.array([]))
