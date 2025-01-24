import numpy as np

from floppy.counting.counter import FlopCounter


def test_argmin_basic():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([4, 2, 3, 1])
        result = np.argmin(a)
        assert counter.flops == 3  # n-1 comparisons for n elements
        np.testing.assert_equal(result, 3)

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.argmin(a)
        assert counter.flops == 3
        np.testing.assert_equal(result, 0)


def test_argmin_axis():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.argmin(a, axis=0)
        assert counter.flops == 5
        np.testing.assert_equal(result, np.array([0, 0, 0]))

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.argmin(a, axis=1)
        assert counter.flops == 3
        np.testing.assert_equal(result, np.array([0, 0]))


def test_argmax_basic():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([4, 2, 3, 1])
        result = np.argmax(a)
        assert counter.flops == 3  # n-1 comparisons for n elements
        np.testing.assert_equal(result, 0)

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.argmax(a)
        assert counter.flops == 3
        np.testing.assert_equal(result, 3)


def test_argmax_axis():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.argmax(a, axis=0)
        assert counter.flops == 5
        np.testing.assert_equal(result, np.array([1, 1, 1]))

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.argmax(a, axis=1)
        assert counter.flops == 3
        np.testing.assert_equal(result, np.array([1, 1]))
