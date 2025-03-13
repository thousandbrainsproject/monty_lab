import numpy as np

from floppy.counting.counter import FlopCounter


def test_median_np_function():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4, 5])
        result = np.median(a)
        assert counter.flops == 0  # odd length array, no FLOPs needed
        np.testing.assert_equal(result, 3)


def test_median_even_length():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        result = np.median(a)
        assert counter.flops == 2  # even length array: 1 addition + 1 division
        np.testing.assert_equal(result, 2.5)


def test_median_method():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        result = np.median(a)
        assert counter.flops == 2
        np.testing.assert_equal(result, 2.5)


def test_median_axis():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.median(a, axis=0)  # Median along columns
        assert (
            counter.flops == 2
        )  # even length arrays: 1 addition + 1 division per column
        np.testing.assert_equal(result, np.array([2.5, 3.5, 4.5]))

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.median(a, axis=1)  # Median along rows
        assert counter.flops == 2  # even length arrays: 1 addition + 1 division per row
        np.testing.assert_equal(result, np.array([1.5, 3.5]))


def test_median_keepdims():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.median(a, keepdims=True)
        assert counter.flops == 2  # even length array: 1 addition + 1 division
        np.testing.assert_equal(result, np.array([[3.5]]))


def test_median_where():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        mask = np.array([True, False, True, False])
        result = np.median(a, where=mask)
        assert counter.flops == 0  # odd length array after masking
        np.testing.assert_equal(result, 2)


def test_median_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.median(a)
        assert counter.flops == 0
        np.testing.assert_equal(result, np.nan)


def test_median_single_element():
    counter = FlopCounter()
    with counter:
        a = np.array([1])
        result = np.median(a)
        assert counter.flops == 0  # single element, no FLOPs needed
        np.testing.assert_equal(result, 1)
