import numpy as np

from floppy.counting.counter import FlopCounter


def test_sum_np_function():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3, 4])
        result = np.sum(a)
        assert counter.flops == 3  # n-1 additions for n elements
        np.testing.assert_equal(result, 10)


@pytest.mark.xfail(reason="TrackedArray object has no attribute 'sum'")
def test_sum_method():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3, 4])
        result = a.sum()
        assert counter.flops == 3
        np.testing.assert_equal(result, 10)


def test_sum_axis():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.sum(a, axis=0)  # Sum columns
        assert counter.flops == 5
        np.testing.assert_equal(result, np.array([5, 7, 9]))

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.sum(a, axis=1)  # Sum rows
        assert counter.flops == 3
        np.testing.assert_equal(result, np.array([3, 7]))


def test_sum_keepdims():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.sum(a, keepdims=True)
        assert counter.flops == 5
        np.testing.assert_equal(result, np.array([[21]]))


def test_sum_where():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3, 4])
        mask = np.array([True, False, True, False])
        result = np.sum(a, where=mask)
        assert counter.flops == 3
        np.testing.assert_equal(result, 4)


def test_sum_empty():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([])
        result = np.sum(a)
        assert counter.flops == 0
        np.testing.assert_equal(result, 0)
