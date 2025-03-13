import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_add_operator_syntax():
    counter = FlopCounter(test_mode=True)

    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a + b
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([5, 7, 9]))

def test_add_ufunc_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.add(a, b)
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([5, 7, 9]))

def test_add_method_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a.add(b)
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([5, 7, 9]))

def test_add_augmented_assignment():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a += b
        assert counter.flops == 3
        np.testing.assert_array_equal(a, np.array([5, 7, 9]))

def test_add_broadcasting():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a + b
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([3, 4, 5]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b + a
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([3, 4, 5]))

def test_add_within_operation():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.array([7, 8, 9])
        # Addition happens inside np.flipud
        result = np.flipud(a + b + c)
        # 3 adds for first a+b (3 elements)
        # 3 adds for (a+b)+c (3 elements)
        assert counter.flops == 6
        np.testing.assert_array_equal(
            result, np.array([24, 23, 22])
        )  # flipped [12,15,18] + [10,8,6]


def test_add_empty_arrays():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([])
        b = np.array([])
        result = a + b
        assert counter.flops == 0
        assert len(result) == 0


def test_add_with_views():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        # Test with array views/slices
        result = a[::2] + b[::2]  # [1, 3] + [5, 7]
        assert counter.flops == 2
        np.testing.assert_array_equal(result, np.array([6, 10]))


def test_add_non_contiguous():
    counter = FlopCounter(test_mode=True)
    with counter:
        # Create non-contiguous array by transposing
        a = np.array([[1, 2], [3, 4]]).T  # Non-contiguous in memory
        b = np.array([[5, 6], [7, 8]]).T
        result = a + b
        assert counter.flops == 4
        np.testing.assert_array_equal(result, np.array([[6, 8], [9, 11]]))


def test_add_mixed_dtypes():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = a + b  # Should promote to float64
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([5.0, 7.0, 9.0]))


def test_add_with_indexing():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        indices = np.array([0, 2])
        # Addition within fancy indexing
        result = a[indices] + b[indices]  # [1, 3] + [5, 7]
        assert counter.flops == 2
        np.testing.assert_array_equal(result, np.array([6, 10]))
