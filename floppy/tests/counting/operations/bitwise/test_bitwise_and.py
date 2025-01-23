import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_bitwise_and_operator_syntax():
    """Test the bitwise AND operator (&) with numpy arrays.

    The bitwise AND compares each pair of numbers bit by bit:
    - Returns 1 for each bit position where both inputs have 1
    - Returns 0 for all other positions

    Example calculation:
        a = [1,    2,    3   ]
        b = [4,    5,    6   ]

        1 & 4 = 0:
            0001 (1)
          & 0100 (4)
            ----
            0000 (0)

        2 & 5 = 0:
            0010 (2)
          & 0101 (5)
            ----
            0000 (0)

        3 & 6 = 2:
            0011 (3)
          & 0110 (6)
            ----
            0010 (2)
    """
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a & b
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([0, 0, 2]))


def test_bitwise_and_ufunc_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.bitwise_and(a, b)
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([0, 0, 2]))


def test_bitwise_and_augmented_assignment():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a &= b
        assert counter.flops == 3
        np.testing.assert_array_equal(a, np.array([0, 0, 2]))


def test_bitwise_and_broadcasting():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a & b
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([0, 2, 2]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b & a
        assert counter.flops == 3
        np.testing.assert_array_equal(result, np.array([0, 2, 2]))
