import numpy as np
import pytest

from floppy.counting.core import FlopCounter


def test_arctan2_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        y = np.array([1, 2, 3])
        x = np.array([1, 1, 1])
        result = np.arctan2(y, x)
        assert counter.flops == 120  # 40 FLOPs per element * 3 elements
        np.testing.assert_allclose(
            result, np.array([0.78539816, 1.10714872, 1.24904577])
        )


def test_arctan2_broadcasting():
    counter = FlopCounter()
    with counter:
        y = 2
        x = 1
        _ = np.arctan2(y, x)
        assert counter.flops == 40  # 40 FLOPs for scalar operation

    counter.flops = 0
    with counter:
        y = np.array([[1, 2], [3, 4]])
        x = np.array([[1, 1], [1, 1]])
        _ = np.arctan2(y, x)
        assert counter.flops == 160  # 40 FLOPs per element * 4 elements


def test_arctan2_empty():
    counter = FlopCounter()
    with counter:
        y = np.array([])
        x = np.array([])
        _ = np.arctan2(y, x)
        assert counter.flops == 0


def test_arctan2_quadrants():
    counter = FlopCounter()
    with counter:
        # Test all four quadrants
        y = np.array([1, 1, -1, -1])
        x = np.array([1, -1, 1, -1])
        result = np.arctan2(y, x)
        assert counter.flops == 160  # 40 FLOPs per element * 4 elements
        np.testing.assert_allclose(
            result, np.array([0.78539816, 2.35619449, -0.78539816, -2.35619449])
        )


def test_arctan2_special_cases():
    counter = FlopCounter()
    with counter:
        # Test special cases: x=0, y=0, and both=0
        y = np.array([1, 0, 0])
        x = np.array([0, 1, 0])
        result = np.arctan2(y, x)
        assert counter.flops == 120  # 40 FLOPs per element * 3 elements
        np.testing.assert_allclose(result, np.array([1.57079633, 0.0, 0.0]))
