import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_std_scalar():
    """Test std of scalar value."""
    counter = FlopCounter()
    with counter:
        x = np.array(5)  # scalar array
        _ = np.std(x)
    assert counter.flops == 5  # 4*1 + 1 FLOPs for single element


def test_std_scalar_python():
    """Test std of Python scalar."""
    counter = FlopCounter()
    with counter:
        _ = np.std(5)
    assert counter.flops == 5  # 4*1 + 1 FLOPs for single element


def test_std_1d():
    """Test std of 1D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        _ = np.std(x)
    assert counter.flops == 21  # 4*5 + 1 FLOPs for 5 elements


def test_std_2d():
    """Test std of 2D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.std(x)
    assert counter.flops == 25  # 4*6 + 1 FLOPs for 6 elements


def test_std_single():
    """Test std of single element."""
    counter = FlopCounter()
    with counter:
        x = np.array([1])
        _ = np.std(x)
    assert counter.flops == 5  # 4*1 + 1 FLOPs for single element


def test_std_axis():
    """Test std with axis argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.std(x, axis=0)  # std of each column
    assert counter.flops == 25  # 4*6 + 1 FLOPs for 6 elements


def test_std_keepdims():
    """Test std with keepdims=True."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.std(x, keepdims=True)
    assert counter.flops == 25  # 4*6 + 1 FLOPs for 6 elements


def test_std_dtype():
    """Test std with dtype argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        _ = np.std(x, dtype=np.float64)
    assert counter.flops == 25  # 4*6 + 1 FLOPs for 6 elements


def test_std_method():
    """Test array.std() method call."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4])
        _ = x.std()
    assert counter.flops == 17  # 4*4 + 1 FLOPs for 4 elements


def test_std_broadcast():
    """Test std with broadcasting."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        _ = np.std(x + y)  # broadcast y to x's shape then std
    assert counter.flops == 31  # 6 FLOPs for addition + (4*6 + 1) FLOPs for std


def test_std_multi_axis():
    """Test std with multiple axes."""
    counter = FlopCounter()
    with counter:
        x = np.ones((2, 3, 4))
        _ = np.std(x, axis=(0, 2))
    assert counter.flops == 97  # 4*24 + 1 FLOPs for 24 elements
