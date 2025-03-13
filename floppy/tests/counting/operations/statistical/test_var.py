import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_var_scalar():
    """Test var of scalar value."""
    counter = FlopCounter()
    with counter:
        x = np.array(5)  # scalar array
        _ = np.var(x)
    assert counter.flops == 4  # 4*1 FLOPs for single element


def test_var_scalar_python():
    """Test var of Python scalar."""
    counter = FlopCounter()
    with counter:
        _ = np.var(5)
    assert counter.flops == 4  # 4*1 FLOPs for single element


def test_var_1d():
    """Test var of 1D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        _ = np.var(x)
    assert counter.flops == 20  # 4*5 FLOPs for 5 elements


def test_var_2d():
    """Test var of 2D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.var(x)
    assert counter.flops == 24  # 4*6 FLOPs for 6 elements


def test_var_empty():
    """Test var of empty array."""
    counter = FlopCounter()
    with counter:
        x = np.array([])
        try:
            _ = np.var(x)
        except RuntimeWarning:
            pass
    assert counter.flops == 0


def test_var_single():
    """Test var of single element."""
    counter = FlopCounter()
    with counter:
        x = np.array([1])
        _ = np.var(x)
    assert counter.flops == 4  # 4*1 FLOPs for single element


def test_var_axis():
    """Test var with axis argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.var(x, axis=0)  # var of each column
    assert counter.flops == 24  # 4*6 FLOPs for 6 elements


def test_var_keepdims():
    """Test var with keepdims=True."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.var(x, keepdims=True)
    assert counter.flops == 24  # 4*6 FLOPs for 6 elements


def test_var_dtype():
    """Test var with dtype argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        _ = np.var(x, dtype=np.float64)
    assert counter.flops == 24  # 4*6 FLOPs for 6 elements


def test_var_method():
    """Test array.var() method call."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4])
        _ = x.var()
    assert counter.flops == 16  # 4*4 FLOPs for 4 elements


def test_var_broadcast():
    """Test var with broadcasting."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        _ = np.var(x + y)  # broadcast y to x's shape then var
    assert counter.flops == 30  # 6 FLOPs for addition + 4*6 FLOPs for var


def test_var_multi_axis():
    """Test var with multiple axes."""
    counter = FlopCounter()
    with counter:
        x = np.ones((2, 3, 4))
        _ = np.var(x, axis=(0, 2))
    assert counter.flops == 96  # 4*24 FLOPs for 24 elements
