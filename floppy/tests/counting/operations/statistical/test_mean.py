import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_mean_scalar():
    """Test mean of scalar value."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array(5)  # scalar array
        _ = np.mean(x)
    assert counter.flops == 1  # scalar mean is just a copy


def test_mean_scalar_python():
    """Test mean of Python scalar."""
    counter = FlopCounter(test_mode=True)
    with counter:
        _ = np.mean(5)
    assert counter.flops == 1  # scalar mean is just a copy


def test_mean_1d():
    """Test mean of 1D array."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        _ = np.mean(x)
    assert counter.flops == 5  # n elements = 5 FLOPs


def test_mean_2d():
    """Test mean of 2D array."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.mean(x)
    assert counter.flops == 6  # 2*3 = 6 elements


def test_mean_3d():
    """Test mean of 3D array."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.ones((2, 3, 4))
        _ = np.mean(x)
    assert counter.flops == 24  # 2*3*4 = 24 elements


def test_mean_empty():
    """Test mean of empty array."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([])
        try:
            _ = np.mean(x)
        except RuntimeWarning:
            pass
    assert counter.flops == 0


def test_mean_single():
    """Test mean of single element."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([1])
        _ = np.mean(x)
    assert counter.flops == 1


def test_mean_axis():
    """Test mean with axis argument."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.mean(x, axis=0)  # mean of each column
    assert counter.flops == 6  # 2 elements per column * 3 columns


def test_mean_keepdims():
    """Test mean with keepdims=True."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.mean(x, keepdims=True)
    assert counter.flops == 6  # same FLOPs as without keepdims


def test_mean_dtype():
    """Test mean with dtype argument."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        _ = np.mean(x, dtype=np.float64)
    assert counter.flops == 6  # dtype doesn't affect FLOP count


@pytest.mark.xfail(reason="Method call not supported yet")
def test_mean_method():
    """Test array.mean() method call."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([1, 2, 3, 4])
        _ = x.mean()
    assert counter.flops == 4  # n=4 elements


def test_mean_broadcast():
    """Test mean with broadcasting."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        _ = np.mean(x + y)  # broadcast y to x's shape then mean
    assert counter.flops == 12  # 6 FLOPs for addition + 6 FLOPs for mean


def test_mean_multi_axis():
    """Test mean with multiple axes."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.ones((2, 3, 4))
        _ = np.mean(x, axis=(0, 2))
    assert counter.flops == 24  # total elements remain same
