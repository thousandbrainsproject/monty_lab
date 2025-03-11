import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_average_scalar():
    """Test average of scalar value."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array(5)  # scalar array
        _ = np.average(x)
    assert counter.flops == 2  # 1 addition + 1 division


def test_average_scalar_python():
    """Test average of Python scalar."""
    counter = FlopCounter(test_mode=True)
    with counter:
        _ = np.average(5)
    assert counter.flops == 2  # 1 addition + 1 division


def test_average_1d():
    """Test average of 1D array."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        _ = np.average(x)
    assert counter.flops == 6  # 5 additions + 1 division


def test_average_2d():
    """Test average of 2D array."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.average(x)
    assert counter.flops == 7  # 6 additions + 1 division


def test_average_single():
    """Test average of single element."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([1])
        _ = np.average(x)
    assert counter.flops == 2  # 1 addition + 1 division


def test_average_weighted_1d():
    """Test weighted average of 1D array."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        _ = np.average(x, weights=weights)
    assert counter.flops == 11  # 5 multiplications + 5 additions + 1 division


def test_average_weighted_2d():
    """Test weighted average of 2D array."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        weights = np.array([0.5, 0.5])
        _ = np.average(x, weights=weights, axis=0)  # average along first axis
    assert counter.flops == 13  # weighted sum (2 * 6) + division (1)


def test_average_axis():
    """Test average with axis argument."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.average(x, axis=0)  # average of each column
    assert counter.flops == 7  # 6 additions + 1 division


def test_average_broadcast():
    """Test average with broadcasting."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        _ = np.average(x + y)  # broadcast y to x's shape then average
    assert counter.flops == 13  # 6 FLOPs for addition + (6 additions + 1 division)


def test_average_weighted_broadcast():
    """Test weighted average with broadcasting."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        weights = np.ones_like(x)
        _ = np.average(x + y, weights=weights)
    assert (
        counter.flops == 19
    )  # 6 FLOPs for addition + (6 multiplications + 6 additions + 1 division)


def test_average_multi_axis():
    """Test average with multiple axes."""
    counter = FlopCounter(test_mode=True)
    with counter:
        x = np.ones((2, 3, 4))
        _ = np.average(x, axis=(0, 2))
    assert counter.flops == 25  # 24 additions + 1 division
