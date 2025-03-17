# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np

from floppy.counting.base import FlopCounter


def test_std_scalar():
    """Test std of scalar value."""
    counter = FlopCounter()
    with counter:
        x = np.array(5)  # scalar array
        result = np.std(x)
        assert counter.flops == 0  # scalar inputs return 0 FLOPs
        np.testing.assert_allclose(result, 0)


def test_std_scalar_python():
    """Test std of Python scalar."""
    counter = FlopCounter()
    with counter:
        result = np.std(5)
        assert counter.flops == 0  # scalar inputs return 0 FLOPs
        np.testing.assert_allclose(result, 0)


def test_std_1d():
    """Test std of 1D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        result = np.std(x)
        assert counter.flops == 40  # 4*5 + 20 FLOPs for 5 elements
        np.testing.assert_allclose(result, 1.41421356)


def test_std_2d():
    """Test std of 2D array."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.std(x)
        assert counter.flops == 44  # 4*6 + 20 FLOPs for 6 elements
        np.testing.assert_allclose(result, 1.707825127659933)


def test_std_single():
    """Test std of single element."""
    counter = FlopCounter()
    with counter:
        x = np.array([1])
        result = np.std(x)
        assert counter.flops == 24  # 4*1 + 20 FLOPs for single element
        np.testing.assert_allclose(result, 0)


def test_std_axis():
    """Test std with axis argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.std(x, axis=0)  # std of each column
        assert counter.flops == 44  # 4*6 + 20 FLOPs for 6 elements
        np.testing.assert_allclose(result, np.array([1.5, 1.5, 1.5]))


def test_std_keepdims():
    """Test std with keepdims=True."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.std(x, keepdims=True)
        assert counter.flops == 44  # 4*6 + 20 FLOPs for 6 elements
        np.testing.assert_allclose(
            result,
            np.array([[1.707825127659933]]),
        )


def test_std_dtype():
    """Test std with dtype argument."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = np.std(x, dtype=np.float64)
        assert counter.flops == 44  # 4*6 + 20 FLOPs for 6 elements
        np.testing.assert_allclose(
            result,
            np.array([[1.707825127659933, 1.707825127659933, 1.707825127659933]]),
        )


def test_std_method():
    """Test array.std() method call."""
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4])
        result = x.std()
        assert counter.flops == 36  # 4*4 + 20 FLOPs for 4 elements
        np.testing.assert_allclose(result, 1.118033988749895)


def test_std_broadcast():
    """Test std with broadcasting."""
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        result = np.std(x + y)  # broadcast y to x's shape then std
        assert counter.flops == 50  # 6 FLOPs for addition + (4*6 + 20) FLOPs for std
        np.testing.assert_allclose(result, 2.217355782608345)


def test_std_multi_axis():
    """Test std with multiple axes."""
    counter = FlopCounter()
    with counter:
        x = np.ones((2, 3, 4))
        result = np.std(x, axis=(0, 2))
        assert counter.flops == 116  # 4*24 + 20 FLOPs for 24 elements
        np.testing.assert_allclose(result, np.array([0.0, 0.0, 0.0]))
