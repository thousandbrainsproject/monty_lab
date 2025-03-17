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


def test_add_operator_syntax() -> None:
    """Test addition using operator syntax (a + b)."""
    counter = FlopCounter()

    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a + b
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([5, 7, 9]))

def test_add_ufunc_syntax() -> None:
    """Test addition using numpy ufunc syntax (np.add)."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.add(a, b)
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([5, 7, 9]))

def test_add_method_syntax() -> None:
    """Test addition using method syntax (a.add)."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a.add(b)
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([5, 7, 9]))

def test_add_augmented_assignment() -> None:
    """Test addition using augmented assignment (+=)."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a += b
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(a, np.array([5, 7, 9]))

def test_add_broadcasting() -> None:
    """Test addition with broadcasting between array and scalar."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a + b
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([3, 4, 5]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b + a
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([3, 4, 5]))

def test_add_within_operation() -> None:
    """Test addition within a chain of operations."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.array([7, 8, 9])
        result = np.flipud(a + b + c)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([18, 15, 12]))


def test_add_empty_arrays() -> None:
    """Test addition with empty arrays."""
    counter = FlopCounter()
    with counter:
        a = np.array([])
        b = np.array([])
        result = a + b
        assert counter.flops == 0
        assert len(result) == 0


def test_add_with_views() -> None:
    """Test addition with array views using slicing."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        result = a[::2] + b[::2]
        assert counter.flops == 2  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([6, 10]))



def test_add_mixed_dtypes() -> None:
    """Test addition with mixed data types."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = a + b  # Should promote to float64
        assert counter.flops == 3  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([5.0, 7.0, 9.0]))


def test_add_with_indexing() -> None:
    """Test addition with fancy indexing."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        indices = np.array([0, 2])
        result = a[indices] + b[indices]
        assert counter.flops == 2  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([6, 10]))
