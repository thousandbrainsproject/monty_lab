# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
import pytest

from floppy.counting.base import FlopCounter


def test_add_operator_syntax():
    counter = FlopCounter()

    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a + b
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([5, 7, 9]))

def test_add_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.add(a, b)
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([5, 7, 9]))

def test_add_method_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a.add(b)
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([5, 7, 9]))

def test_add_augmented_assignment():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a += b
        assert counter.flops == 3
        np.testing.assert_allclose(a, np.array([5, 7, 9]))

def test_add_broadcasting():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a + b
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([3, 4, 5]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b + a
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([3, 4, 5]))

def test_add_within_operation():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.array([7, 8, 9])
        result = np.flipud(a + b + c)
        assert counter.flops == 6
        np.testing.assert_allclose(result, np.array([18, 15, 12]))


def test_add_empty_arrays():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        b = np.array([])
        result = a + b
        assert counter.flops == 0
        assert len(result) == 0


def test_add_with_views():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        # Test with array views/slices
        result = a[::2] + b[::2]  # [1, 3] + [5, 7]
        assert counter.flops == 2
        np.testing.assert_allclose(result, np.array([6, 10]))



def test_add_mixed_dtypes():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = a + b  # Should promote to float64
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([5.0, 7.0, 9.0]))


def test_add_with_indexing():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        indices = np.array([0, 2])
        # Addition within fancy indexing
        result = a[indices] + b[indices]  # [1, 3] + [5, 7]
        assert counter.flops == 2
        np.testing.assert_allclose(result, np.array([6, 10]))
