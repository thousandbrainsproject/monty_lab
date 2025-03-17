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


def test_multiply_operator_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a * b
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([4, 10, 18]))


def test_multiply_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.multiply(a, b)
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([4, 10, 18]))


def test_multiply_method_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a.multiply(b)
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([4, 10, 18]))


def test_multiply_augmented_assignment():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a *= b
        assert counter.flops == 3
        np.testing.assert_allclose(a, np.array([4, 10, 18]))


def test_multiply_broadcasting():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a * b
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([2, 4, 6]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b * a
        assert counter.flops == 3
        np.testing.assert_allclose(result, np.array([2, 4, 6]))
