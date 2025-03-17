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


def test_mod_operator_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a % b
        assert counter.flops == 9
        np.testing.assert_allclose(result, np.array([1, 2, 3]))


def test_mod_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.mod(a, b)
        assert counter.flops == 9
        np.testing.assert_allclose(result, np.array([1, 2, 3]))


def test_mod_augmented_assignment():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a %= b
        assert counter.flops == 9
        np.testing.assert_allclose(a, np.array([1, 2, 3]))


def test_mod_broadcasting():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = a % b
        assert counter.flops == 9
        np.testing.assert_allclose(result, np.array([1, 0, 1]))

    counter.flops = 0
    with counter:
        a = np.array([1, 2, 3])
        b = 2
        result = b % a
        assert counter.flops == 9
        np.testing.assert_allclose(result, np.array([0, 0, 2]))
