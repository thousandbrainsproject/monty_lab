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


def test_sin_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.sin(a)
        assert counter.flops == 60
        np.testing.assert_allclose(
            result, np.array([0.84147098, 0.90929743, 0.14112001])
        )


def test_sin_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        result = np.sin(a)
        assert counter.flops == 20
        np.testing.assert_allclose(result, 0.90929743)

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.sin(a)
        assert counter.flops == 80
        np.testing.assert_allclose(
            result, np.array([[0.84147098, 0.90929743], [0.14112001, -0.7568025]])
        )


def test_sin_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.sin(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]))


def test_sine_chaining():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.sin(np.sin(a))
        assert counter.flops == 120
        np.testing.assert_allclose(
            result, np.array([0.74562414, 0.78907234, 0.14065208])
        )
