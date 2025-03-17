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


def test_trace_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.trace(a)
        assert counter.flops == 1
        np.testing.assert_allclose(result, 5)


def test_trace_method():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = a.trace()
        assert counter.flops == 1
        np.testing.assert_allclose(result, 5)


def test_trace_rectangular():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.trace(a)
        assert counter.flops == 1
        np.testing.assert_allclose(result, 6)


def test_trace_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([[]])  # or np.zeros((0,0))
        result = np.trace(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 0)


def test_trace_3d():
    counter = FlopCounter()
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = np.trace(a)
        assert counter.flops == 2
        np.testing.assert_allclose(result, np.array([8, 10]))
