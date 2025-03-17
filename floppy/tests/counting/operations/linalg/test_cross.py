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


def test_cross_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        result = np.cross(a, b)
        assert counter.flops == 9
        np.testing.assert_allclose(result, np.array([0, 0, 1]))


def test_cross_multiple():
    counter = FlopCounter()
    with counter:
        # Multiple cross products at once
        a = np.array([[1, 0, 0], [2, 0, 0]])
        b = np.array([[0, 1, 0], [0, 2, 0]])
        result = np.cross(a, b)
        assert counter.flops == 18  # 9 flops * 2 cross products
        np.testing.assert_allclose(result, np.array([[0, 0, 1], [0, 0, 4]]))


def test_cross_broadcasting():
    counter = FlopCounter()
    with counter:
        # Broadcasting a single vector against multiple vectors
        a = np.array([1, 0, 0])
        b = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]])
        result = np.cross(a, b)
        assert counter.flops == 27  # 9 flops * 3 cross products
        np.testing.assert_allclose(result, np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3]]))
