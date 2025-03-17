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


def test_cos_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.cos(a)
        assert counter.flops == 60  # 20 flops * 3 elements
        np.testing.assert_allclose(
            result, np.array([0.54030231, -0.41614684, -0.9899925])
        )


def test_cos_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        result = np.cos(a)
        assert counter.flops == 20
        np.testing.assert_allclose(result, -0.41614684)

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.cos(a)
        assert counter.flops == 80
        np.testing.assert_allclose(
            result, np.array([[0.54030231, -0.41614684], [-0.9899925, -0.65364362]])
        )


def test_cos_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.cos(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]))
