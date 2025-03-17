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


def test_condition_number_2x2():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.cond(a)
        assert counter.flops == 113
        np.testing.assert_allclose(result, 14.933034373659265)


def test_condition_number_3x3():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        result = np.linalg.cond(a)
        assert counter.flops == 379
        np.testing.assert_allclose(result, 88.4482799206987)


def test_condition_number_4x4():
    counter = FlopCounter()
    with counter:
        a = np.eye(4)  # 4x4 identity matrix
        result = np.linalg.cond(a)
        assert counter.flops == 897
        np.testing.assert_allclose(result, 1)
