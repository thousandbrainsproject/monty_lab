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


def test_log_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.log(a)
        assert counter.flops == 60
        np.testing.assert_allclose(result, np.array([0.0, 0.69314718, 1.09861229]))


def test_log_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        result = np.log(a)
        assert counter.flops == 20
        np.testing.assert_allclose(result, 0.69314718)

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.log(a)
        assert counter.flops == 80
        np.testing.assert_allclose(
            result, np.array([[0.0, 0.69314718], [1.09861229, 1.38629436]])
        )


def test_log_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.log(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]))
