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


def test_arctan_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.arctan(a)
        assert counter.flops == 60
        np.testing.assert_allclose(
            result, np.array([0.78539816, 1.10714872, 1.24904577])
        )


def test_arctan_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        result = np.arctan(a)
        assert counter.flops == 20
        np.testing.assert_allclose(result, np.array([1.10714872]))

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.arctan(a)
        assert counter.flops == 80
        np.testing.assert_allclose(
            result, np.array([[0.78539816, 1.10714872], [1.24904577, 1.32581766]])
        )

def test_arctan_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.arctan(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]))


