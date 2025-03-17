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


def test_eig_2x2():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _, result = np.linalg.eig(a)
        assert counter.flops == 240
        np.testing.assert_allclose(
            result, np.array([[-0.82456484, -0.41597356], [0.56576746, -0.90937671]])
        )

def test_eig_3x3():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        _, result = np.linalg.eig(a)
        assert counter.flops == 810
        np.testing.assert_allclose(
            result,
            np.array(
                [
                    [-0.23197069, -0.78583024, 0.40824829],
                    [-0.52532209, -0.08675134, -0.81649658],
                    [-0.8186735, 0.61232756, 0.40824829],
                ]
            ),
        )


def test_eig_1x1():
    counter = FlopCounter()
    with counter:
        a = np.array([[5]])
        _, result = np.linalg.eig(a)
        assert counter.flops == 30
        np.testing.assert_allclose(result, np.array([[1.0]]))


def test_eig_zero():
    counter = FlopCounter()
    with counter:
        a = np.zeros((2, 2))
        _, result = np.linalg.eig(a)
        assert counter.flops == 240
        np.testing.assert_allclose(result, np.array([[1.0, 0.0], [0.0, 1.0]]))
