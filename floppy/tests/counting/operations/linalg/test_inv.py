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


def test_inv_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 0], [0, 1]])
        result = np.linalg.inv(a)
        assert counter.flops == 13  # Basic 2x2 matrix inversion
        np.testing.assert_allclose(result, np.array([[1, 0], [0, 1]]))


def test_inv_3x3():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 0, 2], [0, 1, 0], [2, 0, 1]])
        result = np.linalg.inv(a)
        assert counter.flops == 36  # 3x3 matrix inversion
        np.testing.assert_allclose(
            result,
            np.array(
                [
                    [-0.33333333, 0.0, 0.66666667],
                    [0.0, 1.0, 0.0],
                    [0.66666667, 0.0, -0.33333333],
                ]
            ),
        )


def test_inv_identity():
    counter = FlopCounter()
    with counter:
        a = np.eye(3)  # 3x3 identity matrix
        result = np.linalg.inv(a)
        assert counter.flops == 36  # Same as regular 3x3
        np.testing.assert_allclose(result, np.eye(3))




def test_inv_1x1():
    counter = FlopCounter()
    with counter:
        a = np.array([[4.0]])
        result = np.linalg.inv(a)
        assert counter.flops == 1  # Simple reciprocal for 1x1
        np.testing.assert_allclose(result, np.array([[0.25]]))
