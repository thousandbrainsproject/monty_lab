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


def test_solve_2x2():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([5, 6])
        result = np.linalg.solve(a, b)
        assert counter.flops == 13
        np.testing.assert_allclose(
            result,
            np.array([-4, 4.5]),
        )


def test_solve_3x3():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([10, 11, 12])
        result = np.linalg.solve(a, b)
        assert counter.flops == 36
        np.testing.assert_allclose(result, np.array([-25.33333333, 41.66666667, -16.0]))


def test_solve_1x1():
    counter = FlopCounter()
    with counter:
        a = np.array([[2]])
        b = np.array([4])
        result = np.linalg.solve(a, b)
        assert counter.flops == 1
        np.testing.assert_allclose(result, np.array([2]))


def test_solve_zero():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 0], [0, 1]])
        b = np.array([0, 0])
        result = np.linalg.solve(a, b)
        assert counter.flops == 13
        np.testing.assert_allclose(result, np.array([0, 0]))


def test_solve_identity():
    counter = FlopCounter()
    with counter:
        a = np.eye(3)
        b = np.array([1, 2, 3])
        result = np.linalg.solve(a, b)
        assert counter.flops == 36
        np.testing.assert_allclose(result, np.array([1, 2, 3]))


def test_solve_multiple_rhs():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.linalg.solve(a, b)
        assert counter.flops == 21
        np.testing.assert_allclose(result, np.array([[-3.0, -4.0], [4.0, 5.0]]))


def test_solve_batched():
    counter = FlopCounter()
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = np.array([[5, 6], [7, 8]])
        result = np.linalg.solve(a, b)
        assert counter.flops == 21
        np.testing.assert_allclose(result, np.array([[-4.0, 4.5], [-4.0, 4.5]]))


def test_solve_large():
    counter = FlopCounter()
    with counter:
        n = 5
        a = np.random.rand(n, n)
        b = np.random.rand(n)
        result = np.linalg.solve(a, b)
        expected_flops = int(2 / 3 * n**3 + 2 * n**2)
        assert counter.flops == expected_flops
        residual = np.linalg.norm(np.dot(a, result) - b)
        assert residual < 1e-10
