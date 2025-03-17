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


def test_norm_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.norm(a)  # Frobenius norm by default
        assert counter.flops == 27
        np.testing.assert_allclose(result, 5.477225575051661)


def test_norm_1d():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.linalg.norm(a)  # L2 norm for vector
        assert counter.flops == 26
        np.testing.assert_allclose(result, 3.7416573867739413)


def test_norm_rectangular():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.linalg.norm(a)
        assert counter.flops == 31
        np.testing.assert_allclose(result, 9.539392014169456)


def test_norm_3d():
    counter = FlopCounter()
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = np.linalg.norm(a)
        assert counter.flops == 36
        np.testing.assert_allclose(result, 14.2828568570857)


def test_norm_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([[]])
        result = np.linalg.norm(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 0)


def test_norm_l1():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.linalg.norm(a, ord=1)  # L1 norm
        assert counter.flops == 5
        np.testing.assert_allclose(result, 6)


def test_norm_l2():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.linalg.norm(a, ord=2)  # L2 norm
        assert counter.flops == 26
        np.testing.assert_allclose(result, 3.7416573867739413)


def test_norm_max():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.linalg.norm(a, ord=np.inf)  # Max norm
        assert counter.flops == 5
        np.testing.assert_allclose(result, 3)


def test_matrix_norm_l1():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.norm(a, ord=1)  # Maximum column sum
        assert counter.flops == 5
        np.testing.assert_allclose(result, 6.0)


def test_matrix_norm_inf():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.norm(a, ord=np.inf)  # Maximum row sum
        assert counter.flops == 5
        np.testing.assert_allclose(result, 7.0)


def test_norm_nuclear():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.norm(a, ord="nuc")  # Nuclear norm
        assert counter.flops == 114
        np.testing.assert_allclose(result, 5.8309518948453)


def test_norm_spectral():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.norm(a, ord=2)  # Spectral norm
        assert counter.flops == 112
        np.testing.assert_allclose(result, 5.464985704219043)
