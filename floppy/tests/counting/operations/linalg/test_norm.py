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


def test_norm_basic() -> None:
    """Test Frobenius norm behavior and flop count for 2x2 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.norm(a)  # Frobenius norm by default
        assert counter.flops == 27  # noqa: PLR2004
        np.testing.assert_allclose(result, 5.477225575051661)


def test_norm_1d() -> None:
    """Test L2 norm behavior and flop count for 1D array."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.linalg.norm(a)  # L2 norm for vector
        assert counter.flops == 26  # noqa: PLR2004
        np.testing.assert_allclose(result, 3.7416573867739413)


def test_norm_rectangular() -> None:
    """Test Frobenius norm behavior and flop count for 2x3 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.linalg.norm(a)
        assert counter.flops == 31  # noqa: PLR2004
        np.testing.assert_allclose(result, 9.539392014169456)


def test_norm_3d() -> None:
    """Test Frobenius norm behavior and flop count for 2x2x2 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = np.linalg.norm(a)
        assert counter.flops == 36  # noqa: PLR2004
        np.testing.assert_allclose(result, 14.2828568570857)


def test_norm_empty() -> None:
    """Test Frobenius norm behavior and flop count for empty matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[]])
        result = np.linalg.norm(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, 0)


def test_norm_l1() -> None:
    """Test L1 norm behavior and flop count for 1D array."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.linalg.norm(a, ord=1)  # L1 norm
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, 6)


def test_norm_l2() -> None:
    """Test L2 norm behavior and flop count for 1D array."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.linalg.norm(a, ord=2)  # L2 norm
        assert counter.flops == 26  # noqa: PLR2004
        np.testing.assert_allclose(result, 3.7416573867739413)


def test_norm_max() -> None:
    """Test max norm behavior and flop count for 1D array."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.linalg.norm(a, ord=np.inf)  # Max norm
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, 3)


def test_matrix_norm_l1() -> None:
    """Test maximum column sum norm behavior and flop count for 2x2 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.norm(a, ord=1)  # Maximum column sum
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, 6.0)


def test_matrix_norm_inf() -> None:
    """Test maximum row sum norm behavior and flop count for 2x2 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.norm(a, ord=np.inf)  # Maximum row sum
        assert counter.flops == 5  # noqa: PLR2004
        np.testing.assert_allclose(result, 7.0)


def test_norm_nuclear() -> None:
    """Test nuclear norm behavior and flop count for 2x2 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.norm(a, ord="nuc")  # Nuclear norm
        assert counter.flops == 114  # noqa: PLR2004
        np.testing.assert_allclose(result, 5.8309518948453)


def test_norm_spectral() -> None:
    """Test spectral norm behavior and flop count for 2x2 matrix."""
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.linalg.norm(a, ord=2)  # Spectral norm
        assert counter.flops == 112  # noqa: PLR2004
        np.testing.assert_allclose(result, 5.464985704219043)
