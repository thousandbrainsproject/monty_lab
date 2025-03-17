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


def test_convolve_basic() -> None:
    """Test basic convolution operation with 'valid' mode."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        kernel = np.array([1, 2])
        result = np.convolve(a, kernel, mode="valid")
        assert counter.flops == 9
        np.testing.assert_allclose(result, np.array([4, 7, 10]))


def test_convolve_full() -> None:
    """Test convolution operation with 'full' mode."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        kernel = np.array([1, 2, 3])
        result = np.convolve(a, kernel, mode="full")
        assert counter.flops == 25
        np.testing.assert_allclose(result, np.array([1, 4, 10, 12, 9]))


def test_convolve_same() -> None:
    """Test convolution operation with 'same' mode."""
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3, 4])
        kernel = np.array([1, 2, 3])
        result = np.convolve(a, kernel, mode="same")
        assert counter.flops == 20
        np.testing.assert_allclose(result, np.array([4, 10, 16, 17]))
