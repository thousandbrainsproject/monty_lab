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


def test_linspace_basic():
    counter = FlopCounter()
    with counter:
        result = np.linspace(0, 1, 5)
        assert counter.flops == 6
        np.testing.assert_allclose(result, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_linspace_single_point():
    counter = FlopCounter()
    with counter:
        result = np.linspace(0, 1, 1)
        assert counter.flops == 2
        np.testing.assert_allclose(result, np.array([0.0]))


def test_linspace_negative_range():
    counter = FlopCounter()
    with counter:
        result = np.linspace(-1, 1, 5)
        assert counter.flops == 6
        np.testing.assert_allclose(result, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_linspace_with_endpoint():
    counter = FlopCounter()
    with counter:
        result = np.linspace(0, 1, 5, endpoint=False)
        assert counter.flops == 6
        np.testing.assert_allclose(result, np.array([0.0, 0.2, 0.4, 0.6, 0.8]))


def test_linspace_with_retstep():
    counter = FlopCounter()
    with counter:
        result, step = np.linspace(0, 1, 5, retstep=True)
        # For 5 points: 2 + (5-1) = 6 FLOPs
        assert counter.flops == 6
        np.testing.assert_allclose(result, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        assert step == 0.25


def test_linspace_with_dtype():
    counter = FlopCounter()
    with counter:
        result = np.linspace(0, 1, 5, dtype=np.float32)
        # For 5 points: 2 + (5-1) = 6 FLOPs
        assert counter.flops == 6
        np.testing.assert_allclose(
            result, np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        )


def test_linspace_large_array():
    counter = FlopCounter()
    with counter:
        result = np.linspace(0, 1, 1000)
        # For 1000 points: 2 + (1000-1) = 1001 FLOPs
        assert counter.flops == 1001
        assert len(result) == 1000
        assert result[0] == 0.0
        assert result[-1] == 1.0
        np.testing.assert_allclose(np.diff(result), 1 / 999)  # Check uniform spacing
