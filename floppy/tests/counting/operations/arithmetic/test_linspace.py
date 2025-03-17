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


def test_linspace_basic() -> None:
    """Test np.linspace behavior and flop count with basic array."""
    counter = FlopCounter()
    with counter:
        result = np.linspace(0, 1, 5)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_linspace_single_point() -> None:
    """Test np.linspace behavior and flop count with single point."""
    counter = FlopCounter()
    with counter:
        result = np.linspace(0, 1, 1)
        assert counter.flops == 2  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([0.0]))


def test_linspace_negative_range() -> None:
    """Test np.linspace behavior and flop count with negative range."""
    counter = FlopCounter()
    with counter:
        result = np.linspace(-1, 1, 5)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_linspace_with_endpoint() -> None:
    """Test np.linspace behavior and flop count with endpoint parameter."""
    counter = FlopCounter()
    with counter:
        result = np.linspace(0, 1, 5, endpoint=False)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([0.0, 0.2, 0.4, 0.6, 0.8]))


def test_linspace_with_retstep() -> None:
    """Test np.linspace behavior and flop count with retstep parameter."""
    counter = FlopCounter()
    with counter:
        result, step = np.linspace(0, 1, 5, retstep=True)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(result, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        assert step == 0.25


def test_linspace_with_dtype() -> None:
    """Test np.linspace behavior and flop count with dtype parameter."""
    counter = FlopCounter()
    with counter:
        result = np.linspace(0, 1, 5, dtype=np.float32)
        assert counter.flops == 6  # noqa: PLR2004
        np.testing.assert_allclose(
            result, np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        )


def test_linspace_large_array() -> None:
    """Test np.linspace behavior and flop count with large array."""
    counter = FlopCounter()
    with counter:
        result = np.linspace(0, 1, 1000)
        assert counter.flops == 1001  # noqa: PLR2004
        assert len(result) == 1000  # noqa: PLR2004
        assert result[0] == 0.0
        assert result[-1] == 1.0
        np.testing.assert_allclose(np.diff(result), 1 / 999)  # Check uniform spacing
