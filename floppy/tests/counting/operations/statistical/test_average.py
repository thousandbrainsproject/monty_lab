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


def test_average_scalar():
    counter = FlopCounter()
    with counter:
        x = np.array(5)  # scalar array
        result = np.average(x)
        assert counter.flops == 0  # scalar inputs require no computation
        np.testing.assert_allclose(result, 5.0)


def test_average_scalar_python():
    counter = FlopCounter()
    with counter:
        result = np.average(5)
        assert counter.flops == 0  # scalar inputs require no computation
        np.testing.assert_allclose(result, 5.0)


def test_average_1d():
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        result = np.average(x)
        assert counter.flops == 6  # 5 additions + 1 division
        np.testing.assert_allclose(result, 3.0)


def test_average_2d():
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.average(x)
        assert counter.flops == 7  # 6 additions + 1 division
        np.testing.assert_allclose(result, 3.5)


def test_average_single():
    counter = FlopCounter()
    with counter:
        x = np.array([1])
        result = np.average(x)
        assert counter.flops == 2  # 1 addition + 1 division
        np.testing.assert_allclose(result, 1.0)


def test_average_weighted_1d():
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        result = np.average(x, weights=weights)
        assert counter.flops == 11  # 5 multiplications + 5 additions + 1 division
        np.testing.assert_allclose(result, 3.2)


def test_average_weighted_2d():
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        weights = np.array([0.5, 0.5])
        result = np.average(x, weights=weights, axis=0)  # average along first axis
        assert counter.flops == 16  # weighted sum (2 * 6) + division (1)
        np.testing.assert_allclose(result, np.array([2.5, 3.5, 4.5]))


def test_average_axis():
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.average(x, axis=0)  # average of each column
        assert counter.flops == 7  # 6 additions + 1 division
        np.testing.assert_allclose(result, np.array([2.5, 3.5, 4.5]))


def test_average_broadcast():
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        result = np.average(x + y)  # broadcast y to x's shape then average
        assert counter.flops == 13  # 6 FLOPs for addition + (6 additions + 1 division)
        np.testing.assert_allclose(result, 5.5)


def test_average_weighted_broadcast():
    counter = FlopCounter()
    with counter:
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 3])
        weights = np.ones_like(x)
        result = np.average(x + y, weights=weights)
        assert (
            counter.flops == 19
        )  # 6 FLOPs for addition + (6 multiplications + 6 additions + 1 division)
        np.testing.assert_allclose(result, 5.5)


def test_average_multi_axis():
    counter = FlopCounter()
    with counter:
        x = np.ones((2, 3, 4))
        result = np.average(x, axis=(0, 2))
        assert counter.flops == 25  # 24 additions + 1 division
        np.testing.assert_allclose(result, np.ones(3))


def test_masked_average():
    counter = FlopCounter()
    with counter:
        x = np.array([1, 2, 3, 4, 5])
        mask = np.array([True, False, True, False, True])
        masked_x = np.ma.array(x, mask=mask)
        result = np.ma.average(masked_x)
        assert counter.flops == 6
        np.testing.assert_allclose(result, 3.0)  # average of [2, 4]
