import numpy as np
import pytest

from floppy.counting.core import FlopCounter


def test_tan_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([1, 2, 3])
        result = np.tan(a)
        assert counter.flops == 60
        np.testing.assert_allclose(
            result, np.array([1.55740772, -2.18503986, -0.14254654])
        )


def test_tan_broadcasting():
    counter = FlopCounter()
    with counter:
        a = 2
        _ = np.tan(a)
        assert counter.flops == 20

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.tan(a)
        assert counter.flops == 80


def test_tan_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        _ = np.tan(a)
        assert counter.flops == 0
