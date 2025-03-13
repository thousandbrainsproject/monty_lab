"""Run using python tests/test_arccosine.py. Do not use pytest."""

import numpy as np

from floppy.counting.counter import FlopCounter


def test_arccos_ufunc_syntax():
    counter = FlopCounter()
    with counter:
        a = np.array([-0.5, 0.0, 0.5])  # values within valid domain [-1, 1]
        result = np.arccos(a)
        assert counter.flops == 132  # 44 FLOPs * 3 elements = 132
        np.testing.assert_allclose(
            result, np.array([2.0943951, 1.57079633, 1.04719755])
        )


def test_arccos_scalar():
    counter = FlopCounter()
    with counter:
        a = 0.5
        result = np.arccos(a)
        assert counter.flops == 44
        np.testing.assert_allclose(result, np.array([1.04719755]))


def test_arccos_broadcasting():
    counter = FlopCounter()
    with counter:
        a = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = np.arccos(a)
        assert counter.flops == 176  # 44 FLOPs * 4 elements = 176
        np.testing.assert_allclose(
            result, np.array([[1.47062894, 1.36943841], [1.26610367, 1.15927948]])
        )


def test_arccos_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.arccos(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]))
