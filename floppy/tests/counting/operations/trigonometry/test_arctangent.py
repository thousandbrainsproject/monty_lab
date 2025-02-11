"""Run using python tests/test_arctangent.py. Do not use pytest."""

import numpy as np

from floppy.counting.counter import FlopCounter


def test_arctan_ufunc_syntax():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        result = np.arctan(a)
        assert counter.flops == 60
        np.testing.assert_allclose(
            result, np.array([0.78539816, 1.10714872, 1.24904577])
        )


def test_arctan_broadcasting():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = 2
        result = np.arctan(a)
        assert counter.flops == 20
        np.testing.assert_allclose(result, np.array([1.10714872]))

    counter.flops = 0
    with counter:
        a = np.array([[1, 2], [3, 4]])
        result = np.arctan(a)
        assert counter.flops == 80
        np.testing.assert_allclose(
            result, np.array([[0.78539816, 1.10714872], [1.24904577, 1.32581766]])
        )

@pytest.mark.skip(reason="Not implemented")
def test_arctan_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([])
        result = np.arctan(a)
        assert counter.flops == 0
        np.testing.assert_allclose(result, np.array([]))


