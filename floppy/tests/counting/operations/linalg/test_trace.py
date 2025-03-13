import numpy as np
import pytest

from floppy.counting.counter import FlopCounter


def test_trace_basic():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.trace(a)
        assert counter.flops == 1


def test_trace_method():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = a.trace()
        assert counter.flops == 1


def test_trace_rectangular():
    counter = FlopCounter()
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        _ = np.trace(a)
        assert counter.flops == 1


def test_trace_empty():
    counter = FlopCounter()
    with counter:
        a = np.array([[]])  # or np.zeros((0,0))
        _ = np.trace(a)
        assert counter.flops == 0


def test_trace_3d():
    counter = FlopCounter()
    with counter:
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        _ = np.trace(a)
        assert counter.flops == 2
