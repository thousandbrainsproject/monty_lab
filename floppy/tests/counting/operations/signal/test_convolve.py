import numpy as np

from floppy.counting.counter import FlopCounter


def test_convolve_basic():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3, 4])
        kernel = np.array([1, 2])
        result = np.convolve(a, kernel, mode="valid")
        # For valid mode with kernel size 2:
        # - 2 multiplications per output
        # - 1 addition per output
        # - 3 outputs
        assert counter.flops == 9  # (2 * 2 - 1) * 3
        np.testing.assert_equal(result, np.array([5, 8, 11]))


def test_convolve_full():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        kernel = np.array([1, 2, 3])
        result = np.convolve(a, kernel, mode="full")
        # For full mode with kernel size 3:
        # - 3 multiplications per output
        # - 2 additions per output
        # - 5 outputs
        assert counter.flops == 25  # (2 * 3 - 1) * 5
        np.testing.assert_equal(result, np.array([1, 4, 10, 12, 9]))


def test_convolve_same():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3, 4])
        kernel = np.array([1, 2, 3])
        result = np.convolve(a, kernel, mode="same")
        # For same mode with kernel size 3:
        # - 3 multiplications per output
        # - 2 additions per output
        # - 4 outputs
        assert counter.flops == 20  # (2 * 3 - 1) * 4
        np.testing.assert_equal(result, np.array([8, 14, 20, 17]))


def test_convolve_empty_input():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([])
        kernel = np.array([1, 2])
        result = np.convolve(a, kernel, mode="valid")
        assert counter.flops == 0
        np.testing.assert_equal(result, np.array([]))


def test_convolve_empty_kernel():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([1, 2, 3])
        kernel = np.array([])
        result = np.convolve(a, kernel, mode="valid")
        assert counter.flops == 0
        np.testing.assert_equal(result, np.array([]))


def test_convolve_2d():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6]])
        kernel = np.array([[1, 2], [3, 4]])
        result = np.convolve2d(a, kernel, mode="valid")
        # For 2D convolution with 2x2 kernel:
        # - 4 multiplications per output
        # - 3 additions per output
        # - 1 output
        assert counter.flops == 7  # (2 * 4 - 1) * 1
        np.testing.assert_equal(result, np.array([[37]]))
