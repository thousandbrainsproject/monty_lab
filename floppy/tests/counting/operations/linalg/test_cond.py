import numpy as np

from floppy.counting.counter import FlopCounter


def test_condition_number_2x2():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2], [3, 4]])
        _ = np.linalg.cond(a)
        # For 2x2 matrix: 14*(2^3) + 1 = 113 FLOPs
        assert counter.flops == 113


def test_condition_number_3x3():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        _ = np.linalg.cond(a)
        # For 3x3 matrix: 14*(3^3) + 1 = 379 FLOPs
        assert counter.flops == 379


def test_condition_number_4x4():
    counter = FlopCounter(test_mode=True)
    with counter:
        a = np.eye(4)  # 4x4 identity matrix
        _ = np.linalg.cond(a)
        # For 4x4 matrix: 14*(4^3) + 1 = 897 FLOPs
        assert counter.flops == 897
