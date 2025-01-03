from floppy.flop_counting.counter import FlopCounter
import numpy as np


def test_counter():
    with FlopCounter() as flops:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = a + b
        print(flops.flops)  # got 0


if __name__ == "__main__":
    test_counter()
