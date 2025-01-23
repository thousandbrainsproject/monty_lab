import numpy as np
from floppy.counting.counter import FlopCounter
import pytest


class TestAdd:
    def setup_method(self):
        self.original_array = np.array

    def teardown_method(self):
        np.array = self.original_array

    def test_add_operator_syntax(self):
        counter = FlopCounter(test_mode=True)

        with counter:
            a = np.array([1, 2, 3])
            b = np.array([4, 5, 6])
            result = a + b
            assert counter.flops == 3
            np.testing.assert_array_equal(result, np.array([5, 7, 9]))

    def test_add_ufunc_syntax(self):
        counter = FlopCounter(test_mode=True)
        with counter:
            a = np.array([1, 2, 3])
            b = np.array([4, 5, 6])
            result = np.add(a, b)
            assert counter.flops == 3
            np.testing.assert_array_equal(result, np.array([5, 7, 9]))

    # FIXME
    @pytest.mark.xfail(reason="TrackedArray object has no attribute 'add'")
    def test_add_method_syntax(self):
        counter = FlopCounter(test_mode=True)
        with counter:
            a = np.array([1, 2, 3])
            b = np.array([4, 5, 6])
            result = a.add(b)
            assert counter.flops == 3
            np.testing.assert_array_equal(result, np.array([5, 7, 9]))

    def test_add_augmented_assignment(self):
        counter = FlopCounter(test_mode=True)
        with counter:
            a = np.array([1, 2, 3])
            b = np.array([4, 5, 6])
            a += b
            assert counter.flops == 3
            np.testing.assert_array_equal(a, np.array([5, 7, 9]))

    def test_add_broadcasting(self):
        counter = FlopCounter(test_mode=True)
        with counter:
            a = np.array([1, 2, 3])
            b = 2
            result = a + b
            assert counter.flops == 3
            np.testing.assert_array_equal(result, np.array([3, 4, 5]))

        counter.flops = 0
        with counter:
            a = np.array([1, 2, 3])
            b = 2
            result = b + a
            assert counter.flops == 3
            np.testing.assert_array_equal(result, np.array([3, 4, 5]))