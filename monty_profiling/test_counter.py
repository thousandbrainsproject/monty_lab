import numpy as np
from contextlib import contextmanager


class TrackedArray:
    def __init__(self, array, counter):
        self.array = array
        self.counter = counter

    def __array__(self):
        return self.array

    # Basic arithmetic operations
    def __add__(self, other):
        result = self.array + np.asarray(other)
        self.counter.flops += result.size
        return TrackedArray(result, self.counter)

    def __sub__(self, other):
        result = self.array - np.asarray(other)
        self.counter.flops += result.size
        return TrackedArray(result, self.counter)

    def __mul__(self, other):
        result = self.array * np.asarray(other)
        self.counter.flops += result.size
        return TrackedArray(result, self.counter)

    def __truediv__(self, other):
        result = self.array / np.asarray(other)
        self.counter.flops += result.size
        return TrackedArray(result, self.counter)

    # Reverse operations
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        result = np.asarray(other) - self.array
        self.counter.flops += result.size
        return TrackedArray(result, self.counter)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        result = np.asarray(other) / self.array
        self.counter.flops += result.size
        return TrackedArray(result, self.counter)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def __getitem__(self, key):
        result = self.array[key]
        if isinstance(result, np.ndarray):
            return TrackedArray(result, self.counter)
        return result

    def __len__(self):
        return len(self.array)


class FlopCounter:
    def __init__(self):
        self.flops = 0
        self._old_array = None
        self._old_array_func = None

    def __enter__(self):
        # Store the original array class and array function
        self._old_array = np.ndarray
        self._old_array_func = np.array

        # Override numpy array creation to return our tracked arrays
        np.ndarray = lambda *args, **kwargs: TrackedArray(
            self._old_array(*args, **kwargs), self
        )

        # Override np.array function
        def tracked_array(*args, **kwargs):
            return TrackedArray(self._old_array_func(*args, **kwargs), self)

        np.array = tracked_array

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original array class and function
        np.ndarray = self._old_array
        np.array = self._old_array_func


def test_basic_operations():
    counter = FlopCounter()

    with counter:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        c = a + b  # Should count 3 FLOPs
        d = a * b  # Should count 3 more FLOPs

    assert counter.flops == 6


def test_mixed_operations():
    counter = FlopCounter()

    with counter:
        a = np.array([1, 2, 3])
        b = 2  # scalar

        c = a * b  # Should count 3 FLOPs
        d = b + a  # Should count 3 more FLOPs

    assert counter.flops == 6


if __name__ == "__main__":
    test_basic_operations()
    test_mixed_operations()
