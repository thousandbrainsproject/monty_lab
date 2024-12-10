import numpy as np
from functools import wraps
from contextlib import ContextDecorator
import threading
from typing import Any, Dict, Optional


class FlopCounter(ContextDecorator):
    """Count FLOPs in NumPy operations automatically."""

    _thread_local = threading.local()

    def __init__(self):
        self.flops = 0
        self._original_funcs = {}

    def __enter__(self):
        # Store thread-local counter
        FlopCounter._thread_local.counter = self

        # Save original NumPy functions
        self._original_funcs = {
            "add": np.add,
            "subtract": np.subtract,
            "multiply": np.multiply,
            "divide": np.divide,
            "dot": np.dot,
            "matmul": np.matmul,
        }

        # Patch NumPy functions to count FLOPs
        def count_elementwise(func_name):
            orig_func = self._original_funcs[func_name]

            def wrapper(*args, **kwargs):
                result = orig_func(*args, **kwargs)
                # Count one FLOP per element
                if isinstance(result, np.ndarray):
                    self.flops += np.size(result)
                else:
                    self.flops += 1
                return result

            return wrapper

        def count_matmul(*args, **kwargs):
            result = self._original_funcs["matmul"](*args, **kwargs)
            # For matrix multiplication of (m,n) and (n,p) matrices
            # Number of FLOPs is approximately 2*m*n*p
            if len(args) >= 2:
                shape1 = np.array(args[0]).shape
                shape2 = np.array(args[1]).shape
                if len(shape1) == 2 and len(shape2) == 2:
                    self.flops += 2 * shape1[0] * shape1[1] * shape2[1]
            return result

        def count_dot(*args, **kwargs):
            result = self._original_funcs["dot"](*args, **kwargs)
            # For vectors, dot product is n multiplies and n-1 adds
            if len(args) >= 2:
                shape1 = np.array(args[0]).shape
                if len(shape1) == 1:
                    self.flops += 2 * shape1[0] - 1
                else:
                    # For matrices, similar to matmul
                    shape2 = np.array(args[1]).shape
                    if len(shape2) == 2:
                        self.flops += 2 * shape1[0] * shape1[1] * shape2[1]
            return result

        # Replace NumPy functions with counting versions
        np.add = count_elementwise("add")
        np.subtract = count_elementwise("subtract")
        np.multiply = count_elementwise("multiply")
        np.divide = count_elementwise("divide")
        np.matmul = count_matmul
        np.dot = count_dot

        return self

    def __exit__(self, *exc):
        # Restore original NumPy functions
        for func_name, orig_func in self._original_funcs.items():
            setattr(np, func_name, orig_func)
        return False

    def add_flops(self, count: int):
        """Manually add FLOPs"""
        self.flops += count


def count_flops(func):
    """Decorator to count FLOPs in a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with FlopCounter() as counter:
            result = func(*args, **kwargs)
            wrapper.flops = counter.flops  # Store flop count in the wrapper
            print(f"FLOPs in {func.__name__}: {counter.flops:,}")
        return result

    return wrapper


# Example usage:
if __name__ == "__main__":
    # Example 1: Using as a decorator
    @count_flops
    def matrix_operations(size):
        # Create some matrices
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)

        # Do some operations
        c = np.dot(a, b)
        d = np.add(c, a)
        return d

    # Example 2: Using as a context manager
    with FlopCounter() as counter:
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        c = np.dot(a, b)
        print(f"FLOPs in context: {counter.flops:,}")

    # Run example function
    result = matrix_operations(100)
