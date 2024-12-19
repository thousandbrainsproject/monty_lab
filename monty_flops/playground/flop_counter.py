import numpy as np
from functools import wraps
from contextlib import ContextDecorator
import warnings
from typing import Optional, Dict, Any


class MatmulFlopCounter:
    """A helper class to handle FLOP counting for matmul operations."""

    @staticmethod
    def count_flops(args, result, flop_counter):
        """
        Given the input arguments and result of a matmul operation, determine the FLOPs.
        Warn if broadcasting is involved, and handle batch dimensions if present.
        """
        if not all(isinstance(arg, np.ndarray) for arg in args[:2]):
            return  # Skip if inputs aren't arrays

        shapes = [arg.shape for arg in args[:2]]
        if len(shapes) < 2:
            return

        # Check for broadcasting in any dimension
        if shapes[0] != shapes[1]:
            warnings.warn(
                "Broadcasting involved in matmul. FLOP count may be approximate."
            )

        try:
            result_shape = result.shape
            if len(result_shape) < 2:
                return

            M = result_shape[-2]
            P = result_shape[-1]

            # Validate matrix dimensions
            if len(shapes[0]) >= 2 and len(shapes[1]) >= 2:
                N = shapes[0][-1]
                if N != shapes[1][-2]:
                    warnings.warn(
                        f"Invalid matrix dimensions: {shapes[0]} and {shapes[1]}"
                    )
                    return

                # Compute batch count
                batch_dims = result_shape[:-2]
                batch_count = np.prod(batch_dims) if batch_dims else 1

                flop_counter.add_flops(2 * M * N * P * batch_count)
        except (AttributeError, IndexError) as e:
            warnings.warn(f"Error counting matmul FLOPs: {str(e)}")


class UfuncWrapper:
    def __init__(self, ufunc, flop_counter, operation_name):
        self.ufunc = ufunc
        self.flop_counter = flop_counter
        self.operation_name = operation_name
        self.not_supported_list = set()  # Using set for efficiency

    def __call__(self, *args, **kwargs):
        try:
            result = self.ufunc(*args, **kwargs)

            if result is None:
                return result

            size = np.size(result) if isinstance(result, np.ndarray) else 1

            if self.operation_name in {"add", "subtract", "multiply", "divide"}:
                self.flop_counter.add_flops(size)
            elif self.operation_name == "matmul":
                MatmulFlopCounter.count_flops(args, result, self.flop_counter)
            elif self.operation_name not in self.not_supported_list:
                warnings.warn(
                    f"Operation {self.operation_name} not supported for FLOP counting"
                )
                self.not_supported_list.add(self.operation_name)

            return result
        except Exception as e:
            warnings.warn(f"Error in {self.operation_name}: {str(e)}")
            raise

    def __getattr__(self, name):
        return getattr(self.ufunc, name)


class FunctionWrapper:
    def __init__(self, func, flop_counter, func_name):
        self.func = func
        self.flop_counter = flop_counter
        self.func_name = func_name

    def __call__(self, *args, **kwargs):
        try:
            result = self.func(*args, **kwargs)

            if len(args) < 2:
                return result

            # Handle different array shapes for dot product
            if self.func_name == "dot":
                a = np.asarray(args[0])
                b = np.asarray(args[1])

                # Handle vector dot product
                if (
                    a.ndim == 1
                    or (a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1))
                ) and (
                    b.ndim == 1
                    or (b.ndim == 2 and (b.shape[0] == 1 or b.shape[1] == 1))
                ):
                    n = max(a.size, b.size)
                    self.flop_counter.add_flops(2 * n - 1)
                # Handle matrix multiplication
                elif a.ndim >= 2 and b.ndim >= 2:
                    self.flop_counter.add_flops(
                        2 * a.shape[0] * a.shape[1] * b.shape[1]
                    )

            return result
        except Exception as e:
            warnings.warn(f"Error in {self.func_name}: {str(e)}")
            raise

    def __getattr__(self, name):
        return getattr(self.func, name)


class FlopCounter(ContextDecorator):
    """Count FLOPs in NumPy operations automatically."""

    def __init__(self):
        self.flops = 0
        self._original_funcs: Dict[str, Any] = {}

    def __enter__(self):
        funcs_to_wrap = [
            ("add", np.add),
            ("subtract", np.subtract),
            ("multiply", np.multiply),
            ("divide", np.divide),
            ("matmul", np.matmul),
            ("dot", np.dot),
        ]

        for func_name, func in funcs_to_wrap:
            self._wrap_function(func_name, func)

        return self

    def __exit__(self, *exc):
        for func_name, orig_func in self._original_funcs.items():
            setattr(np, func_name, orig_func)
        return False

    def _wrap_function(self, func_name: str, func: Any):
        self._original_funcs[func_name] = func
        wrapper = (
            UfuncWrapper(func, self, func_name)
            if isinstance(func, np.ufunc)
            else FunctionWrapper(func, self, func_name)
        )
        setattr(np, func_name, wrapper)

    def add_flops(self, count: int):
        self.flops += count