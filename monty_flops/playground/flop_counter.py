import numpy as np
from functools import wraps
from contextlib import ContextDecorator
import threading
import warnings


class MatmulFlopCounter:
    """A helper class to handle FLOP counting for matmul operations."""

    @staticmethod
    def count_flops(args, result, flop_counter):
        """
        Given the input arguments and result of a matmul operation, determine the FLOPs.
        Warn if broadcasting is involved, and handle batch dimensions if present.
        """
        # Extract shapes of inputs
        shapes = [np.shape(arg) for arg in args if isinstance(arg, np.ndarray)]
        # Check for broadcasting in batch dimensions
        if any(len(s) > 2 for s in shapes):
            # If we have multiple arrays with differing leading shapes:
            batch_shapes = [s[:-2] for s in shapes if len(s) > 2]
            if len(batch_shapes) > 1:
                first = batch_shapes[0]
                for bs in batch_shapes[1:]:
                    if bs != first:
                        warnings.warn(
                            "Broadcasting involved in matmul. FLOP count may be approximate."
                        )
                        break

        # Count FLOPs:
        # For a (M x N) * (N x P) => (M x P) multiplication:
        # FLOPs ≈ 2 * M * N * P (exact: M*P*(2N-1), but we use the approximation)

        result_shape = result.shape
        if len(result_shape) >= 2:
            M = result_shape[-2]
            P = result_shape[-1]

            # Determine N from inputs:
            if len(args) >= 2:
                shape1 = np.array(args[0]).shape
                shape2 = np.array(args[1]).shape
                if len(shape1) >= 2 and len(shape2) >= 2:
                    N_candidate_1 = shape1[-1]
                    N_candidate_2 = shape2[-2]
                    if N_candidate_1 == N_candidate_2:
                        N = N_candidate_1
                        # Compute batch count
                        batch_dims = result_shape[:-2]
                        batch_count = 1
                        for d in batch_dims:
                            batch_count *= d

                        flop_counter.add_flops(2 * M * N * P * batch_count)


class UfuncWrapper:
    def __init__(self, ufunc, flop_counter, operation_name):
        self.ufunc = ufunc
        self.flop_counter = flop_counter
        self.operation_name = operation_name
        self.not_supported_list = []

    def __call__(self, *args, **kwargs):
        result = self.ufunc(*args, **kwargs)

        size = np.size(result) if isinstance(result, np.ndarray) else 1
        if self.operation_name in ["add", "subtract", "multiply", "divide"]:
            self.flop_counter.add_flops(1 * size)
        elif self.operation_name == "matmul":
            MatmulFlopCounter.count_flops(args, result, self.flop_counter)
        else:
            # print only once if the operation is not supported
            if self.operation_name not in self.not_supported_list:
                print(f"Operation {self.operation_name} not supported")
                self.not_supported_list.append(self.operation_name)

        return result

    def __getattr__(self, name):
        return getattr(self.ufunc, name)


class FunctionWrapper:
    def __init__(self, func, flop_counter, func_name):
        self.func = func
        self.flop_counter = flop_counter
        self.func_name = func_name

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)

        # FLOP counting logic for non-ufunc functions
        if self.func_name == "matmul":
            # For matrix multiplication of (m,n) and (n,p):
            # FLOPs ≈ 2*m*n*p
            if len(args) >= 2:
                shape1 = np.array(args[0]).shape
                shape2 = np.array(args[1]).shape
                if len(shape1) == 2 and len(shape2) == 2:
                    self.flop_counter.add_flops(2 * shape1[0] * shape1[1] * shape2[1])
        elif self.func_name == "dot":
            # For vectors: dot product = 2n - 1 FLOPs
            # For matrices: ~2*m*n*p FLOPs
            if len(args) >= 2:
                shape1 = np.array(args[0]).shape
                shape2 = np.array(args[1]).shape
                if len(shape1) == 1 and len(shape2) == 1:
                    self.flop_counter.add_flops(2 * shape1[0] - 1)
                elif len(shape1) == 2 and len(shape2) == 2:
                    self.flop_counter.add_flops(2 * shape1[0] * shape1[1] * shape2[1])

        return result

    def __getattr__(self, name):
        return getattr(self.func, name)


class FlopCounter(ContextDecorator):
    """Count FLOPs in NumPy operations automatically."""

    _thread_local = threading.local()

    def __init__(self):
        self.flops = 0
        self._original_funcs = {}

    def __enter__(self):
        FlopCounter._thread_local.counter = self

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

    def _wrap_function(self, func_name, func):
        # Store original function
        self._original_funcs[func_name] = func

        # Wrap depending on whether it's a ufunc or a regular function
        if isinstance(func, np.ufunc):
            wrapper = UfuncWrapper(func, self, func_name)
        else:
            wrapper = FunctionWrapper(func, self, func_name)

        setattr(np, func_name, wrapper)

    def add_flops(self, count: int):
        self.flops += count
