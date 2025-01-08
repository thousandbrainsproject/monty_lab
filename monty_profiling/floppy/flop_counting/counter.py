from contextlib import ContextDecorator
from typing import Dict, Any
import numpy as np
from .wrappers import FunctionWrapper, UfuncWrapper
from .operations import *
import inspect
import os

class TrackedArray(np.ndarray):
    """Array wrapper that tracks floating point operations using the operation registry."""

    def __new__(cls, input_array, counter):
        obj = np.asarray(input_array).view(cls)
        obj.counter = counter
        obj._wrapped_array = np.asarray(input_array)
        return obj

    def __array_finalize__(self, obj):
        # Ensure counter and _wrapped_array are properly maintained when creating new arrays
        if obj is None:
            return
        self.counter = getattr(obj, "counter", None)
        # Copy the _wrapped_array attribute, or use the array itself if not present
        self._wrapped_array = getattr(obj, "_wrapped_array", np.asarray(self))

    def __array_function__(self, func, types, args, kwargs):
        """
        Intercept high-level NumPy functions called on TrackedArray (via __array_function__).
        """
        # If the call involves non-TrackedArray types, just defer to NumPy
        if not all(issubclass(t, TrackedArray) for t in types):
            return NotImplemented

        # Known stack functions that re-call themselves
        STACK_FUNCTIONS = {
            "vstack",
            "hstack",
            "dstack",
            "stack",
            "column_stack",
            "row_stack",
            "concatenate",
        }
        if func.__name__ in STACK_FUNCTIONS:
            # Just unwrap everything and call the "original" np function directly
            def unwrap_sequence(arg):
                if isinstance(arg, (list, tuple)):
                    return type(arg)(unwrap_sequence(x) for x in arg)
                if isinstance(arg, TrackedArray):
                    return arg._wrapped_array
                return arg

            unwrapped_args = tuple(unwrap_sequence(a) for a in args)
            unwrapped_kwargs = {k: unwrap_sequence(v) for k, v in kwargs.items()}

            raw_result = getattr(np, func.__name__)(*unwrapped_args, **unwrapped_kwargs)
            # Wrap it in a TrackedArray only once
            return type(self)(raw_result, self.counter)

        # Get the original NumPy function
        numpy_func = func.__get__(None, np.ndarray)

        def safe_unwrap(arg):
            if isinstance(arg, TrackedArray):
                return arg._wrapped_array
            elif isinstance(arg, (list, tuple)):
                return type(arg)(safe_unwrap(x) for x in arg)
            return arg

        # Carefully unwrap arguments while preserving shapes
        unwrapped_args = tuple(safe_unwrap(arg) for arg in args)
        unwrapped_kwargs = {k: safe_unwrap(v) for k, v in kwargs.items()}

        try:
            # Execute operation with unwrapped arguments
            result = numpy_func(*unwrapped_args, **unwrapped_kwargs)

            # Count FLOPs if this is a tracked operation
            if self.counter._is_active and func.__name__ in self.counter._operations:
                try:
                    operation = self.counter._operations[func.__name__]
                    flops = operation.count_flops(*unwrapped_args, result=result)
                    if flops is not None:
                        self.counter.add_flops(flops)
                except Exception:
                    pass  # Ignore errors in FLOP counting

            # Wrap the result if needed
            if isinstance(result, np.ndarray):
                return type(self)(result, self.counter)
            elif isinstance(result, tuple):
                return tuple(
                    type(self)(x, self.counter) if isinstance(x, np.ndarray) else x
                    for x in result
                )
            return result
        except Exception as e:
            # If operation fails, try with original arrays as fallback
            return func(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercept lower-level NumPy ufuncs (like add, multiply, sin, etc.).
        """
        # Get the underlying NumPy ufunc
        numpy_ufunc = getattr(np, ufunc.__name__)

        # Unwrap inputs
        clean_inputs = []
        for inp in inputs:
            if isinstance(inp, TrackedArray):
                clean_inputs.append(inp._wrapped_array)
            else:
                clean_inputs.append(inp)

        # Unwrap kwargs
        clean_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, TrackedArray):
                clean_kwargs[k] = v._wrapped_array
            else:
                clean_kwargs[k] = v

        # Perform the operation
        result = getattr(numpy_ufunc, method)(*clean_inputs, **clean_kwargs)

        # Count FLOPs if needed
        if self.counter._is_active and ufunc.__name__ in self.counter._operations:
            try:
                operation = self.counter._operations[ufunc.__name__]
                flops = operation.count_flops(*clean_inputs, result=result)
                if flops is not None:
                    self.counter.add_flops(flops)
            except Exception:
                pass

        return (
            type(self)(result, self.counter)
            if isinstance(result, np.ndarray)
            else result
        )

    def _handle_basic_op(self, other: Any, op_name: str, reverse: bool = False):
        """Handle arithmetic operations (+, -, *, /, @) between arrays."""
        if op_name not in self.counter._operations:
            return NotImplemented

        # Handle scalar operations properly
        other_value = other._wrapped_array if isinstance(other, TrackedArray) else other
        args = (
            (other_value, self._wrapped_array)
            if reverse
            else (self._wrapped_array, other_value)
        )

        # Execute operation and count FLOPs
        operation = self.counter._operations[op_name]
        try:
            result = getattr(np, op_name)(*args)
        except AttributeError:
            # Some operations like matmul might not work with scalars
            return NotImplemented

        if self.counter._is_active:
            flops = operation.count_flops(*args, result=result)
            if flops is not None:
                self.counter.add_flops(flops)

        return TrackedArray(result, self.counter)

    # Arithmetic operations
    def __add__(self, other):
        return self._handle_basic_op(other, "add")

    def __sub__(self, other):
        return self._handle_basic_op(other, "subtract")

    def __mul__(self, other):
        return self._handle_basic_op(other, "multiply")

    def __truediv__(self, other):
        return self._handle_basic_op(other, "divide")

    def __matmul__(self, other):
        return self._handle_basic_op(other, "matmul")

    # Reverse arithmetic operations
    def __radd__(self, other):
        return self._handle_basic_op(other, "add", reverse=True)

    def __rsub__(self, other):
        return self._handle_basic_op(other, "subtract", reverse=True)

    def __rmul__(self, other):
        return self._handle_basic_op(other, "multiply", reverse=True)

    def __rtruediv__(self, other):
        return self._handle_basic_op(other, "divide", reverse=True)

    def __rmatmul__(self, other):
        return self._handle_basic_op(other, "matmul", reverse=True)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return (
            type(self)(result, self.counter)
            if isinstance(result, np.ndarray)
            else result
        )

    def __repr__(self):
        return f"TrackedArray(array={super().__repr__()}, counter={id(self.counter)})"


class FlopCounter(ContextDecorator):
    """Count FLOPs in NumPy operations automatically."""

    def __init__(self):
        self.flops = 0
        self._original_array = None
        self._original_array_func = None
        self._is_active = False
        self.skip_library_calls = True
        self._operations = {
            "matmul": MatmulOperation(),
            "sum": SumOperation(),
            "dot": MatmulOperation(),
            "add": Addition(),
            "subtract": Subtraction(),
            "multiply": Multiplication(),
            "divide": Division(),
            "clip": ClipOperation(),
            "where": WhereOperation(),
            "min": MinOperation(),
            "max": MaxOperation(),
            "round": RoundOperation(),
            "isnan": IsnanOperation(),
            "sin": SineOperation(),
            "cos": CosineOperation(),
            "cross": CrossOperation(),
            "arccos": ArccosOperation(),
            "tan": TangentOperation(),
            "arctan": ArcTangentOperation(),
            "arcsin": ArcSineOperation(),
            "linalg.norm": NormOperation(),
            "linalg.cond": CondOperation(),
            "linalg.inv": InvOperation(),
            "linalg.eig": EigOperation(),
            "log": LogOperation(),
            "mean": MeanOperation(),
            "std": StdOperation(),
            "var": VarOperation(),
            "average": AverageOperation(),
            "trace": TraceOperation(),
            "argmin": ArgminOperation(),
            "argmax": ArgmaxOperation(),
            "floor_divide": FloorDivideOperation(),
            "mod": ModuloOperation(),
            "bitwise_and": BitwiseAndOperation(),
            "bitwise_or": BitwiseOrOperation(),
            "power": PowerOperation(),
        }

    def __enter__(self):
        # Store original numpy array class and array function
        self._original_array = np.ndarray
        self._original_array_func = np.array

        # Override numpy array creation to return tracked arrays
        def tracked_array(*args, **kwargs):
            arr = self._original_array_func(*args, **kwargs)
            return TrackedArray(arr, self)

        np.array = tracked_array

        # Enable FLOP counting after setup is complete
        self._is_active = True
        return self

    def __exit__(self, *exc):
        """Clean up by restoring original NumPy functionality."""
        self._is_active = False

        # Restore original array functionality
        np.array = self._original_array_func

        return False

    def add_flops(self, count: int):
        """Add to the FLOP count only if counter is active."""
        if self._is_active:
            self.flops += count

        if self.skip_library_calls:
            # Check the call stack to see if we're inside library code (site-packages, numpy, etc.)
            stack_frames = inspect.stack()
            for frame_info in stack_frames:
                frame_file = frame_info.filename
                # Adjust these conditions as needed for your environment:
                if (
                    "site-packages" in frame_file
                    or "numpy" in frame_file
                    or "scipy" in frame_file
                ):
                    # Skip counting if we're inside library code
                    return

        self.flops += count