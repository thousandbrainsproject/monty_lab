from contextlib import ContextDecorator
from typing import Dict, Any
import numpy as np
from .wrappers import FunctionWrapper, UfuncWrapper
from .operations import *

class TrackedArray:
    """Array wrapper that tracks floating point operations using the operation registry."""

    def __init__(self, array: np.ndarray, counter: "FlopCounter"):
        """Initialize TrackedArray with NumPy array and FlopCounter.

        Args:
            array (np.ndarray): NumPy array to wrap.
            counter (FlopCounter): FlopCounter instance for tracking operations.
        """
        self.array = array
        self.counter = counter

    def __getattr__(self, name):
        """Handle any NumPy array methods not explicitly defined.

        This allows TrackedArray to support all numpy.ndarray methods by forwarding
        them to the underlying array and wrapping the result if needed.
        """
        # Get the attribute from the underlying numpy array
        array_attr = getattr(self.array, name)

        # If it's a method, wrap it to handle the return value
        if callable(array_attr):

            def wrapped(*args, **kwargs):
                result = array_attr(*args, **kwargs)
                # If result is a numpy array, wrap it in TrackedArray
                if isinstance(result, np.ndarray):
                    return TrackedArray(result, self.counter)
                return result

            return wrapped

        # If it's not a method, return it directly
        return array_attr

    def __array__(self) -> np.ndarray:
        """REturn the underlying NumPy array."""
        return self.array

    def __array_function__(self, func, types, args, kwargs):
        """Handle NumPy function calls by delegating to the operation registry.

        This method handles both registered operations (with FLOP counting) and
        unregistered operations while maintaining proper array wrapping.
        """
        # First handle registered operations with FLOP counting
        if func.__name__ in self.counter._operations:
            clean_args = [
                arg.array if isinstance(arg, TrackedArray) else arg for arg in args
            ]
            # Execute the operation
            result = func(*clean_args, **kwargs)
            # Count FLOPs using the registered operation
            operation = self.counter._operations[func.__name__]
            flops = operation.count_flops(*clean_args, result=result)
            if flops is not None:
                self.counter.add_flops(flops)
            # Return result wrapped in TrackedArray if needed
            if isinstance(result, np.ndarray):
                return TrackedArray(result, self.counter)
            return result

        # For unregistered operations, execute them with unwrapped arrays
        clean_args = [
            arg.array if isinstance(arg, TrackedArray) else arg for arg in args
        ]
        clean_kwargs = {
            k: v.array if isinstance(v, TrackedArray) else v for k, v in kwargs.items()
        }

        result = func(*clean_args, **clean_kwargs)

        # Wrap numpy array results in TrackedArray
        if isinstance(result, np.ndarray):
            return TrackedArray(result, self.counter)
        elif isinstance(result, tuple):
            # Handle functions that return multiple arrays (like histogram)
            return tuple(
                TrackedArray(x, self.counter) if isinstance(x, np.ndarray) else x
                for x in result
            )
        return result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle NumPy ufunc calls (low-level universal functions).

        This provides support for operations like np.add.reduce, np.multiply.outer, etc.
        For registered operations, it counts FLOPs. For unregistered operations, it
        executes them while maintaining the TrackedArray wrapper.
        """
        # Handle registered operations with FLOP counting
        if ufunc.__name__ in self.counter._operations:
            clean_inputs = [
                inp.array if isinstance(inp, TrackedArray) else inp for inp in inputs
            ]
            # Execute the ufunc
            result = getattr(ufunc, method)(*clean_inputs, **kwargs)
            # Count FLOPs using registered operation
            operation = self.counter._operations[ufunc.__name__]
            flops = operation.count_flops(*clean_inputs, result=result)
            if flops is not None:
                self.counter.add_flops(flops)
            return (
                TrackedArray(result, self.counter)
                if isinstance(result, np.ndarray)
                else result
            )

        # Handle unregistered operations by executing them with unwrapped arrays
        clean_inputs = [
            inp.array if isinstance(inp, TrackedArray) else inp for inp in inputs
        ]
        clean_kwargs = {
            k: v.array if isinstance(v, TrackedArray) else v for k, v in kwargs.items()
        }

        result = getattr(ufunc, method)(*clean_inputs, **clean_kwargs)

        # Wrap the result appropriately
        if isinstance(result, np.ndarray):
            return TrackedArray(result, self.counter)
        elif isinstance(result, tuple):
            return tuple(
                TrackedArray(x, self.counter) if isinstance(x, np.ndarray) else x
                for x in result
            )
        return result

    def _handle_basic_op(self, other: Any, op_name: str, reverse: bool = False):
        """Handle arithmetic operations (+, -, *, /, @) between arrays."""
        if op_name not in self.counter._operations:
            return NotImplemented

        # Handle scalar operations properly
        other_value = other.array if isinstance(other, TrackedArray) else other
        args = (other_value, self.array) if reverse else (self.array, other_value)

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

    # Array attributes and methods
    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size

    @property
    def dtype(self):
        return self.array.dtype

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        result = self.array[key]
        return (
            TrackedArray(result, self.counter)
            if isinstance(result, np.ndarray)
            else result
        )

    def __repr__(self):
        return f"TrackedArray(array={self.array}, counter={id(self.counter)})"


class FlopCounter(ContextDecorator):
    """Count FLOPs in NumPy operations automatically."""

    def __init__(self):
        self.flops = 0
        self._original_array = None
        self._original_array_func = None
        self._is_active = False

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