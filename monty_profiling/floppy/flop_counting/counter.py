from contextlib import ContextDecorator
from typing import Dict, Any
import numpy as np
from .operations import *
import inspect

def should_skip_flop_counting():
    stack_frames = inspect.stack()
    for frame in stack_frames:
        # Skip counting if inside a library call (e.g., NumPy internals)
        if (
            "site-packages" in frame.filename
            or "numpy" in frame.filename
            or "torch" in frame.filename
            or "habitat_sim" in frame.filename
        ):
            return True
    return False


class TrackedArray(np.ndarray):
    """Array wrapper that tracks floating point operations using the operation registry."""

    def __new__(cls, input_array, counter):
        # Unwrap if the input_array is already a TrackedArray
        if isinstance(input_array, TrackedArray):
            input_array = input_array.view(np.ndarray)

        # Create the TrackedArray
        obj = np.asarray(input_array).view(cls)
        obj.counter = counter
        return obj

    def __array_finalize__(self, obj):
        """Ensure proper initialization of TrackedArray attributes during array creation.

        This method is called by NumPy whenever a new array is created from an existing one,
        including through views, slices, or other array operations. It ensures that the FLOP
        counter reference is properly inherited by the new array.

        Args:
            obj: Optional[ndarray]
                The array from which this array was created. None if the array is being
                created from scratch rather than derived from an existing array.

        Note:
            This method is essential for maintaining the FLOP counting functionality across
            all array operations, as it ensures derived arrays maintain their connection
            to the original FlopCounter instance.

            The method handles three main cases:
            1. Direct array creation (obj is None)
            2. View creation (obj is TrackedArray)
            3. Result of array operations (obj is TrackedArray)
        """
        if obj is None:
            return
        self.counter = getattr(obj, "counter", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercept NumPy ufuncs (like add, multiply, etc.) and count FLOPs.
        """

        # 1) Unwrap inputs into base NumPy arrays
        clean_inputs = []
        for inp in inputs:
            while isinstance(inp, TrackedArray):
                inp = inp.view(np.ndarray)
            clean_inputs.append(inp)

        # 2) Handle 'out' parameter separately
        if "out" in kwargs:
            out = kwargs["out"]
            if isinstance(out, tuple):
                # Create a new tuple with unwrapped elements
                clean_out = tuple(
                    o.view(np.ndarray) if isinstance(o, TrackedArray) else o
                    for o in out
                )
                kwargs["out"] = clean_out
            elif isinstance(out, TrackedArray):
                # Unwrap single 'out' parameter if it's a TrackedArray
                kwargs["out"] = out.view(np.ndarray)

        # 3) Perform the actual ufunc operation
        result = getattr(ufunc, method)(*clean_inputs, **kwargs)

        # 4) Count FLOPs if active
        if self.counter and self.counter._is_active:
            if should_skip_flop_counting():
                return result
            op_name = ufunc.__name__
            if op_name in self.counter._ufunc_operations:
                flops = self.counter._ufunc_operations[op_name].count_flops(
                    *clean_inputs, result=result
                )
                self.counter.add_flops(flops)

        # 5) Wrap the result back into a TrackedArray, if applicable
        if "out" in kwargs:
            # If 'out' was provided, ensure it is properly wrapped
            out = kwargs["out"]
            if isinstance(out, tuple):
                # Create a new tuple with wrapped elements
                wrapped_out = tuple(
                    TrackedArray(o, self.counter)
                    if isinstance(o, np.ndarray) and not isinstance(o, TrackedArray)
                    else o
                    for o in out
                )
                if len(wrapped_out) == 1:
                    return wrapped_out[0]
                return wrapped_out  # Return the wrapped output tuple
            elif isinstance(out, np.ndarray) and not isinstance(out, TrackedArray):
                return TrackedArray(out, self.counter)

        # If the result is a scalar or already wrapped, return as is
        if isinstance(result, np.ndarray) and not isinstance(result, TrackedArray):
            return TrackedArray(result, self.counter)

        return result

    """
    Patch direct calls, e.g. arr.sum(), @, etc.
    """
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return (
            type(self)(result, self.counter)
            if isinstance(result, np.ndarray)
            else result
        )

    def __repr__(self):
        return f"TrackedArray(array={super().__repr__()}, counter={id(self.counter)})"

class TrackedArray(np.ndarray):
    """Array wrapper that tracks floating point operations using the operation registry."""

    def __new__(cls, input_array, counter):
        # Unwrap if the input_array is already a TrackedArray
        if isinstance(input_array, TrackedArray):
            input_array = input_array.view(np.ndarray)

        # Create the TrackedArray
        obj = np.asarray(input_array).view(cls)
        obj.counter = counter
        return obj

    def __array_finalize__(self, obj):
        # Ensure counter and _wrapped_array are properly maintained when creating new arrays
        if obj is None:
            return
        self.counter = getattr(obj, "counter", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercept NumPy ufuncs (like add, multiply, etc.) and count FLOPs.
        """

        # 1) Unwrap inputs into base NumPy arrays
        clean_inputs = []
        for inp in inputs:
            while isinstance(inp, TrackedArray):
                inp = inp.view(np.ndarray)
            clean_inputs.append(inp)

        # 2) Handle 'out' parameter separately
        if "out" in kwargs:
            out = kwargs["out"]
            if isinstance(out, tuple):
                # Create a new tuple with unwrapped elements
                clean_out = tuple(
                    o.view(np.ndarray) if isinstance(o, TrackedArray) else o
                    for o in out
                )
                kwargs["out"] = clean_out
            elif isinstance(out, TrackedArray):
                # Unwrap single 'out' parameter if it's a TrackedArray
                kwargs["out"] = out.view(np.ndarray)

        # 3) Perform the actual ufunc operation
        result = getattr(ufunc, method)(*clean_inputs, **kwargs)

        # 4) Count FLOPs if active
        if self.counter and self.counter._is_active:
            if should_skip_flop_counting():
                return result
            op_name = ufunc.__name__
            if op_name in self.counter._ufunc_operations:
                flops = self.counter._ufunc_operations[op_name].count_flops(
                    *clean_inputs, result=result
                )
                self.counter.add_flops(flops)

        # 5) Wrap the result back into a TrackedArray, if applicable
        if "out" in kwargs:
            # If 'out' was provided, ensure it is properly wrapped
            out = kwargs["out"]
            if isinstance(out, tuple):
                # Create a new tuple with wrapped elements
                wrapped_out = tuple(
                    TrackedArray(o, self.counter)
                    if isinstance(o, np.ndarray) and not isinstance(o, TrackedArray)
                    else o
                    for o in out
                )
                if len(wrapped_out) == 1:
                    return wrapped_out[0]
                return wrapped_out  # Return the wrapped output tuple
            elif isinstance(out, np.ndarray) and not isinstance(out, TrackedArray):
                return TrackedArray(out, self.counter)

        # If the result is a scalar or already wrapped, return as is
        if isinstance(result, np.ndarray) and not isinstance(result, TrackedArray):
            return TrackedArray(result, self.counter)

        return result

    """
    Patch direct calls, e.g. arr.sum(), @, etc.
    """
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
    """
    Context manager that:
    1) monkey-patches np.array => returns TrackedArray
    2) optionally monkey-patches a few high-level calls (np.sum, np.matmul, etc.)
    3) accumulates FLOPs for each operation
    """

    def __init__(self):
        self.flops = 0
        self._is_active = False
        self.skip_library_calls = True

        self._original_array_func = None
        self._original_funcs = {}

        self._ufunc_operations = {
            "add": Addition(),
            "subtract": Subtraction(),
            "multiply": Multiplication(),
            "divide": Division(),
            "power": PowerOperation(),
            "floor_divide": FloorDivideOperation(),
            "mod": ModuloOperation(),
            "bitwise_and": BitwiseAndOperation(),
            "bitwise_or": BitwiseOrOperation(),
            "sin": SineOperation(),
            "cos": CosineOperation(),
            "tan": TangentOperation(),
            "arctan": ArcTangentOperation(),
            "arcsin": ArcSineOperation(),
            "arccos": ArccosOperation(),
            "cross": CrossOperation(),
            "clip": ClipOperation(),
            "where": WhereOperation(),
            "min": MinOperation(),
            "max": MaxOperation(),
            "round": RoundOperation(),
            "isnan": IsnanOperation(),
            "log": LogOperation(),
        }

        self._function_operations = {
            "matmul": MatmulOperation(),
            "sum": SumOperation(),
            "dot": MatmulOperation(),
            "linalg.norm": NormOperation(),
            "linalg.cond": CondOperation(),
            "linalg.inv": InvOperation(),
            "linalg.eig": EigOperation(),
            "mean": MeanOperation(),
            "std": StdOperation(),
            "var": VarOperation(),
            "average": AverageOperation(),
            "trace": TraceOperation(),
            "argmin": ArgminOperation(),
            "argmax": ArgmaxOperation(),
        }
        self._patch_targets = {
            "matmul": (np, "matmul"),
            "sum": (np, "sum"),
            "dot": (np, "dot"),
            # Functions in np.linalg
            "linalg.norm": (np.linalg, "norm"),
            "linalg.cond": (np.linalg, "cond"),
            "linalg.inv": (np.linalg, "inv"),
            "linalg.eig": (np.linalg, "eig"),
            # Statistics / reductions in np
            "mean": (np, "mean"),
            "std": (np, "std"),
            "var": (np, "var"),
            "average": (np, "average"),
            "trace": (np, "trace"),
            "argmin": (np, "argmin"),
            "argmax": (np, "argmax"),
        }
        self._patch_targets = {
            "matmul": (np, "matmul"),
            "sum": (np, "sum"),
            "dot": (np, "dot"),
            # Functions in np.linalg
            "linalg.norm": (np.linalg, "norm"),
            "linalg.cond": (np.linalg, "cond"),
            "linalg.inv": (np.linalg, "inv"),
            "linalg.eig": (np.linalg, "eig"),
            # Statistics / reductions in np
            "mean": (np, "mean"),
            "std": (np, "std"),
            "var": (np, "var"),
            "average": (np, "average"),
            "trace": (np, "trace"),
            "argmin": (np, "argmin"),
            "argmax": (np, "argmax"),
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

        # Monkey-patch the functions in _patch_targets
        for name, (mod, attr) in self._patch_targets.items():
            original_func = getattr(mod, attr)
            self._original_funcs[name] = original_func

            def make_wrapper(func_name, func):
                """
                Create a closure that intercepts the high-level call,
                counts FLOPs, then calls the original.
                """

                def wrapper(*args, **kwargs):
                    # Preprocess arguments to ensure they are valid for `func`
                    clean_args = []
                    for arg in args:
                        if isinstance(arg, TrackedArray):
                            # Unwrap nested TrackedArray to its base array
                            while isinstance(arg, TrackedArray):
                                arg = arg.view(np.ndarray)
                        clean_args.append(arg)

                    # Clean kwargs similarly
                    clean_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, TrackedArray):
                            # Unwrap TrackedArray in kwargs
                            clean_kwargs[k] = v.view(np.ndarray)
                        else:
                            clean_kwargs[k] = v

                    # Call the original function with cleaned arguments
                    result = func(*clean_args, **clean_kwargs)

                    # If recognized, count FLOPs via self._function_operations
                    if func_name in self._function_operations and self._is_active:
                        flops = self._function_operations[func_name].count_flops(
                            *clean_args, result=result
                        )
                        if flops is not None:
                            self.add_flops(flops)

                    return result

                return wrapper

            wrapped_func = make_wrapper(name, original_func)
            setattr(mod, attr, wrapped_func)

        # Enable the flop counter after patching is complete
        self._is_active = True
        return self

    def __exit__(self, *exc):
        """
        Deactivate the FLOP counter, restore original functionality.
        """
        self._is_active = False

        # Restore original array function
        np.array = self._original_array_func

        # Restore original monkey-patched functions
        for name, original_func in self._original_funcs.items():
            mod, attr = self._patch_targets[name]
            setattr(mod, attr, original_func)

        return False

    def add_flops(self, count: int):
        """Add to the FLOP count only if counter is active."""
        if not self._is_active:
            return

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