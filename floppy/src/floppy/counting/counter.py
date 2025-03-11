import inspect
import threading
import time
from contextlib import ContextDecorator
from pathlib import Path
from typing import List, Optional

import numpy as np

from .logger import LogManager, Operation
from .registry import OperationRegistry


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
        if obj is None:
            return
        self.counter = getattr(obj, "counter", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Intercept NumPy ufuncs and count FLOPs."""
        # Map of ufunc reductions to their corresponding function operations
        reduction_map = {
            "add": "sum",
            "minimum": "min",
            "maximum": "max",
        }

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
                clean_out = tuple(
                    o.view(np.ndarray) if isinstance(o, TrackedArray) else o
                    for o in out
                )
                kwargs["out"] = clean_out
            elif isinstance(out, TrackedArray):
                kwargs["out"] = out.view(np.ndarray)

        # 3) Perform the actual ufunc operation
        result = getattr(ufunc, method)(*clean_inputs, **kwargs)

        # 4) Count FLOPs if active
        if self.counter and not self.counter.should_skip_counting():
            op_name = ufunc.__name__
            # Check if this is a reduction operation
            if method == "reduce" and op_name in reduction_map:
                func_op_name = reduction_map[op_name]
                operation = self.counter.registry.get_operation(func_op_name)
                if operation:
                    flops = operation.count_flops(
                        *clean_inputs, result=result, **kwargs
                    )
                    self.counter.add_flops(flops)
            else:
                operation = self.counter.registry.get_operation(op_name)
                if operation:
                    flops = operation.count_flops(*clean_inputs, result=result)
                    self.counter.add_flops(flops)

        # 5) Wrap the result back into a TrackedArray, if applicable
        if "out" in kwargs:
            out = kwargs["out"]
            if isinstance(out, tuple):
                wrapped_out = tuple(
                    TrackedArray(o, self.counter)
                    if isinstance(o, np.ndarray) and not isinstance(o, TrackedArray)
                    else o
                    for o in out
                )
                if len(wrapped_out) == 1:
                    return wrapped_out[0]
                return wrapped_out
            elif isinstance(out, np.ndarray) and not isinstance(out, TrackedArray):
                return TrackedArray(out, self.counter)

        if isinstance(result, np.ndarray) and not isinstance(result, TrackedArray):
            return TrackedArray(result, self.counter)

        return result

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return (
            type(self)(result, self.counter)
            if isinstance(result, np.ndarray)
            else result
        )

    def __repr__(self):
        return f"TrackedArray(array={super().__repr__()}, counter={id(self.counter)})"

    def __getattribute__(self, name):
        """Intercept ALL attribute access, including built-in methods."""
        # Get the counter without triggering __getattribute__ again
        counter = super().__getattribute__("counter")

        # First check if it's a tracked method using the registry
        if counter is not None:
            try:
                ufunc_name = counter.registry.get_ufunc_name(name)
                if ufunc_name:

                    def wrapped_method(*args, **kwargs):
                        # Unwrap TrackedArray arguments
                        clean_args = []
                        for arg in args:
                            if isinstance(arg, TrackedArray):
                                arg = arg.view(np.ndarray)
                            clean_args.append(arg)

                        # Get the base array
                        base_array = self.view(np.ndarray)

                        # Call the original numpy function with base_array as first argument
                        result = getattr(np, ufunc_name)(
                            base_array, *clean_args, **kwargs
                        )

                        # Wrap result if it's an array (don't count FLOPs here since __array_ufunc__ will handle it)
                        if isinstance(result, np.ndarray) and not isinstance(
                            result, TrackedArray
                        ):
                            return TrackedArray(result, counter)
                        return result

                    return wrapped_method
            except:
                pass  # If any error occurs during registry lookup, fall back to normal attribute access

        # For non-tracked attributes, use normal attribute access
        return super().__getattribute__(name)


class FlopCounter(ContextDecorator):
    """Context manager that tracks FLOP operations."""

    def __init__(
        self,
        test_mode=False,
        log_manager: Optional[LogManager] = None,
        skip_paths: Optional[List[str]] = None,
        include_paths: Optional[List[str]] = None,
    ):
        self.flops = 0
        self._is_active = False
        self.log_manager = log_manager
        self.test_mode = test_mode
        self.skip_paths = skip_paths if skip_paths is not None else []
        self.include_paths = include_paths if include_paths is not None else []
        self._original_array_func = None
        self._original_funcs = {}
        self._original_ufuncs = {}
        self._flops_lock = threading.Lock()
        self._operation_stack = []  # Track nested operations

        # Initialize the operation registry
        self.registry = OperationRegistry.create_default_registry()

        # Create patch targets from registry using module locations
        self.patch_targets = {
            name: self.registry.get_module_location(name)
            for name in self.registry.get_all_operations().keys()
        }

    def _tracked_array(self, *args, **kwargs):
        """Create a tracked array from numpy array creation."""
        arr = self._original_array_func(*args, **kwargs)
        return TrackedArray(arr, self)

    def _make_wrapper(self, func_name, func):
        """Create a closure that intercepts the high-level call and counts FLOPs."""
        def wrapper(*args, **kwargs):
            # Push this operation onto the stack
            self._operation_stack.append(func_name)

            try:
                # Preprocess arguments
                clean_args = []
                for arg in args:
                    if isinstance(arg, TrackedArray):
                        while isinstance(arg, TrackedArray):
                            arg = arg.view(np.ndarray)
                        clean_args.append(arg)
                    else:
                        clean_args.append(arg)

                # Clean kwargs
                clean_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, TrackedArray):
                        clean_kwargs[k] = v.view(np.ndarray)
                    else:
                        clean_kwargs[k] = v

                # Call the original function
                result = func(*clean_args, **clean_kwargs)

                # Only count FLOPs if this is the outermost operation
                if self._is_active and len(self._operation_stack) == 1:
                    operation = self.registry.get_operation(func_name)
                    if operation:
                        frame_locals = inspect.currentframe().f_locals
                        frame_locals["func_name"] = func_name.split(".")[-1]
                        flops = operation.count_flops(
                            *clean_args, result=result, **clean_kwargs
                        )
                        if flops is not None:
                            self.add_flops(flops)

                return result
            finally:
                # Always pop the operation from stack, even if there's an error
                self._operation_stack.pop()

        def reduce_wrapper(array, axis=0, dtype=None, out=None, **kwargs):
            # Push this operation onto the stack
            self._operation_stack.append(f"{func_name}.reduce")

            try:
                if isinstance(array, TrackedArray):
                    array = array.view(np.ndarray)
                if out is not None and isinstance(out, TrackedArray):
                    out = out.view(np.ndarray)

                # Use the stored original ufunc for reduction
                original_ufunc = self._original_ufuncs.get(func_name)
                if original_ufunc is None:
                    original_ufunc = getattr(np, func_name)
                    self._original_ufuncs[func_name] = original_ufunc

                result = original_ufunc.reduce(array, axis, dtype, out, **kwargs)

                # Only count FLOPs if this is the outermost operation
                if self._is_active and len(self._operation_stack) == 1:
                    operation = self.registry.get_operation(func_name)
                    if operation:
                        size = array.size if axis is None else array.shape[axis]
                        if size > 0:
                            flops = operation.count_flops(array, result=result)
                            if flops is not None:
                                self.add_flops(flops * (size - 1))

                return result
            finally:
                # Always pop the operation from stack
                self._operation_stack.pop()

        wrapper.reduce = reduce_wrapper
        return wrapper

    def __enter__(self):
        # Store original numpy array class and array functions
        self._original_array = np.ndarray
        self._original_array_func = np.array

        # Store original ufuncs before patching
        for name in self.patch_targets:
            if hasattr(np, name):
                self._original_ufuncs[name] = getattr(np, name)

        # Override numpy array creation to return tracked arrays
        np.array = self._tracked_array

        # Monkey-patch the functions in _patch_targets
        for name, (mod, attr) in self.patch_targets.items():
            original_func = getattr(mod, attr)
            self._original_funcs[name] = original_func
            wrapped_func = self._make_wrapper(name, original_func)
            setattr(mod, attr, wrapped_func)

        # Enable the flop counter after patching is complete
        self._is_active = True
        return self

    def __exit__(self, *exc):
        """
        Deactivate the FLOP counter, restore original functionality.
        """
        self._is_active = False

        # Restore original array functions
        np.array = self._original_array_func

        # Restore original monkey-patched functions
        for name, original_func in self._original_funcs.items():
            mod, attr = self.patch_targets[name]
            setattr(mod, attr, original_func)

        if self.log_manager:
            self.log_manager.flush()

        return False

    def should_skip_counting(self) -> bool:
        """
        Determine if FLOP counting should be skipped based on the call stack.
        """
        frame = inspect.currentframe()
        try:
            # First check if we're in a high-level operation
            in_wrapper = False
            temp_frame = frame
            while temp_frame:
                if temp_frame.f_code.co_name == "wrapper":
                    in_wrapper = True
                    break
                temp_frame = temp_frame.f_back

            # If we're in a wrapper and this is an array_ufunc call, skip it
            if in_wrapper:
                temp_frame = frame
                while temp_frame:
                    if temp_frame.f_code.co_name == "__array_ufunc__":
                        return True
                    temp_frame = temp_frame.f_back

            # Path-based skipping logic
            while frame:
                filename = frame.f_code.co_filename
                if any(path in filename for path in self.include_paths):
                    return False
                if any(path in filename for path in self.skip_paths):
                    return True
                frame = frame.f_back
        finally:
            del frame

        return False

    def add_flops(self, count: int):
        """Add to the FLOP count only if counter is active and not in library code."""
        if not self.should_skip_counting():
            with self._flops_lock:
                print(
                    f"Adding {count} flops from {inspect.stack()[1].function}"
                )  # Debug line
                self.flops += count
            if self.log_manager:
                self._log_operation(count)

    def _log_operation(self, count: int) -> None:
        """Log the FLOP operation with details about the calling context.

        This method traverses the call stack to find the first caller outside of the
        floppy/counting directory and logs the operation with relevant metadata including
        the file, line number, function name, and timestamp.

        Args:
            count: The number of FLOPs to log for this operation.
        """
        caller_frame = inspect.currentframe().f_back.f_back  # Skip add_flops frame
        while caller_frame:
            filename = caller_frame.f_code.co_filename
            file_path = Path(filename)

            # Check if file is within floppy/counting directory
            if not str(file_path).startswith(
                str(Path("~/tbp/monty_lab/floppy/src/floppy/counting").expanduser())
            ):
                operation = Operation(
                    flops=count,
                    filename=filename,
                    line_no=caller_frame.f_lineno,
                    function_name=caller_frame.f_code.co_name,
                    timestamp=time.time(),
                )
                self.log_manager.log_operation(operation)
                break
            caller_frame = caller_frame.f_back
