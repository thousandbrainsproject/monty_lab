# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import inspect
import threading
import time
from contextlib import ContextDecorator
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..logger import FlopLogEntry, LogManager
from ..registry import OperationRegistry
from .tracked_array import TrackedArray


class FlopCounter(ContextDecorator):
    """Context manager that tracks floating point operations (FLOPs) in numerical computations.

    The FlopCounter tracks FLOPs by monkey-patching NumPy functions and monitoring array
    operations. It can be used to measure computational complexity and performance
    characteristics of numerical computations. As a context manager and decorator, it can
    be used either with a 'with' statement or as a function/method decorator.

    Key Features:
        - Accurate FLOP counting for NumPy operations
        - Thread-safe operation tracking
        - Configurable path filtering for selective monitoring
        - Optional detailed logging of operations
        - Support for nested operations without double-counting
        - Can be used as both context manager and decorator
    """

    def __init__(
        self,
        log_manager: Optional[LogManager] = None,
        skip_paths: Optional[List[str]] = None,
        include_paths: Optional[List[str]] = None,
    ):
        """Initialize a FlopCounter instance.

        Args:
            log_manager: Optional LogManager instance to record FLOP operations with
                metadata like file, line number, and timestamp. If None, operations
                will be counted but not logged.
            skip_paths: Optional list of path substrings. Any code from files containing
                these substrings in their paths will be excluded from FLOP counting.
                This is useful for excluding library code or test files.
            include_paths: Optional list of path substrings that override skip_paths.
                Even if a path matches a skip_path pattern, if it also matches an
                include_path pattern, FLOPs will still be counted. This allows fine-grained
                control over which code paths are monitored.

        Note:
            - The counter is not active until used as a context manager with 'with'
            - Thread-safe FLOP counting is ensured using locks
            - Nested operations are tracked to avoid double-counting
            - NumPy operations are temporarily monkey-patched while the counter is active
        """
        self.flops = 0
        self._is_active = False
        self.log_manager = log_manager
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

    def _tracked_array(self, *args: Any, **kwargs: Any) -> TrackedArray:
        """Intercept NumPy array creation to return a tracked array.

        This internal method is used to monkey-patch numpy.array() during the FlopCounter
        context. It ensures that any array created using np.array() within the context
        is automatically wrapped in a TrackedArray for FLOP counting.

        The method:
        1. Calls the original np.array function to create the array
        2. Wraps the result in a TrackedArray linked to this counter
        3. Preserves all original np.array functionality and arguments

        Args:
            *args: Variable length argument list passed to np.array()
            **kwargs: Arbitrary keyword arguments passed to np.array()

        Returns:
            TrackedArray: A tracked version of the array that would have been created
                by the original np.array() call.
        """
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
        """Deactivate the FLOP counter, restore original functionality."""
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
        """Determine if FLOP counting should be skipped based on the call stack.

        This method analyzes the current call stack to determine whether FLOP counting
        should be skipped. It checks two conditions:
        1. If we're in a high-level operation wrapper and it's an array_ufunc call
        2. If the calling code is in a path that should be skipped based on skip_paths
           and include_paths configuration

        Returns:
            bool: True if FLOP counting should be skipped, False otherwise.
                 Returns True if:
                 - The call is from a wrapper's array_ufunc
                 - The call originates from a path in skip_paths (unless overridden by include_paths)
                 Returns False if:
                 - The call originates from a path in include_paths
                 - None of the above conditions are met
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

    def add_flops(self, count: int) -> None:
        """Add to the FLOP count if counter is active and not in library code.

        This method increments the total FLOP count and logs the operation if a log manager
        is configured. The count is only incremented if the counter is active and the
        calling code is not in a skipped library path.

        Args:
            count: The number of FLOPs to add to the total count.

        Returns:
            None
        """
        if not self.should_skip_counting():
            with self._flops_lock:
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
                operation = FlopLogEntry(
                    flops=count,
                    filename=filename,
                    line_no=caller_frame.f_lineno,
                    function_name=caller_frame.f_code.co_name,
                    timestamp=time.time(),
                    episode=self.episode,
                    parent_method=self.parent_method,
                    is_wrapped_method=self.is_wrapped_method,
                )
                self.log_manager.log_operation(operation)
                break
            caller_frame = caller_frame.f_back
