# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .flop_counter import FlopCounter


class TrackedArray(np.ndarray):
    """A NumPy array subclass that tracks floating point operations for FLOP counting.

    TrackedArray wraps standard NumPy arrays to monitor and count floating point operations
    performed on the array. It integrates with FlopCounter to provide accurate FLOP counts
    for numerical computations. The class maintains the full functionality of np.ndarray
    while transparently tracking operations through NumPy's __array_ufunc__ protocol.

    Key Features:
        - Transparent operation tracking: All NumPy operations are monitored automatically
        - Zero overhead when not counting: Operations bypass tracking when counter is inactive
        - Preserves array behavior: Maintains all standard NumPy array functionality
        - Thread-safe tracking: Safe to use in multi-threaded environments
        - Nested operation support: Correctly handles operations within operations

    The array automatically tracks operations like:
        - Basic arithmetic (+, -, *, /, etc.)
        - Mathematical functions (sqrt, exp, log, etc.)
        - Linear algebra operations (dot, matmul, etc.)
        - Reductions (sum, mean, etc.)
        - Universal functions (ufuncs)

    Note:
        - Arrays are only tracked when used within a FlopCounter context
        - Slicing and viewing operations maintain tracking capabilities
        - Non-floating point operations are not counted
        - Tracking has minimal performance impact when counter is inactive
    """

    def __new__(
        cls, input_array: np.ndarray, counter: Optional["FlopCounter"]
    ) -> "TrackedArray":
        """Create a new TrackedArray instance.

        This method creates a new TrackedArray that wraps a NumPy array and tracks its floating point operations
        using a FlopCounter. If the input array is already a TrackedArray, it is unwrapped to its base NumPy array
        to prevent infinite recursion when creating nested TrackedArrays.

        Args:
            input_array: The input array to wrap. Can be a regular NumPy array or a TrackedArray.
            counter: The FlopCounter instance that will track floating point operations performed on this array.
                    If None, operations will not be tracked.

        Returns:
            A new TrackedArray instance wrapping the input array.

        Example:
            >>> from floppy.counting.base import FlopCounter
            >>> with FlopCounter() as counter:
            ...     a = TrackedArray(np.array([1, 2, 3]), counter)
            ...     b = a + 1  # This operation will be tracked
        """
        # Unwrap if the input_array is already a TrackedArray
        if isinstance(input_array, TrackedArray):
            input_array = input_array.view(np.ndarray)

        # Create the TrackedArray
        obj = np.asarray(input_array).view(cls)
        obj.counter = counter
        return obj

    def __array_finalize__(self, obj: Optional["TrackedArray"]) -> None:
        """Initialize attributes when a new TrackedArray is created through array operations.

        This method is called whenever a new TrackedArray is created through array operations
        like slicing, reshaping, or other NumPy operations. It ensures that the counter attribute
        is properly inherited from the parent array.

        Args:
            obj: The parent array from which this array was created. Can be None if the array
                was created directly through __new__.

        Returns:
            None

        Note:
            This is a NumPy internal method that is automatically called during array creation
            and manipulation. It should not be called directly by users.
        """
        if obj is None:
            return
        self.counter = getattr(obj, "counter", None)

    def __getitem__(
        self, key: Union[int, slice, tuple, np.ndarray]
    ) -> Union["TrackedArray", Any]:
        """Get an item or slice from the array while preserving FLOP tracking capabilities.

        This method overrides NumPy's __getitem__ to ensure that any array returned from
        indexing or slicing operations remains a TrackedArray with the same FLOP counter.
        This is crucial for maintaining FLOP counting across array operations that involve
        accessing or slicing data.

        Args:
            key: The index, slice, or mask to access array elements. Can be:
                - int: Single index (e.g., arr[0])
                - slice: Range of indices (e.g., arr[1:10])
                - tuple: Multiple indices for multi-dimensional arrays (e.g., arr[1:2, 0])
                - ndarray: Boolean or integer mask (e.g., arr[arr > 0])

        Returns:
            TrackedArray if the result is an array, otherwise the raw element value.
            This ensures that subsequent operations on array slices continue to be tracked.

        Example:
            >>> with FlopCounter() as counter:
            ...     arr = TrackedArray(np.array([[1, 2], [3, 4]]), counter)
            ...     slice = arr[0]  # Returns TrackedArray([1, 2])
            ...     element = arr[0, 0]  # Returns 1 (raw value)
        """
        result = super().__getitem__(key)
        return (
            type(self)(result, self.counter)
            if isinstance(result, np.ndarray)
            else result
        )

    def _unwrap_array(self, arr: Union[np.ndarray, Any]) -> np.ndarray:
        """Unwrap a TrackedArray to its base NumPy array.

        Args:
            arr: Array to unwrap, can be TrackedArray or any other type

        Returns:
            Unwrapped NumPy array if input was TrackedArray, otherwise original input
        """
        while isinstance(arr, TrackedArray):
            arr = arr.view(np.ndarray)
        return arr

    def _unwrap_output_param(
        self, out: Union[tuple, np.ndarray, Any]
    ) -> Union[tuple, np.ndarray, Any]:
        """Unwrap the 'out' parameter which can be a tuple of arrays or single array.

        Args:
            out: Output parameter to unwrap

        Returns:
            Unwrapped output parameter with all TrackedArrays converted to base NumPy arrays
        """
        if isinstance(out, tuple):
            return tuple(self._unwrap_array(o) for o in out)
        return self._unwrap_array(out)

    def _wrap_result(
        self, result: Union[np.ndarray, tuple, Any]
    ) -> Union["TrackedArray", tuple, Any]:
        """Wrap NumPy array results back into TrackedArrays.

        Args:
            result: Result to wrap, can be array, tuple of arrays, or other type

        Returns:
            Result with all NumPy arrays wrapped in TrackedArray
        """
        if isinstance(result, tuple):
            wrapped = tuple(
                TrackedArray(o, self.counter)
                if isinstance(o, np.ndarray) and not isinstance(o, TrackedArray)
                else o
                for o in result
            )
            return wrapped[0] if len(wrapped) == 1 else wrapped
        if isinstance(result, np.ndarray) and not isinstance(result, TrackedArray):
            return TrackedArray(result, self.counter)
        return result

    def _count_flops(
        self, ufunc: np.ufunc, method: str, inputs: tuple, result: Any, **kwargs: Any
    ) -> None:
        """Count FLOPs for the operation if tracking is active.

        Args:
            ufunc: The NumPy universal function that was applied
            method: The method of the ufunc that was called
            inputs: The input arrays to the operation
            result: The result of the operation
            **kwargs: Additional keyword arguments passed to the operation
        """
        if not (self.counter and not self.counter._should_exclude_operation()):
            return

        op_name = ufunc.__name__
        operation = self.counter.registry.get_operation(op_name)
        if operation:
            flops = operation.count_flops(*inputs, result=result, **kwargs)
            if flops is not None:
                self.counter.add_flops(flops)

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Union[None, "TrackedArray", np.ndarray, tuple]:
        """Intercept NumPy universal functions (ufuncs) to track floating point operations.

        This special method is called by NumPy when a ufunc (universal function) is applied to this array.
        It enables FLOP counting by:
        1. Unwrapping TrackedArrays to their base NumPy arrays before the operation
        2. Performing the requested operation using the original NumPy ufunc
        3. Counting the floating point operations (FLOPs) via the counter's registry
        4. Re-wrapping the results back into TrackedArrays to maintain tracking

        Args:
            ufunc: The NumPy universal function being applied (e.g., np.add, np.multiply)
            method: The ufunc method being called. Can be:
                   - '__call__': Standard element-wise operation (e.g., a + b)
                   - 'reduce': Reduction operation (e.g., np.add.reduce -> np.sum)
                   - 'accumulate': Cumulative operation (e.g., np.add.accumulate)
                   - 'outer': Outer product operation (e.g., np.multiply.outer)
            *inputs: Variable length argument list of input arrays. These can be:
                    - TrackedArrays (will be unwrapped)
                    - Regular NumPy arrays
                    - Python scalars
            **kwargs: Keyword arguments for the ufunc. Special handling for:
                     - 'out': Pre-allocated output array(s) for results

        Notes:
            - All array inputs and outputs maintain their FLOP tracking capabilities
            - The 'out' parameter is specially handled to preserve FLOP tracking
            - FLOPs are only counted when the counter is active and not in skip mode
            - Operations are tracked through the counter's registry based on the ufunc name
        """
        clean_inputs = [self._unwrap_array(inp) for inp in inputs]
        if "out" in kwargs:
            kwargs["out"] = self._unwrap_output_param(kwargs["out"])

        # Perform the operation
        result = getattr(ufunc, method)(*clean_inputs, **kwargs)

        # Count the FLOPs
        self._count_flops(ufunc, method, clean_inputs, result, **kwargs)

        if "out" in kwargs:
            return self._wrap_result(kwargs["out"])
        return self._wrap_result(result)

    def __getattribute__(self, name: str) -> Any:
        """Intercept ALL attribute access to handle NumPy ufuncs and maintain FLOP tracking.

        This method overrides Python's default attribute lookup to intercept all attribute
        access, including built-in methods. This is crucial for FLOP tracking because:
        1. We need to intercept NumPy's ufunc methods before they're called
        2. Using __getattr__ would miss built-in attributes and only catch missing ones
        3. We need to wrap method returns in TrackedArray to maintain tracking

        The method specifically:
        - Safely retrieves the counter without recursive attribute lookup
        - Checks if the attribute is a tracked ufunc method
        - Wraps the method to ensure results maintain FLOP tracking
        - Falls back to normal attribute access for non-tracked attributes

        Args:
            name: The name of the attribute being accessed

        Returns:
            Any: The attribute value, possibly wrapped in a tracking method if it's
                 a NumPy ufunc method that needs FLOP counting

        Example:
            >>> import numpy as np
            >>> from floppy.counting.base import FlopCounter
            >>> with FlopCounter() as counter:
            ...     arr = TrackedArray(np.array([1, 2, 3]), counter)
            ...     # Direct ufunc access is tracked
            ...     result = arr.sum()  # Tracked operation
            ...     # Normal attributes work as expected
            ...     shape = arr.shape  # Regular attribute access

        Note:
            This uses super().__getattribute__ for the counter to avoid infinite
            recursion, as accessing self.counter would trigger __getattribute__ again.
            Wrapped methods are cached to improve performance on repeated access.
        """
        # Get the counter without triggering __getattribute__ again
        counter = super().__getattribute__("counter")

        # Check for cached wrapped method
        try:
            cache_dict = super().__getattribute__("_wrapped_methods_cache")
        except AttributeError:
            # Initialize cache if it doesn't exist
            cache_dict = {}
            super().__setattr__("_wrapped_methods_cache", cache_dict)

        # First check if it's a tracked method using the registry
        if counter is not None:
            # Check cache first
            if name in cache_dict:
                return cache_dict[name]

            try:
                ufunc_name = counter.registry.get_ufunc_name(name)
                if ufunc_name:

                    def wrapped_method(
                        *args: Any, **kwargs: Any
                    ) -> Union[TrackedArray, Any]:
                        # Unwrap TrackedArray arguments
                        clean_args = []
                        for arg in args:
                            if isinstance(arg, TrackedArray):
                                arg = arg.view(np.ndarray)
                            clean_args.append(arg)

                        # Get the base array
                        base_array = self.view(np.ndarray)

                        try:
                            # Call the original numpy function with base_array as first argument
                            result = getattr(np, ufunc_name)(
                                base_array, *clean_args, **kwargs
                            )

                            # Wrap result if it's an array
                            if isinstance(result, np.ndarray) and not isinstance(
                                result, TrackedArray
                            ):
                                return TrackedArray(result, counter)
                            return result
                        except AttributeError as e:
                            raise AttributeError(
                                f"NumPy function '{ufunc_name}' not found"
                            ) from e
                        except Exception as e:
                            raise RuntimeError(
                                f"Error in wrapped method '{name}': {e!s}"
                            ) from e

                    # Cache the wrapped method
                    cache_dict[name] = wrapped_method
                    return wrapped_method
            except AttributeError:
                # Registry lookup failed, fall back to normal attribute access
                pass
            except Exception as e:
                # Log unexpected errors but don't break attribute access
                import warnings

                warnings.warn(
                    f"Unexpected error in __getattribute__ for {name}: {e!s}"
                )

        # For non-tracked attributes, use normal attribute access
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        """Create a string representation of the TrackedArray.

        This method provides a detailed string representation of the TrackedArray that includes:
        1. The underlying NumPy array data (using NumPy's __repr__)
        2. The unique identifier of the associated FlopCounter (using id())

        The counter's id is included instead of the counter object itself to:
        - Avoid recursive repr calls
        - Provide a way to identify which arrays share the same counter
        - Keep the representation concise while still being informative

        Returns:
            str: A string in the format "TrackedArray(array=<array_repr>, counter=<counter_id>)"
                where <array_repr> is the NumPy array representation and <counter_id> is
                the memory address of the counter object.

        Example:
            >>> arr = TrackedArray(np.array([1, 2, 3]), counter)
            >>> repr(arr)
            'TrackedArray(array=array([1, 2, 3]), counter=140712834927872)'
        """
        return f"TrackedArray(array={super().__repr__()}, counter={id(self.counter)})"
