# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np


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
