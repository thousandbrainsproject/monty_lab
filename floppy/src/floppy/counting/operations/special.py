from typing import Any

import numpy as np

__all__ = [
    "ClipOperation",
    "WhereOperation",
    "RoundOperation",
    "IsnanOperation",
    "DiffOperation",
]


class ClipOperation:
    """FLOP count for clip operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for clipping.

        Each element requires:
        - 2 comparisons (for min and max bounds) when both bounds are provided
        - 1 comparison when only one bound is provided (other is None)

        Args:
            *args: Should contain (array, min, max)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.clip parameters.
                     These currently don't affect the FLOP count.
                     Example kwargs: out, where, casting, order, etc.

        Returns:
            int: Number of comparison FLOPs
        """
        array, min_val, max_val = args[:3]
        comparisons_per_element = 1 if (min_val is None or max_val is None) else 2
        return comparisons_per_element * np.size(result)


class WhereOperation:
    """FLOP count for where operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for where operation.

        Each element requires 1 comparison operation. The size is based on the result
        array to account for potential broadcasting between x and y arrays.

        Args:
            *args: Input arrays (condition, x, y)
            result: The result array after broadcasting
            **kwargs: Additional keyword arguments that match numpy.where parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of comparison operations
        """
        return np.size(result)


class RoundOperation:
    """FLOP count for round operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for round operation.

        Each element requires 1 comparison operation.

        Args:
            *args: Input arrays (first argument is the array to round)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.round parameters.
                     These currently don't affect the FLOP count.
                     Example kwargs: decimals, out, etc.

        Returns:
            int: Number of comparison operations
        """
        return np.size(result)


class IsnanOperation:
    """FLOP count for isnan operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for isnan operation.

        Each element requires 1 comparison operation.

        Args:
            *args: Input arrays (first argument is the array to check)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.isnan parameters.
                     These currently don't affect the FLOP count.
                     Example kwargs: out, where, etc.

        Returns:
            int: Number of comparison operations
        """
        return np.size(result)


class DiffOperation:
    """FLOP count for diff operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for diff operation.

        Each element in the output requires one subtraction between adjacent elements
        in the input array.

        Args:
            *args: Input array and optional n (number of times to diff)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.diff parameters.
                     These currently don't affect the FLOP count.
                     Example kwargs: axis, prepend, append

        Returns:
            int: Number of subtraction operations
        """
        return np.size(result)
