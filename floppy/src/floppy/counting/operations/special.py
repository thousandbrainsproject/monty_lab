from typing import Any

import numpy as np

__all__ = [
    "ClipOperation",
    "WhereOperation",
    "RoundOperation",
    "IsnanOperation",
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
            result: Not used but kept for consistency with other operations
            **kwargs: Additional keyword arguments that match numpy.clip parameters.
                     These currently don't affect the FLOP count.
                     Example kwargs: out, where, casting, order, etc.

        Returns:
            int: Number of comparison FLOPs
        """
        array, min_val, max_val = args[:3]
        comparisons_per_element = 1 if (min_val is None or max_val is None) else 2
        return comparisons_per_element * np.size(array)


class WhereOperation:
    """FLOP count for where operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for where operation.

        Each element requires 1 comparison operation.

        Args:
            *args: Input arrays (condition, x, y)
            result: Not used but kept for consistency
            **kwargs: Additional keyword arguments that match numpy.where parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of comparison operations
        """
        return np.size(args[0])


class RoundOperation:
    """FLOP count for round operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for round operation.

        Each element requires 1 comparison operation.

        Args:
            *args: Input arrays (first argument is the array to round)
            result: Not used but kept for consistency
            **kwargs: Additional keyword arguments that match numpy.round parameters.
                     These currently don't affect the FLOP count.
                     Example kwargs: decimals, out, etc.

        Returns:
            int: Number of comparison operations
        """
        return np.size(args[0])


class IsnanOperation:
    """FLOP count for isnan operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for isnan operation.

        Each element requires 1 comparison operation.

        Args:
            *args: Input arrays (first argument is the array to check)
            result: Not used but kept for consistency
            **kwargs: Additional keyword arguments that match numpy.isnan parameters.
                     These currently don't affect the FLOP count.
                     Example kwargs: out, where, etc.

        Returns:
            int: Number of comparison operations
        """
        return np.size(args[0])
