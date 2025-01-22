import numpy as np
from typing import Any

__all__ = [
    "SumOperation",
    "MinOperation",
    "MaxOperation",
    "ArgminOperation",
    "ArgmaxOperation",
]


class SumOperation:
    """FLOP count for sum operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for sum operation.

        This implementation provides an upper bound by counting (n-1) additions
        where n is the total size of the array. We ignore the specific axis of
        reduction since the actual FLOP count would be less than or equal to
        this upper bound when summing along specific axes.

        Args:
            *args: Input arrays (first argument is the array to sum)
            result: Result of the operation

        Returns:
            int: Number of floating point operations (additions)
        """
        return np.size(args[0]) - 1


class MinOperation:
    """FLOP count for min operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for min operation.

        Finding the minimum requires comparing each pair of elements sequentially.
        For n elements, we need (n-1) comparisons total.
        """
        return np.size(args[0]) - 1


class MaxOperation:
    """FLOP count for max operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for max operation.

        Finding the maximum requires comparing each pair of elements sequentially.
        For n elements, we need (n-1) comparisons total.
        """
        return np.size(args[0]) - 1


class ArgminOperation:
    """FLOP count for argmin operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for argmin operation.

        Finding the minimum index requires comparing each element with the current minimum,
        similar to min operation.
        """
        return np.size(args[0]) - 1


class ArgmaxOperation:
    """FLOP count for argmax operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for argmax operation.

        Finding the maximum index requires comparing each element with the current maximum,
        similar to max operation.
        """
        return np.size(args[0]) - 1
