import numpy as np
from typing import Any

__all__ = [
    "MeanOperation",
    "StdOperation",
    "VarOperation",
    "AverageOperation",
]


class MeanOperation:
    """FLOP count for mean operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for mean operation.

        Mean requires:
        - (n-1) additions to sum all elements
        - 1 division for the final average
        """
        return np.size(args[0])  # (n-1) additions + 1 division


class StdOperation:
    """FLOP count for standard deviation operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for standard deviation operation.

        Standard deviation requires:
        - n FLOPs for mean calculation
        - n subtractions from mean
        - n multiplications for squaring
        - (n-1) additions for sum
        - 1 division for mean of squares
        - 1 square root
        Total: 4n + 1 FLOPs
        """
        n = np.size(args[0])
        return 4 * n + 1


class VarOperation:
    """FLOP count for variance operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for variance operation.

        Variance requires:
        - n FLOPs for mean calculation
        - n subtractions from mean
        - n multiplications for squaring
        - (n-1) additions for sum
        - 1 division for final result
        Total: 4n FLOPs
        """
        n = np.size(args[0])
        return 4 * n


class AverageOperation:
    """FLOP count for average operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for average operation.

        Unweighted average:
        - n additions for sum
        - 1 division
        Total: n + 1 FLOPs

        Weighted average:
        - n multiplications for weights
        - n additions for weighted sum
        - 1 division by sum of weights
        Total: 2n + 1 FLOPs

        Args:
            *args: Input arrays
            result: Result of the operation
            **kwargs: Additional keyword arguments (e.g., weights)

        Returns:
            int: Total FLOPs
        """
        if not args:
            return 0

        array = args[0]  # Input array
        weights = kwargs.get("weights", None)

        n = np.size(array)  # Total number of elements in the input array

        if weights is not None:
            # Weighted average: weighted sum + sum of weights + division
            return 2 * n + 1  # 2n operations for weighted sum and 1 division
        else:
            # Unweighted average: sum + division
            return n + 1  # n operations for sum and 1 division
