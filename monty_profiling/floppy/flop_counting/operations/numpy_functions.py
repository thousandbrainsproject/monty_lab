import numpy as np
from typing import Any


class SumOperation:
    """FLOP count for sum operation."""
    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for sum operation."""
        return np.size(args[0]) - 1


class ClipOperation:
    """FLOP count for clip operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for clipping.

        Each element requires 2 comparisons (for min and max bounds).
        """
        return 2 * np.size(args[0])


class WhereOperation:
    """FLOP count for where operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for where operation.

        Each element requires 1 comparison operation.
        """
        return np.size(args[0])


class MinOperation:
    """FLOP count for min operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for min operation.

        Each element requires 1 comparison operation.
        """
        return np.size(args[0]) - 1


class MaxOperation:
    """FLOP count for max operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for max operation.

        Each element requires 1 comparison operation.
        """
        return np.size(args[0]) - 1


class RoundOperation:
    """FLOP count for round operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for round operation.

        Each element requires 1 comparison operation.
        """
        return np.size(args[0])


class IsnanOperation:
    """FLOP count for isnan operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for isnan operation.

        Each element requires 1 comparison operation.
        """
        return np.size(args[0])


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


class TraceOperation:
    """FLOP count for trace operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for trace operation.

        Trace is the sum of diagonal elements, requiring (n-1) additions
        where n is the number of diagonal elements.
        """
        n = min(args[0].shape)  # Get the minimum dimension (for non-square matrices)
        return n - 1


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


class LogOperation:
    """FLOP count for logarithm operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for logarithm operation.

        Each element requires 1 logarithm operation.
        """
        return np.size(args[0])

class PowerOperation:
    """FLOP count for power operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for power operation.

        Each element requires 1 power operation.
        """
        return np.size(args[0])


class FloorDivideOperation:
    """FLOP count for floor divide operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for floor divide operation.

        Each element requires 1 division operation.
        """
        return np.size(args[0])


class ModuloOperation:
    """FLOP count for modulo operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for modulo operation.

        Each element requires 1 modulo operation.
        """
        return np.size(args[0])


class BitwiseAndOperation:
    """FLOP count for bitwise and operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for bitwise and operation.

        Each element requires 1 bitwise and operation.
        """
        return np.size(args[0])


class BitwiseOrOperation:
    """FLOP count for bitwise or operation."""



    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for bitwise or operation.

        Each element requires 1 bitwise or operation.
        """
        return np.size(args[0])
