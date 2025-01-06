import numpy as np
from typing import Any, Optional
from .base import BaseOperation


class ClipOperation(BaseOperation):
    """FLOP count for clip operation."""

    def __init__(self):
        super().__init__("clip")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for clipping.

        Each element requires 2 comparisons (for min and max bounds).
        """
        return 2 * np.size(args[0])


class WhereOperation(BaseOperation):
    """FLOP count for where operation."""

    def __init__(self):
        super().__init__("where")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for where operation.

        Each element requires 1 comparison operation.
        """
        return np.size(args[0])


class MinOperation(BaseOperation):
    """FLOP count for min operation."""

    def __init__(self):
        super().__init__("min")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for min operation.

        Each element requires 1 comparison operation.
        """
        return np.size(args[0]) - 1


class MaxOperation(BaseOperation):
    """FLOP count for max operation."""

    def __init__(self):
        super().__init__("max")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for max operation.

        Each element requires 1 comparison operation.
        """
        return np.size(args[0]) - 1


class RoundOperation(BaseOperation):
    """FLOP count for round operation."""

    def __init__(self):
        super().__init__("round")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for round operation.

        Each element requires 1 comparison operation.
        """
        return np.size(args[0])


class IsnanOperation(BaseOperation):
    """FLOP count for isnan operation."""

    def __init__(self):
        super().__init__("isnan")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for isnan operation.

        Each element requires 1 comparison operation.
        """
        return np.size(args[0])


class ArgminOperation(BaseOperation):
    """FLOP count for argmin operation."""

    def __init__(self):
        super().__init__("argmin")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for argmin operation.

        Finding the minimum index requires comparing each element with the current minimum,
        similar to min operation.
        """
        return np.size(args[0]) - 1


class ArgmaxOperation(BaseOperation):
    """FLOP count for argmax operation."""

    def __init__(self):
        super().__init__("argmax")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for argmax operation.

        Finding the maximum index requires comparing each element with the current maximum,
        similar to max operation.
        """
        return np.size(args[0]) - 1


class TraceOperation(BaseOperation):
    """FLOP count for trace operation."""

    def __init__(self):
        super().__init__("trace")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for trace operation.

        Trace is the sum of diagonal elements, requiring (n-1) additions
        where n is the number of diagonal elements.
        """
        n = min(args[0].shape)  # Get the minimum dimension (for non-square matrices)
        return n - 1


class MeanOperation(BaseOperation):
    """FLOP count for mean operation."""

    def __init__(self):
        super().__init__("mean")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for mean operation.

        Mean requires:
        - (n-1) additions to sum all elements
        - 1 division for the final average
        """
        return np.size(args[0])  # (n-1) additions + 1 division


class StdOperation(BaseOperation):
    """FLOP count for standard deviation operation."""

    def __init__(self):
        super().__init__("std")

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


class VarOperation(BaseOperation):
    """FLOP count for variance operation."""

    def __init__(self):
        super().__init__("var")

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


class AverageOperation(BaseOperation):
    """FLOP count for average operation."""

    def __init__(self):
        super().__init__("average")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for average operation.

        Weighted average requires:
        - n multiplications for weights
        - (n-1) additions for sum
        - 1 division by sum of weights
        Total: 2n FLOPs
        """
        return 2 * np.size(args[0])


class LogOperation(BaseOperation):
    """FLOP count for logarithm operation."""

    def __init__(self):
        super().__init__("log")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for logarithm operation.

        Each element requires 1 logarithm operation.
        """
        return np.size(args[0])

class PowerOperation(BaseOperation):
    """FLOP count for power operation."""

    def __init__(self):
        super().__init__("power")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for power operation.

        Each element requires 1 power operation.
        """
        return np.size(args[0])


class FloorDivideOperation(BaseOperation):
    """FLOP count for floor divide operation."""

    def __init__(self):
        super().__init__("floor_divide")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for floor divide operation.

        Each element requires 1 division operation.
        """
        return np.size(args[0])


class ModuloOperation(BaseOperation):
    """FLOP count for modulo operation."""

    def __init__(self):
        super().__init__("modulo")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for modulo operation.

        Each element requires 1 modulo operation.
        """
        return np.size(args[0])


class BitwiseAndOperation(BaseOperation):
    """FLOP count for bitwise and operation."""

    def __init__(self):
        super().__init__("bitwise_and")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for bitwise and operation.

        Each element requires 1 bitwise and operation.
        """
        return np.size(args[0])


class BitwiseOrOperation(BaseOperation):
    """FLOP count for bitwise or operation."""

    def __init__(self):
        super().__init__("bitwise_or")

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for bitwise or operation.

        Each element requires 1 bitwise or operation.
        """
        return np.size(args[0])
