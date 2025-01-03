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
