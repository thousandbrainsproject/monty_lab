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
