import numpy as np
from typing import Any

__all__ = [
    "BitwiseAndOperation",
    "BitwiseOrOperation",
]


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
