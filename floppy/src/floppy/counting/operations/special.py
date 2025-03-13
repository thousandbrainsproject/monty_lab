from typing import Any

import numpy as np

__all__ = [
    "DiffOperation",
]


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
