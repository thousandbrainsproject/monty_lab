from typing import Any, Optional

import numpy as np

__all__ = [
    "BitwiseAndOperation",
    "BitwiseOrOperation",
]


class BitwiseAndOperation:
    """Counts floating point operations (FLOPs) for element-wise bitwise AND operations.

    Each element-wise AND operation counts as one FLOP, regardless of input shapes.
    Supports standard NumPy broadcasting rules for input arrays.
    """

    # Constants for the bitwise operation
    FLOPS_PER_ELEMENT = 1  # One operation per element

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for bitwise AND operations.

        Args:
            *args: Tuple[np.ndarray, ...], Input arrays to perform bitwise AND.
                  Typically two arrays of compatible shapes.
            result: np.ndarray, The resulting array from the bitwise operation.
                   Used to determine the total number of element-wise operations.
            **kwargs: Additional numpy.bitwise_and parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Bitwise AND computation:
            - Operates element-wise on input arrays
            - Each element-wise AND requires 1 FLOP
            - Total FLOPs = number of elements in result
            - Supports broadcasting between compatible shapes
        """
        # Validate input
        if not isinstance(result, np.ndarray):
            return None

        # Count total number of element-wise operations
        return self.FLOPS_PER_ELEMENT * np.size(result)


class BitwiseOrOperation:
    """Counts floating point operations (FLOPs) for element-wise bitwise OR operations.

    Each element-wise OR operation counts as one FLOP, regardless of input shapes.
    Supports standard NumPy broadcasting rules for input arrays.
    """

    # Constants for the bitwise operation
    FLOPS_PER_ELEMENT = 1  # One operation per element

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for bitwise OR operations.

        Args:
            *args: Tuple[np.ndarray, ...], Input arrays to perform bitwise OR.
                  Typically two arrays of compatible shapes.
            result: np.ndarray, The resulting array from the bitwise operation.
                   Used to determine the total number of element-wise operations.
            **kwargs: Additional numpy.bitwise_or parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Bitwise OR computation:
            - Operates element-wise on input arrays
            - Each element-wise OR requires 1 FLOP
            - Total FLOPs = number of elements in result
            - Supports broadcasting between compatible shapes
        """
        # Validate input
        if not isinstance(result, np.ndarray):
            return None

        # Count total number of element-wise operations
        return self.FLOPS_PER_ELEMENT * np.size(result)
