# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Optional

import numpy as np

from ..base.protocols import FlopOperation

__all__ = [
    "SumOperation",
]


class SumOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for array sum operations.

    Handles various sum computation cases including:
    - Full array sum: (M,N) -> scalar
    - Axis-specific sum: (M,N) -> (M,) or (N,)
    - Multi-axis sum: (M,N,P) -> (M,) or (N,) or (P,)
    - Batched sums: (*,M,N) -> (*,)
    - Masked sums: (M,N) -> scalar (masked values ignored)
    - NaN sums: (M,N) -> scalar (NaN values ignored)

    This implementation provides an upper bound on the FLOP count by counting
    the total number of additions needed for a full array sum. This upper bound
    is always greater than or equal to the actual FLOP count for any axis-specific
    or partial sum operation, since:
    - A full array sum requires more additions than any axis-specific sum
    - The actual FLOP count for axis-specific sums depends on the reduction order
    - Different reduction orders can lead to different FLOP counts
    """

    # Constants for sum operation
    ADDS_PER_ELEMENT = 1  # Number of additions per element in sum
    EMPTY_ARRAY_COST = 0  # Cost for empty arrays
    SINGLE_ELEMENT_COST = 0  # Cost for single element arrays

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing array sums.

        This implementation provides an upper bound on the FLOP count by counting
        the total number of additions needed for a full array sum. This upper bound
        is always greater than or equal to the actual FLOP count for any axis-specific
        or partial sum operation.

        Args:
            *args: Input arrays where the first array contains the values to sum.
                  For array methods (e.g., arr.sum()), the array may be in kwargs["self"].
            result: The resulting array from the sum operation.
                   Used to determine the number of operations computed.
            **kwargs: Additional keyword arguments:
                     - axis: Optional[Union[int, Tuple[int, ...]]] - Axis along which to operate
                     - keepdims: bool - Whether to keep reduced dimensions
                     - dtype: numpy.dtype - Output data type
                     - self: np.ndarray - For array method calls (e.g., arr.sum())
                     These currently don't affect the FLOP count as we use an upper bound.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Sum computation upper bound:
            - Each sum requires (n-1) additions where n is the total number of elements
            - This provides an upper bound for all sum operations since:
                * Full array sum requires more additions than any axis-specific sum
                * The actual FLOP count for axis-specific sums depends on reduction order
                * Different reduction orders can lead to different FLOP counts
            - For masked/nan sums, only count actual additions (comparisons not counted)
            - Special cases:
                * Empty arrays: 0 FLOPs
                * Single element: 0 FLOPs
                * Full array sum: (total_size - 1) FLOPs
        """
        # Handle both np.sum(arr) and arr.sum() cases
        array = args[0] if args else kwargs.get("self")
        if array is None:
            return None

        # Handle empty arrays
        if np.size(array) == 0:
            return self.EMPTY_ARRAY_COST

        # Handle single element arrays
        if np.size(array) == 1:
            return self.SINGLE_ELEMENT_COST

        # Count total number of additions needed (upper bound for all sum operations)
        return (np.size(array) - 1) * self.ADDS_PER_ELEMENT
