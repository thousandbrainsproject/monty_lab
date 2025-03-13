from typing import Any, Tuple, Union

import numpy as np

__all__ = [
    "SumOperation",
    "MinOperation",
    "MaxOperation",
    "ArgminOperation",
    "ArgmaxOperation",
]


class SumOperation:
    """FLOP count for sum operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for sum operation.

        This implementation provides an upper bound by counting (n-1) additions
        where n is the total size of the array. We ignore the specific axis of
        reduction since the actual FLOP count would be less than or equal to
        this upper bound when summing along specific axes.

        Note: For nansum and masked sum operations, NaN/mask checks are not counted
        as FLOPs since they are comparisons.

        Args:
            *args: Input arrays (first argument is the array to sum)
            result: Result of the operation
            **kwargs: Additional keyword arguments:
                     - axis: Optional[Union[int, Tuple[int, ...]]] - Axis along which to operate
                     - keepdims: bool - Whether to keep reduced dimensions
                     - dtype: numpy.dtype - Output data type
                     These currently don't affect the FLOP count as we use an upper bound.

        Returns:
            int: Number of floating point operations (additions)
        """
        # Handle both np.sum(arr) and arr.sum() cases
        array = args[0] if args else kwargs.get("self", None)
        if array is None or np.size(array) == 0:
            return 0
        return np.size(array) - 1


class MinOperation:
    """FLOP count for min operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for min operation.

        Finding the minimum requires comparing each pair of elements sequentially.
        This implementation provides an upper bound by counting (n-1) comparisons
        where n is the total size of the array, regardless of the axis parameter.

        Args:
            *args: Input arrays
            result: Result of the operation
            **kwargs: Additional keyword arguments:
                     - axis: Optional[Union[int, Tuple[int, ...]]] - Axis along which to operate
                     - keepdims: bool - Whether to keep reduced dimensions
                     - initial: scalar - Starting value for reduction
                     These currently don't affect the FLOP count as we use an upper bound.

        Returns:
            int: Number of comparisons needed (upper bound)
        """
        return np.size(args[0]) - 1


class MaxOperation:
    """FLOP count for max operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for max operation.

        Finding the maximum requires comparing each pair of elements sequentially.
        This implementation provides an upper bound by counting (n-1) comparisons
        where n is the total size of the array, regardless of the axis parameter.

        Args:
            *args: Input arrays
            result: Result of the operation
            **kwargs: Additional keyword arguments:
                     - axis: Optional[Union[int, Tuple[int, ...]]] - Axis along which to operate
                     - keepdims: bool - Whether to keep reduced dimensions
                     - initial: scalar - Starting value for reduction
                     These currently don't affect the FLOP count as we use an upper bound.

        Returns:
            int: Number of comparisons needed (upper bound)
        """
        return np.size(args[0]) - 1


class ArgminOperation:
    """FLOP count for argmin operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for argmin operation.

        Finding the minimum index requires comparing each element with the current minimum,
        similar to min operation. This implementation provides an upper bound by counting
        (n-1) comparisons where n is the total size of the array, regardless of the axis
        parameter. The actual FLOP count when using specific axes would be less than or
        equal to this bound.

        Args:
            *args: Input arrays
            result: Result of the operation
            **kwargs: Additional keyword arguments:
                     - axis: Optional[Union[int, Tuple[int, ...]]] - Axis along which to operate
                     - keepdims: bool - Whether to keep reduced dimensions
                     These currently don't affect the FLOP count as we use an upper bound.

        Returns:
            int: Number of comparisons needed (upper bound)
        """
        return np.size(args[0]) - 1


class ArgmaxOperation:
    """FLOP count for argmax operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for argmax operation.

        Finding the maximum index requires comparing each element with the current maximum,
        similar to max operation. This implementation provides an upper bound by counting
        (n-1) comparisons where n is the total size of the array, regardless of the axis
        parameter.

        Args:
            *args: Input arrays
            result: Result of the operation
            **kwargs: Additional keyword arguments:
                     - axis: Optional[Union[int, Tuple[int, ...]]] - Axis along which to operate
                     - keepdims: bool - Whether to keep reduced dimensions
                     These currently don't affect the FLOP count as we use an upper bound.

        Returns:
            int: Number of comparisons needed (upper bound)
        """
        return np.size(args[0]) - 1
