from typing import Any, Optional, Union

import numpy as np

__all__ = [
    "MeanOperation",
    "StdOperation",
    "VarOperation",
    "AverageOperation",
    "MedianOperation",
]


class MeanOperation:
    """FLOP count for mean operation.

    This class implements FLOP counting for numpy's mean operation.
    The mean operation requires summing all elements and dividing by
    the total count.
    """

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for mean operation.

        Args:
            *args: Input arrays (first argument is the array to compute mean)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.mean parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            Mean calculation requires:
            - (n-1) additions to sum all elements
            - 1 division for the final average
            Total: n FLOPs where n is the total number of elements
        """
        if not args:
            return 0

        array = args[0]
        return np.size(array)  # (n-1) additions + 1 division


class StdOperation:
    """FLOP count for standard deviation operation.

    This class implements FLOP counting for numpy's std operation.
    The standard deviation requires computing the mean, subtracting
    it from each element, squaring the differences, computing their
    mean, and taking the square root.
    """

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for standard deviation operation.

        Args:
            *args: Input arrays (first argument is the array to compute std)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.std parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            Standard deviation calculation requires:
            - n FLOPs for mean calculation
            - n subtractions from mean
            - n multiplications for squaring
            - (n-1) additions for sum
            - 1 division for mean of squares
            - 1 square root
            Total: 4n + 1 FLOPs where n is the total number of elements
        """
        if not args:
            return 0

        array = args[0]
        n = np.size(array)
        return 4 * n + 1


class VarOperation:
    """FLOP count for variance operation.

    This class implements FLOP counting for numpy's var operation.
    The variance requires computing the mean, subtracting it from
    each element, squaring the differences, and computing their mean.
    """

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for variance operation.

        Args:
            *args: Input arrays (first argument is the array to compute variance)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.var parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            Variance calculation requires:
            - n FLOPs for mean calculation
            - n subtractions from mean
            - n multiplications for squaring
            - (n-1) additions for sum
            - 1 division for final result
            Total: 4n FLOPs where n is the total number of elements
        """
        if not args:
            return 0

        array = args[0]
        n = np.size(array)
        return 4 * n


class AverageOperation:
    """FLOP count for average operation.

    This class implements FLOP counting for numpy's average operation.
    Supports both weighted and unweighted averages, with different
    FLOP counts for each case.
    """

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for average operation.

        Args:
            *args: Input arrays (first argument is the array to compute average)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.average parameters.
                     The 'weights' parameter affects the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            Unweighted average requires:
            - n additions for sum
            - 1 division
            Total: n + 1 FLOPs

            Weighted average requires:
            - n multiplications for weights
            - n additions for weighted sum
            - 1 division by sum of weights
            Total: 2n + 1 FLOPs

            where n is the total number of elements
        """
        if not args:
            return 0

        array = args[0]
        weights = kwargs.get("weights", None)
        n = np.size(array)

        if weights is not None:
            return 2 * n + 1  # weighted sum (2n) + division (1)
        else:
            return n + 1  # sum (n) + division (1)


class MedianOperation:
    """FLOP count for median operation.

    This class implements FLOP counting for numpy's median operation.
    Only floating-point operations are counted - specifically the addition
    and division required when averaging the middle elements for even-length arrays.
    Sorting operations are not counted as they involve comparisons rather
    than floating-point operations.
    """

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for median operation.

        Args:
            *args: Input arrays (first argument is the array to compute median)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.median parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            Median calculation only counts floating-point operations:
            - 1 addition + 1 division if array length is even (to sum and average middle elements)
            - 0 FLOPs if array length is odd (just selecting middle element)
            - Sorting operations are not counted as they are comparisons, not FLOPs
        """
        if not args:
            return 0

        array = args[0]
        n = np.size(array)
        if n <= 1:
            return 0

        # Count both addition and division FLOPs for even-length arrays
        return 2 if n % 2 == 0 else 0
