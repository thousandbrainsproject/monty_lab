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
    "AverageOperation",
    "MeanOperation",
    "MedianOperation",
    "StdOperation",
    "VarOperation",
]


class MeanOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for mean operations.

    Handles both single array and batched computations of arithmetic means.
    Each mean calculation requires summing all elements and dividing by the count.

    The mean computation requires:
    - (n-1) additions to sum all elements
    - 1 division for the final average
    Total: n FLOPs

    Note:
        Scalar inputs return 0 FLOPs as they require no computation.
    """

    # Constants for the mean operation
    ADDS_PER_ELEMENT = 1  # Number of additions per element (except first)
    DIV_COST = 1  # Cost of final division operation

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing mean.

        Args:
            *args: Input arrays where the first array contains the values
                  to compute mean. Must be at least 1D.
            result: The resulting array from the mean operation.
                   Used to determine the number of means computed.
            **kwargs: Additional numpy.mean parameters (e.g., axis, keepdims).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.
                          Returns 0 for scalar inputs.

        Note:
            Mean computation:
            - Each mean requires:
                * (n-1) additions to sum elements
                * 1 division for final average
            - Total FLOPs per mean = n
            - For batched inputs, multiply by batch size
            - Scalar inputs return 0 FLOPs
        """
        if not args:
            return None

        array = args[0]
        # Convert Python scalar to numpy array for consistent handling
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        # Return 0 for scalar inputs
        if array.ndim == 0:
            return 0

        n = np.size(array)
        if n == 0:
            return 0  # Empty array

        return n  # (n-1) additions + 1 division


class StdOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for standard deviation operations.

    Handles both single array and batched computations of standard deviations.
    The computation involves mean calculation, squared differences, and square root.

    The standard deviation computation requires:
    - n FLOPs for mean calculation (from MeanOperation)
    - n subtractions from mean
    - n multiplications for squaring
    - (n-1) additions for sum of squares
    - 1 division for mean of squares
    - 1 square root (costs 20 FLOPs)
    Total: 4n + 20 FLOPs

    Note:
        Scalar inputs return 0 FLOPs as they require no computation.
    """

    # Cost of square root operation (other operations cost 1 FLOP each)
    SQRT_COST = 20

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing standard deviation.

        Args:
            *args: Input arrays where the first array contains the values
                  to compute standard deviation. Must be at least 1D.
            result: The resulting array from the std operation.
                   Used to determine the number of stds computed.
            **kwargs: Additional numpy.std parameters (e.g., axis, ddof).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.
                          Returns 0 for scalar inputs.

        Note:
            Standard deviation computation:
            - Each std requires:
                * n FLOPs for mean calculation (from MeanOperation)
                * n subtractions from mean
                * n multiplications for squaring
                * (n-1) additions for sum
                * 1 division for mean of squares
                * 1 square root (costs 20 FLOPs)
            - Total FLOPs per std = 4n + 20
            - For batched inputs, multiply by batch size
            - Scalar inputs return 0 FLOPs
        """
        if not args:
            return None

        array = args[0]
        # Convert Python scalar to numpy array for consistent handling
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        # Return 0 for scalar inputs
        if array.ndim == 0:
            return 0

        n = np.size(array)
        if n == 0:
            return 0  # Empty array

        # Calculate total FLOPs
        # n FLOPs for mean + n subtractions + n multiplications + (n-1) additions + 1 division + 1 sqrt
        return 4 * n + self.SQRT_COST


class VarOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for variance operations.

    Handles both single array and batched computations of variances.
    The computation involves mean calculation and squared differences.

    The variance computation requires:
    - n FLOPs for mean calculation
    - n subtractions from mean
    - n multiplications for squaring
    - (n-1) additions for sum of squares
    - 1 division for final result

    Note:
        Scalar inputs return 0 FLOPs as they require no computation.
    """

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing variance.

        Args:
            *args: Input arrays where the first array contains the values
                  to compute variance. Must be at least 1D.
            result: The resulting array from the var operation.
                   Used to determine the number of variances computed.
            **kwargs: Additional numpy.var parameters (e.g., axis, ddof).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.
                          Returns 0 for scalar inputs.

        Note:
            Variance computation:
            - Each variance requires:
                * n FLOPs for mean calculation
                * n subtractions from mean
                * n multiplications for squaring
                * (n-1) additions for sum
                * 1 division for final result
            - Total FLOPs per variance = 4n
            - For batched inputs, multiply by batch size
            - Scalar inputs return 0 FLOPs
        """
        if not args:
            return None

        array = args[0]
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        # Return 0 for scalar inputs
        if array.ndim == 0:
            return 0

        n = np.size(array)
        if n == 0:
            return 0  # Empty array

        # Mean calculation (n) + squared differences (2n) + sum (n-1) + division (1)
        return 4 * n


class AverageOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for average operations.

    Handles both weighted and unweighted average computations.
    Supports single array, batched, and multi-dimensional inputs.

    The average computation requires:
    Unweighted:
    - n additions for sum
    - 1 division
    Total: n + 1 FLOPs

    Weighted:
    - n multiplications for weights
    - n additions for weighted sum
    - 1 division by sum of weights
    Total: 2n + 1 FLOPs

    Note:
        Scalar inputs return 0 FLOPs as they require no computation.
    """

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing average.

        Args:
            *args: Input arrays where the first array contains the values
                  to compute average. Must be at least 1D.
            result: The resulting array from the average operation.
                   Used to determine the number of averages computed.
            **kwargs: Additional numpy.average parameters (e.g., weights, axis).
                     The 'weights' parameter affects the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.
                          Returns 0 for scalar inputs.

        Note:
            Average computation:
            - Unweighted average requires:
                * n additions for sum
                * 1 division
            - Weighted average requires:
                * n multiplications for weights
                * n additions for weighted sum
                * 1 division by sum of weights
            - Total FLOPs per average = n + 1 (unweighted) or 2n + 1 (weighted)
            - For batched inputs, multiply by batch size
            - Scalar inputs return 0 FLOPs
        """
        if not args:
            return None

        array = args[0]
        # Convert Python scalar to numpy array for consistent handling
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        # Return 0 for scalar inputs
        if array.ndim == 0:
            return 0

        n = np.size(array)
        if n == 0:
            return 0  # Empty array

        weights = kwargs.get("weights")
        if weights is not None:
            # Weighted sum (2n) + division (1)
            return 2 * n + 1
        # Sum (n) + division (1)
        return n + 1


class MedianOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for median operations.

    Handles both single array and batched computations of medians.
    Only counts floating-point operations, not comparison operations.

    The median computation requires:
    - For even-length arrays:
        * 1 addition to sum middle elements
        * 1 division to compute average
    - For odd-length arrays:
        * No floating-point operations (just selection)
    Total: 2 FLOPs for even-length arrays, 0 for odd-length arrays

    Note:
        Scalar inputs return 0 FLOPs as they require no computation.
    """

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing median.

        Args:
            *args: Input arrays where the first array contains the values
                  to compute median. Must be at least 1D.
            result: The resulting array from the median operation.
                   Used to determine the number of medians computed.
            **kwargs: Additional numpy.median parameters (e.g., axis, keepdims).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.
                          Returns 0 for scalar inputs.

        Note:
            Median computation:
            - For even-length arrays:
                * 1 addition to sum middle elements
                * 1 division to compute average
            - For odd-length arrays:
                * No floating-point operations (just selection)
            - Total FLOPs per median = 2 (even) or 0 (odd)
            - For batched inputs, multiply by batch size
            - Sorting operations are not counted as they are comparisons
            - Scalar inputs return 0 FLOPs
        """
        if not args:
            return None

        array = args[0]
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        # Return 0 for scalar inputs
        if array.ndim == 0:
            return 0

        n = np.size(array)
        if n <= 1:
            return 0  # Empty or single-element array

        # Count both addition and division FLOPs for even-length arrays
        return 2 if n % 2 == 0 else 0
