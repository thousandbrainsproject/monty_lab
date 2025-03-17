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
    "AbsoluteOperation",
    "Addition",
    "DiffOperation",
    "Division",
    "LinspaceOperation",
    "ModuloOperation",
    "Multiplication",
    "Subtraction",
]


class ArithmeticOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for basic arithmetic operations.

    Handles both scalar and array operations. For scalar operations, counts one FLOP
    per element in the non-scalar array. For array operations, counts one FLOP per
    element in the result array, taking into account broadcasting.

    Example shapes:
        Scalar + Array: scalar + (M,N) -> (M,N)
        Array + Array: (M,N) + (M,N) -> (M,N)
        Broadcasted: (M,1) + (1,N) -> (M,N)
    """

    def __init__(self, name: str):
        """Initialize the arithmetic operation.

        Args:
            name: String identifier for the operation (e.g., "add", "subtract")
        """
        self.name = name

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> Optional[int]:
        """Counts the floating point operations (FLOPs) for an arithmetic operation.

        Args:
            *args: Variable length argument list containing the input operands.
                Can be one or more arguments where each can be a scalar
                or numpy array.
            result: The result of the arithmetic operation, used to determine the
                final shape after broadcasting.
            **kwargs: Additional keyword arguments that match numpy arithmetic operation
                     parameters. These currently don't affect the FLOP count.

        Returns:
            Optional[int]: The total number of FLOPs performed in the operation.
                         Returns None if operation cannot be performed.
                         Calculated as:
                         - For scalar operations: size of the non-scalar array
                         - For array operations: product of the result array's dimensions

        Note:
            Each arithmetic operation (add, subtract, multiply, divide) is counted as
            1 FLOP per element, following standard practice in numerical analysis.
            This is a simplified model that may not reflect exact hardware costs
            but provides a consistent baseline for operation counting.
        """
        # If only one argument is provided, return the size of the result
        if len(args) == 1:
            return np.size(result)

        # Handle scalar operations
        if np.isscalar(args[0]) or np.isscalar(args[1]):
            # If both arguments are scalars, return 1 FLOP
            if np.isscalar(args[0]) and np.isscalar(args[1]):
                return 1
            # Find the non-scalar array
            array = next(arg for arg in args if not np.isscalar(arg))
            return np.size(array)

        # For array operations, return the size of the result
        return np.size(result)


class Addition(ArithmeticOperation):
    """Counts floating point operations (FLOPs) for addition operations.

    Handles both scalar and array addition operations. Each element-wise addition
    is counted as 1 FLOP, following standard practice in numerical analysis.
    """

    def __init__(self):
        """Initialize the addition operation."""
        super().__init__("add")


class Subtraction(ArithmeticOperation):
    """Counts floating point operations (FLOPs) for subtraction operations.

    Handles both scalar and array subtraction operations. Each element-wise subtraction
    is counted as 1 FLOP, following standard practice in numerical analysis.
    """

    def __init__(self):
        """Initialize the subtraction operation."""
        super().__init__("subtract")


class Multiplication(ArithmeticOperation):
    """Counts floating point operations (FLOPs) for multiplication operations.

    Handles both scalar and array multiplication operations. Each element-wise multiplication
    is counted as 1 FLOP, following standard practice in numerical analysis.
    """

    def __init__(self):
        """Initialize the multiplication operation."""
        super().__init__("multiply")


class Division(ArithmeticOperation):
    """Counts floating point operations (FLOPs) for division operations.

    Handles both scalar and array division operations. Each element-wise division
    is counted as 1 FLOP, following standard practice in numerical analysis.

    Note:
        While division operations can require multiple FLOPs in hardware (e.g., 4 FLOPs),
        we follow the common convention of counting it as 1 FLOP for simplicity.
        This aligns with standard practice in numerical analysis
        (see https://www.stat.cmu.edu/~ryantibs/convexopt-F18/scribes/Lecture_19.pdf).
    """

    def __init__(self):
        """Initialize the division operation."""
        super().__init__("divide")


class AbsoluteOperation(ArithmeticOperation):
    """Counts floating point operations (FLOPs) for absolute value operations.

    Handles both scalar and array absolute value operations. Each element-wise absolute
    value operation is counted as 1 FLOP, following standard practice in numerical analysis.
    """

    def __init__(self):
        """Initialize the absolute value operation."""
        super().__init__("absolute")


class ModuloOperation:
    """Counts floating point operations (FLOPs) for modulo operations.

    Handles both scalar and array modulo operations. Each element-wise modulo
    operation requires three FLOPs to compute the remainder.
    """

    # Constants for the modulo operation
    DIVS_PER_MOD = 1  # Number of divisions per modulo
    MULTS_PER_MOD = 1  # Number of multiplications per modulo
    SUBS_PER_MOD = 1  # Number of subtractions per modulo
    FLOPS_PER_MOD = DIVS_PER_MOD + MULTS_PER_MOD + SUBS_PER_MOD

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> Optional[int]:
        """Counts the floating point operations (FLOPs) for modulo operations.

        Args:
            *args: Variable length argument list containing the input operands.
                Can be one or more arguments where each can be a scalar
                or numpy array.
            result: The result of the modulo operation, used to determine the
                final shape after broadcasting.
            **kwargs: Additional keyword arguments that match numpy.mod parameters.
                     These currently don't affect the FLOP count.

        Returns:
            Optional[int]: The total number of FLOPs performed in the operation.
                         Returns None if operation cannot be performed.
                         Calculated as FLOPS_PER_MOD * size(result) where each element requires:
                         - DIVS_PER_MOD division (quotient = a ÷ b)
                         - MULTS_PER_MOD multiplication (product = quotient * b)
                         - SUBS_PER_MOD subtraction (remainder = a - product)

        """
        return self.FLOPS_PER_MOD * np.size(result)


class LinspaceOperation:
    """Counts floating point operations (FLOPs) for numpy.linspace operations.

    Handles the generation of evenly spaced numbers over a specified interval.
    Each linspace operation requires:
    - 1 subtraction to compute the interval length (stop - start)
    - 1 division to compute the step size ((stop - start) / (num - 1))
    - (n-1) additions to generate the sequence)
    """

    # Constants for the linspace operation
    SUBS_PER_LINSPACE = 1  # Number of subtractions per linspace
    DIVS_PER_LINSPACE = 1  # Number of divisions per linspace
    ADDS_PER_ELEMENT = 1  # Number of additions per element (except first)

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> Optional[int]:
        """Counts the floating point operations (FLOPs) for linspace operations.

        Args:
            *args: Variable length argument list containing start, stop, and num.
                  The actual values don't affect the FLOP count.
            result: The resulting array from the linspace operation.
                   Used to determine the number of points generated.
                   Can be either a numpy array or a tuple containing (array, step).
            **kwargs: Additional keyword arguments that match numpy.linspace parameters.
                     These currently don't affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                         Returns None if operation cannot be performed.

        Note:
            For n points, linspace performs:
            - 1 subtraction (stop - start)
            - 1 division ((stop - start) / (num - 1))
            - (n-1) additions to generate the sequence
            Total: 2 + (n-1) FLOPs where n is the number of points
        """
        # Handle both array and tuple results (from retstep=True)
        if isinstance(result, tuple):
            result = result[0]  # Get the array part of the result

        if not isinstance(result, np.ndarray):
            return None

        n = np.size(result)
        return (
            self.SUBS_PER_LINSPACE
            + self.DIVS_PER_LINSPACE
            + (n - 1) * self.ADDS_PER_ELEMENT
        )


class DiffOperation:
    """Counts floating point operations (FLOPs) for difference operations.

    Handles both single and multiple difference operations along specified axes.
    Each element in the output requires one subtraction between adjacent elements
    in the input array along the specified axis.
    """

    # Constants for the diff operation
    SUBS_PER_DIFF = 1  # Number of subtractions per difference
    FLOPS_PER_DIFF = SUBS_PER_DIFF  # Total FLOPs per difference

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Counts the floating point operations (FLOPs) for difference operations.

        Args:
            *args: Variable length argument list containing:
                - Input array to compute differences
                - Optional n (number of times to compute differences)
            result: The result array from the difference operation.
                   Used to determine the number of differences computed.
            **kwargs: Additional keyword arguments that match numpy.diff parameters.
                     These currently don't affect the FLOP count.
                     Example kwargs:
                     - axis: Axis along which to compute differences

        Returns:
            int: Number of floating point operations (FLOPs).
                 Calculated as FLOPS_PER_DIFF * size(result) where each element requires:
                 - SUBS_PER_DIFF subtraction between adjacent elements

        Note:
            For an input array of shape (N,) and n=1:
            - Output shape is (N-1,)
            - Each element requires one subtraction
            - Total FLOPs = N-1

            For multiple differences (n>1):
            - Each subsequent difference reduces the output size by 1
            - Total FLOPs = sum(N-i for i in range(1, n+1))
        """
        return self.FLOPS_PER_DIFF * np.size(result)
