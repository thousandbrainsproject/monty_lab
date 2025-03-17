# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import inspect
from typing import Any, Optional

import numpy as np

from ..base.protocols import FlopOperation

__all__ = [
    "ExponentialOperation",
    "LogOperation",
    "PowerOperation",
]


class ExponentialOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for exponential operations.

    Handles both scalar and array exponential operations. Each exponential operation
    typically requires multiple FLOPs due to the series expansion or iterative methods
    used in its implementation. The operation is counted as 20 FLOPs per element,
    which is a conservative estimate based on common implementations.


    """

    # Constants for the exponential operation
    FLOPS_PER_EXP = 20  # Conservative estimate of FLOPs per exponential

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> Optional[int]:
        """Counts the floating point operations (FLOPs) for exponential operations.

        Args:
            *args: Variable length argument list containing the input operand.
                  Can be a scalar or numpy array.
            result: The result of the exponential operation, used to determine the
                   final shape after broadcasting.
            **kwargs: Additional keyword arguments that match numpy.exp parameters.
                     These currently don't affect the FLOP count.

        Returns:
            Optional[int]: The total number of FLOPs performed in the operation.
                         Returns None if operation cannot be performed.
                         Calculated as FLOPS_PER_EXP * size(result) where each element requires:
                         - Series expansion or iterative method computations (~20 FLOPs)

        Note:
            The FLOP count is based on common implementations that use series expansions
            or iterative methods. The actual number of FLOPs may vary depending on the
            implementation and desired precision, but 20 FLOPs is a conservative estimate
            that covers most practical cases.
        """
        # Handle Python scalars by checking the first argument
        if np.isscalar(args[0]) and not isinstance(args[0], np.ndarray):
            return self.FLOPS_PER_EXP
        return self.FLOPS_PER_EXP * np.size(result)


class LogOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for logarithm operations.

    Handles both scalar and array logarithm operations. Each logarithm operation
    typically requires multiple FLOPs due to the series expansion or iterative methods
    used in its implementation. The operation is counted as 20 FLOPs per element,
    which is a conservative estimate based on common implementations.

    """

    # Constants for the logarithm operation
    FLOPS_PER_LOG = 20  # Conservative estimate of FLOPs per logarithm

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> Optional[int]:
        """Counts the floating point operations (FLOPs) for logarithm operations.

        Args:
            *args: Variable length argument list containing the input operand.
                  Can be a scalar or numpy array.
            result: The result of the logarithm operation, used to determine the
                   final shape after broadcasting.
            **kwargs: Additional keyword arguments that match numpy.log parameters.
                     These currently don't affect the FLOP count.

        Returns:
            Optional[int]: The total number of FLOPs performed in the operation.
                         Returns None if operation cannot be performed.
                         Calculated as FLOPS_PER_LOG * size(result) where each element requires:
                         - Series expansion or iterative method computations (~20 FLOPs)

        Note:
            The FLOP count is based on common implementations that use series expansions
            or iterative methods. The actual number of FLOPs may vary depending on the
            implementation and desired precision, but 20 FLOPs is a conservative estimate
            that covers most practical cases.
        """
        # Handle Python scalars by checking the first argument
        if np.isscalar(args[0]) and not isinstance(args[0], np.ndarray):
            return self.FLOPS_PER_LOG
        return self.FLOPS_PER_LOG * np.size(result)


class PowerOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for power operations.

    Handles various types of power operations including integer powers, square roots,
    cube roots, reciprocals, and general fractional powers. The FLOP count varies
    significantly based on the type of power operation being performed.

    Example shapes:
        Integer power: base ** integer -> same shape as base
        Square root: sqrt(array) -> same shape as array
        Cube root: cbrt(array) -> same shape as array
        Reciprocal: 1/array -> same shape as array
        Fractional power: base ** fraction -> same shape as base
    """

    # Constants for different types of power operations
    FLOPS_PER_SQRT = 20  # Specialized square root algorithm
    FLOPS_PER_CBRT = 25  # Specialized cube root algorithm
    FLOPS_PER_RECIPROCAL = 1  # Single division operation
    FLOPS_PER_FRACTIONAL = 40  # Logarithm + exponential for fractional powers

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> Optional[int]:
        """Counts the floating point operations (FLOPs) for power operations.

        The FLOP count depends on the type of power operation:
        - Integer powers > 0: Uses repeated multiplication (exponent-1 FLOPs)
        - Integer power = 0: No FLOPs (returns 1)
        - Integer power < 0: Same as positive + 1 division
        - Square root: Uses specialized sqrt algorithm (~20 FLOPs)
        - Cube root: Uses specialized cube root algorithm (~25 FLOPs)
        - Reciprocal: Single division operation (1 FLOP)
        - Other fractional powers: Uses logarithm (~20 FLOPs) and exponential (~20 FLOPs)

        Args:
            *args: Variable length argument list containing:
                  - For regular power: (base, exponent)
                  - For special operations: (base) where the operation type is determined
                    from the calling function name (sqrt, cbrt, reciprocal, square)
            result: The result of the power operation, used to determine the
                   final shape after broadcasting.
            **kwargs: Additional keyword arguments that match numpy power operation parameters.
                     These currently don't affect the FLOP count.

        Returns:
            Optional[int]: The total number of FLOPs performed in the operation.
                         Returns None if operation cannot be performed.
                         Calculated as flops_per_element * size(result) where flops_per_element
                         depends on the type of power operation.

        Note:
            The operation type is determined by:
            1. The number of arguments (1 or 2)
            2. The calling function name (for special operations)
            3. The type and value of the exponent (for regular power operations)

            The FLOP counts are based on common implementations and may vary depending
            on the specific implementation and desired precision.
        """
        # Get the operation name from the stack
        frame = inspect.currentframe()
        try:
            while frame:
                if "func_name" in frame.f_locals:
                    func_name = frame.f_locals["func_name"]
                    if func_name in ["sqrt", "cbrt", "reciprocal", "square"]:
                        if func_name == "sqrt":
                            exponent = 0.5
                        elif func_name == "cbrt":
                            exponent = 1 / 3
                        elif func_name == "reciprocal":
                            exponent = -1
                        elif func_name == "square":
                            exponent = 2
                        base = args[0]
                        break
                frame = frame.f_back
            else:
                # If no special function found, this is a regular power operation
                if len(args) == 1:
                    base = args[0]
                    exponent = 2  # default for square
                else:
                    base, exponent = args

        finally:
            del frame

        # Get size from either operand, whichever is larger
        n = max(np.size(base), np.size(exponent))

        if np.isscalar(exponent):
            if float(exponent).is_integer():
                exp = abs(int(exponent))
                flops_per_element = max(0, exp - 1)  # exp-1 multiplications needed
                if exponent < 0:
                    flops_per_element += 1  # Additional division for negative exponents
            elif exponent == 0.5:  # sqrt case
                flops_per_element = self.FLOPS_PER_SQRT
            elif exponent == 1 / 3:  # cbrt case
                flops_per_element = self.FLOPS_PER_CBRT
            elif exponent == -1:  # reciprocal case
                flops_per_element = self.FLOPS_PER_RECIPROCAL
            else:
                flops_per_element = self.FLOPS_PER_FRACTIONAL
        else:
            # If exponent is an array, use worst case (fractional exponent)
            flops_per_element = self.FLOPS_PER_FRACTIONAL

        return n * flops_per_element
