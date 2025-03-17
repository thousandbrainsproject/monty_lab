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
    "ArcSineOperation",
    "ArcTangent2Operation",
    "ArcTangentOperation",
    "ArccosOperation",
    "CosineOperation",
    "DegreesOperation",
    "RadiansOperation",
    "SineOperation",
    "TangentOperation",
]

class SineOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for element-wise sine operations.

    Each element-wise sine operation counts as 20 FLOPs, based on Taylor series
    implementation with argument reduction and 4-5 terms for good precision.
    Supports standard NumPy broadcasting rules for input arrays.
    """

    # Constants for the sine operation
    FLOPS_PER_ELEMENT = 20  # FLOPs for one sine calculation

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for sine operations.

        Args:
            *args: Input arrays to compute sine.
                  Typically a single array of angles in radians.
            result: The resulting array or value from the sine operation.
                   Used to determine the total number of operations.
            **kwargs: Additional numpy.sin parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Sine computation using Taylor series:
            sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...

            Cost breakdown per element:
            1. Argument reduction to [-π/2, π/2]: 4 FLOPs
               - Division and modulo by 2π
               - Comparison and adjustment

            2. Taylor series (4-5 terms):
               - Power calculation: 2 FLOPs × 5 terms = 10 FLOPs
               - Factorial division: 1 FLOP × 5 terms = 5 FLOPs
               - Addition to sum: 1 FLOP × 1 terms = 1 FLOP

            Total: ~20 FLOPs per element
        """
        # Handle both scalar and array inputs
        num_elements = 1 if np.isscalar(result) else np.size(result)
        return self.FLOPS_PER_ELEMENT * num_elements


class CosineOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for element-wise cosine operations.

    Each element-wise cosine operation counts as 20 FLOPs, based on Taylor series
    implementation with argument reduction and 4-5 terms for good precision.
    Supports standard NumPy broadcasting rules for input arrays.
    """

    # Constants for the cosine operation
    FLOPS_PER_ELEMENT = 20  # FLOPs for one cosine calculation

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for cosine operations.

        Args:
            *args: Input arrays to compute cosine.
                  Typically a single array of angles in radians.
            result: The resulting array or value from the cosine operation.
                   Used to determine the total number of operations.
            **kwargs: Additional numpy.cos parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Cosine computation using Taylor series:
            cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...

            Cost breakdown per element:
            1. Argument reduction to [-π/2, π/2]: 4 FLOPs
               - Division and modulo by 2π
               - Comparison and adjustment

            2. Taylor series (4-5 terms):
               - Power calculation: 2 FLOPs × 5 terms = 10 FLOPs
               - Factorial division: 1 FLOP × 5 terms = 5 FLOPs
               - Addition to sum: 1 FLOP × 1 terms = 1 FLOP

            Total: ~20 FLOPs per element
        """
        # Handle both scalar and array inputs
        num_elements = 1 if np.isscalar(result) else np.size(result)
        return self.FLOPS_PER_ELEMENT * num_elements


class TangentOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for element-wise tangent operations.

    Each element-wise tangent operation counts as 20 FLOPs, based on Taylor series
    implementation with argument reduction and 4-5 terms for good precision.
    Supports standard NumPy broadcasting rules for input arrays.
    """

    # Constants for the tangent operation
    FLOPS_PER_ELEMENT = 20  # FLOPs for one tangent calculation

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for tangent operations.

        Args:
            *args: Input arrays to compute tangent.
                  Typically a single array of angles in radians.
            result: The resulting array or value from the tangent operation.
                   Used to determine the total number of operations.
            **kwargs: Additional numpy.tan parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Tangent computation using Taylor series:
            tan(x) = x + x³/3 + 2x⁵/15 + 17x⁷/315 + ...

            Cost breakdown per element:
            1. Argument reduction to [-π/2, π/2]: 4 FLOPs
               - Division and modulo by 2π
               - Comparison and adjustment

            2. Taylor series (4-5 terms):
               - Power calculation: 2 FLOPs × 5 terms = 10 FLOPs
               - Coefficient multiplication/division: 1 FLOP × 5 terms = 5 FLOPs
               - Addition to sum: 1 FLOP × 1 terms = 1 FLOP

            Total: ~20 FLOPs per element
        """
        # Handle both scalar and array inputs
        num_elements = 1 if np.isscalar(result) else np.size(result)
        return self.FLOPS_PER_ELEMENT * num_elements


class ArcSineOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for element-wise inverse sine operations.

    Each element-wise arcsine operation counts as 33 FLOPs, based on implementation
    using arctangent and square root operations. Supports standard NumPy broadcasting
    rules for input arrays.
    """

    # Constants for the arcsine operation
    FLOPS_PER_ELEMENT = 33  # FLOPs for one arcsine calculation

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for arcsine operations.

        Args:
            *args: Input arrays to compute arcsine.
                  Typically a single array of values in [-1, 1].
            result: The resulting array or value from the arcsine operation.
                   Used to determine the total number of operations.
            **kwargs: Additional numpy.arcsin parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Arcsine computation using arctangent:
            arcsin(x) = arctan(x/sqrt(1-x²))

            Cost breakdown per element:
            1. Square and subtract operations:
               - Multiplication (x²): 1 FLOP
               - Subtraction (1-x²): 1 FLOP

            2. Square root and division:
               - Square root (using Newton iteration): 10 FLOPs
               - Division: 1 FLOP

            3. Arctangent calculation:
               - One arctangent: 20 FLOPs

            Total: 33 FLOPs per element
        """
        # Handle both scalar and array inputs
        num_elements = 1 if np.isscalar(result) else np.size(result)
        return self.FLOPS_PER_ELEMENT * num_elements


class ArccosOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for element-wise inverse cosine operations.

    Each element-wise arccos operation counts as 44 FLOPs, based on implementation
    using arctangent and square root operations. Supports standard NumPy broadcasting
    rules for input arrays.
    """

    # Constants for the arccos operation
    FLOPS_PER_ELEMENT = 44  # FLOPs for one arccos calculation

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for arccos operations.

        Args:
            *args: Input arrays to compute arccos.
                  Typically a single array of values in [-1, 1].
            result: The resulting array or value from the arccos operation.
                   Used to determine the total number of operations.
            **kwargs: Additional numpy.arccos parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Arccos computation using arctangent:
            arccos(x) = 2 * arctan(sqrt(1-x)/sqrt(1+x))

            Cost breakdown per element:
            1. Addition and subtraction:
               - Two subtractions (1-x, 1+x): 2 FLOPs

            2. Square root operations:
               - Two square roots (using Newton iteration): 20 FLOPs

            3. Division and arctangent:
               - Division: 1 FLOP
               - One arctangent: 20 FLOPs

            4. Final scaling:
               - Multiplication by 2: 1 FLOP

            Total: 44 FLOPs per element
        """
        # Handle both scalar and array inputs
        num_elements = 1 if np.isscalar(result) else np.size(result)
        return self.FLOPS_PER_ELEMENT * num_elements


class ArcTangentOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for element-wise inverse tangent operations.

    Each element-wise arctangent operation counts as 20 FLOPs, based on Taylor series
    implementation with argument reduction and 4-5 terms for good precision.
    Supports standard NumPy broadcasting rules for input arrays.
    """

    # Constants for the arctangent operation
    FLOPS_PER_ELEMENT = 20  # FLOPs for one arctangent calculation

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for arctangent operations.

        Args:
            *args: Input arrays to compute arctangent.
                  Typically a single array of values.
            result: The resulting array or value from the arctangent operation.
                   Used to determine the total number of operations.
            **kwargs: Additional numpy.arctan parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Arctangent computation using Taylor series:
            arctan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...

            Cost breakdown per element:
            1. Argument reduction for |x| > 1: 4 FLOPs
               - Reciprocal and comparison
               - π/2 adjustment when needed

            2. Taylor series (4-5 terms):
               - Power calculation: 2 FLOPs × 5 terms = 10 FLOPs
               - Division by odd number: 1 FLOP × 5 terms = 5 FLOPs
               - Addition to sum: 1 FLOP × 1 terms = 1 FLOP

            Total: ~20 FLOPs per element
        """
        # Handle both scalar and array inputs
        num_elements = 1 if np.isscalar(result) else np.size(result)
        return self.FLOPS_PER_ELEMENT * num_elements


class ArcTangent2Operation(FlopOperation):
    """Counts floating point operations (FLOPs) for two-argument inverse tangent operations.

    Each element-wise arctan2 operation counts as 40 FLOPs, based on implementation
    that handles quadrant determination and special cases. Supports standard NumPy
    broadcasting rules for input arrays.
    """

    # Constants for the arctan2 operation
    FLOPS_PER_ELEMENT = 40  # FLOPs for one arctan2 calculation

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for arctan2 operations.

        Args:
            *args: Input arrays (y, x) to compute arctan2.
            result: The resulting array or value from the arctan2 operation.
                   Used to determine the total number of operations.
            **kwargs: Additional numpy.arctan2 parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Arctan2 computation breakdown per element:
            1. Division and sign handling: 5 FLOPs
               - Division operation (y/x): 1 FLOP
               - Sign checks and comparisons: 4 FLOPs
            2. Basic arctangent computation: 20 FLOPs
            3. Quadrant determination and π adjustments: 8 FLOPs
               - Multiplications/additions with π: 3 FLOPs
               - Additional comparisons and adjustments: 5 FLOPs
            4. Special case handling (x=0, y=0): 7 FLOPs
               - Additional comparisons: 3 FLOPs
               - Conditional operations: 4 FLOPs

            Total: 40 FLOPs per element
        """
        # Handle both scalar and array inputs
        num_elements = 1 if np.isscalar(result) else np.size(result)
        return self.FLOPS_PER_ELEMENT * num_elements


class DegreesOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for angle conversion from radians to degrees.

    Handles conversion of angles from radians to degrees by multiplying by 180/π.
    Supports both scalar and array inputs with standard NumPy broadcasting rules.

    The conversion requires:
    - One multiplication per element (by 180/π)
    Total: n FLOPs where n is the total number of elements
    """

    # Constants for the degrees conversion operation
    MULTS_PER_ELEMENT = 1  # Number of multiplications per element

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for converting radians to degrees.

        Args:
            *args: Input arrays where the first array contains angles in radians.
                  Must be compatible with NumPy broadcasting rules.
            result: The resulting array from the degrees conversion operation.
                   Used to determine the number of elements converted.
            **kwargs: Additional numpy.degrees parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Degrees conversion computation:
            - Each element requires one multiplication (by 180/π)
            - Total FLOPs = number_of_elements
            - Supports broadcasting for batched operations
        """
        if not args:
            return 0

        array = args[0]
        # Handle both scalar and array inputs
        num_elements = 1 if np.isscalar(array) else np.size(array)
        return self.MULTS_PER_ELEMENT * num_elements


class RadiansOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for angle conversion from degrees to radians.

    Handles conversion of angles from degrees to radians by multiplying by π/180.
    Supports both scalar and array inputs with standard NumPy broadcasting rules.

    The conversion requires:
    - One multiplication per element (by π/180)
    Total: n FLOPs where n is the total number of elements
    """

    # Constants for the radians conversion operation
    MULTS_PER_ELEMENT = 1  # Number of multiplications per element

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for converting degrees to radians.

        Args:
            *args: Input arrays where the first array contains angles in degrees.
                  Must be compatible with NumPy broadcasting rules.
            result: The resulting array from the radians conversion operation.
                   Used to determine the number of elements converted.
            **kwargs: Additional numpy.radians parameters (e.g., out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Radians conversion computation:
            - Each element requires one multiplication (by π/180)
            - Total FLOPs = number_of_elements
            - Supports broadcasting for batched operations
        """
        if not args:
            return 0

        array = args[0]
        # Handle both scalar and array inputs
        num_elements = 1 if np.isscalar(array) else np.size(array)
        return self.MULTS_PER_ELEMENT * num_elements
