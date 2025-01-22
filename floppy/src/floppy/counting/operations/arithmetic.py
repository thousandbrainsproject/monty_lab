import numpy as np
from typing import Any

__all__ = [
    "Addition",
    "Subtraction",
    "Multiplication",
    "Division",
    "FloorDivideOperation",
    "ModuloOperation",
]

class ArithmeticOperation:
    """Base class for arithmetic operations."""

    def __init__(self, name: str):
        self.name = name

    def count_flops(self, *args: Any, result: Any) -> int:
        """Counts the floating point operations (FLOPs) for an arithmetic operation.

        Handles both scalar and array operations. For scalar operations, counts one FLOP
        per element in the non-scalar array. For array operations, counts one FLOP per
        element in the result array, taking into account broadcasting.

        Args:
            *args: Variable length argument list containing the input operands.
                Expected to have at least 2 arguments where each can be a scalar
                or numpy array.
            result: The result of the arithmetic operation, used to determine the
                final shape after broadcasting.

        Returns:
            int: The total number of FLOPs performed in the operation, calculated as:
                - For scalar operations: size of the non-scalar array
                - For array operations: product of the result array's dimensions
        """
        # Handle scalar operations
        if np.isscalar(args[0]) or np.isscalar(args[1]):
            array_arg = args[1] if np.isscalar(args[0]) else args[0]
            return np.size(array_arg)

        # Get the result shape directly from the actual result
        # This ensures we correctly account for broadcasting
        return np.size(np.asarray(result))


class Addition(ArithmeticOperation):
    """Class for addition operation."""

    def __init__(self):
        super().__init__("add")


class Subtraction(ArithmeticOperation):
    """Class for subtraction operation."""

    def __init__(self):
        super().__init__("subtract")


class Multiplication(ArithmeticOperation):
    """Class for multiplication operation."""

    def __init__(self):
        super().__init__("multiply")


class Division(ArithmeticOperation):
    """Class for division operation.

    Note:
        While division operations can require multiple FLOPs in hardware (e.g., 4 FLOPs),
        we follow the common convention of counting it as 1 FLOP for simplicity.
        This aligns with standard practice in numerical analysis
        (see https://www.stat.cmu.edu/~ryantibs/convexopt-F18/scribes/Lecture_19.pdf).
    """

    def __init__(self):
        super().__init__("divide")

class FloorDivideOperation:
    """FLOP count for floor divide operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for floor divide operation.

        Each element requires:
        - 1 division operation
        - 1 floor/truncation operation
        Total: 2 FLOPs per element
        """
        return 2 * np.size(args[0])


class ModuloOperation:
    """FLOP count for modulo operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for modulo operation.

        Each element requires:
        - 1 division (quotient = a ÷ b)
        - 1 multiplication (product = quotient * b)
        - 1 subtraction (remainder = a - product)
        Total: 3 FLOPs per element
        """
        return 3 * np.size(args[0])