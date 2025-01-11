import numpy as np
import warnings
from typing import Any, Optional, Tuple

__all__ = ["Addition", "Subtraction", "Multiplication", "Division"]

class ArithmeticOperation:
    """Base class for arithmetic operations."""

    def __init__(self, name: str):
        self.name = name

    def _compute_broadcast_shape(
        self, *shapes: Tuple[int, ...]
    ) -> Optional[Tuple[int, ...]]:
        """Compute the broadcast shape for the inputs."""
        try:
            # Handle case where one input is scalar
            if not all(shapes):
                return shapes[1] if not shapes[0] else shapes[0]

            # Create dummy arrays of ones with the given shapes
            arrays = [np.ones(shape) for shape in shapes]
            result = np.broadcast_arrays(*arrays)
            return result[0].shape
        except ValueError as e:
            warnings.warn(f"Invalid broadcast shape: {str(e)}")
            return None

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count the FLOPs for the operation."""
        # Handle scalar operations
        if np.isscalar(args[0]) or np.isscalar(args[1]):
            array_arg = args[1] if np.isscalar(args[0]) else args[0]
            return np.size(array_arg)

        # Handle array operations
        arrays = [np.asarray(arg) for arg in args[:2]]
        # Get the result shape directly from the actual result
        # This ensures we correctly account for broadcasting
        result_shape = np.asarray(result).shape

        return np.prod(result_shape)


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
    """Class for division operation."""

    def __init__(self):
        super().__init__("divide")
