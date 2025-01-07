import numpy as np
import warnings
from typing import Any, Optional, Tuple
from .base import BaseOperation


class ArithmeticOperation(BaseOperation):
    """Base class for arithmetic operations."""

    def validate_inputs(self, *args: Any) -> bool:
        """Validate that inputs can be processed."""
        if len(args) < 2:
            return False
        try:
            # Convert inputs to arrays for shape analysis
            arrays = [np.asarray(arg) for arg in args[:2]]
            return True
        except Exception:
            return False

    def _compute_broadcast_shape(
        self, *shapes: Tuple[int, ...]
    ) -> Optional[Tuple[int, ...]]:
        """Compute the broadcast shape for the inputs."""
        try:
            # Create dummy arrays of ones with the given shapes
            arrays = [np.ones(shape) for shape in shapes]
            # Use broadcast_arrays instead of broadcast
            result = np.broadcast_arrays(*arrays)
            return result[0].shape
        except ValueError as e:
            warnings.warn(f"Invalid broadcast shape: {str(e)}")
            return None

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count the FLOPs for the operation."""
        if not self.validate_inputs(*args):
            return None

        arrays = [np.asarray(arg) for arg in args[:2]]
        shapes = [arr.shape for arr in arrays]

        # Get the output shape (either broadcast or identical)
        output_shape = self._compute_broadcast_shape(*shapes)
        if output_shape is None:
            return None

        # One FLOP per element in the output
        return np.prod(output_shape)


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
