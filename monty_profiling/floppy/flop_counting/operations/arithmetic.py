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
            return np.broadcast(*shapes).shape
        except ValueError as e:
            warnings.warn(f"Invalid broadcast shape: {str(e)}")
            return None

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count the FLOPs for the operation."""
        if not self.validate_inputs(*args):
            return None

        arrays = [np.asarray(arg) for arg in args[:2]]
        shapes = [arr.shape for arr in arrays]

        # If shapes are identical, then no broadcasting is needed
        if shapes[0] == shapes[1]:
            return np.prod(shapes[0]) if shapes[0] else 1

        broadcast_shape = self._compute_broadcast_shape(*shapes)
        if broadcast_shape is None:
            return None

        return np.prod(broadcast_shape)


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
