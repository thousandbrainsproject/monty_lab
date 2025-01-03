from .base import BaseOperation
import numpy as np
from typing import Any, Optional


class SineOperation(BaseOperation):
    """FLOP counter for sine operations."""

    def __init__(self):
        super().__init__("sin")

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for sine operation.

        Sine is typically implemented using Taylor series or CORDIC algorithm.
        We estimate 8 FLOPs per value based on common implementations.
        """
        if not self.validate_inputs(*args):
            return None

        return 8 * np.size(result)

    def validate_inputs(self, *args: Any) -> bool:
        if len(args) != 1:
            return False
        try:
            np.asarray(args[0])
            return True
        except Exception:
            return False


class CosineOperation(BaseOperation):
    """FLOP counter for cosine operations."""

    def __init__(self):
        super().__init__("cos")

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for cosine operation.

        Cosine is typically implemented using Taylor series or CORDIC algorithm.
        We estimate 8 FLOPs per value based on common implementations.
        """
        if not self.validate_inputs(*args):
            return None

        return 8 * np.size(result)

    def validate_inputs(self, *args: Any) -> bool:
        if len(args) != 1:
            return False
        try:
            np.asarray(args[0])
            return True
        except Exception:
            return False


class CrossOperation(BaseOperation):
    """FLOP counter for vector cross product operations."""

    def __init__(self):
        super().__init__("cross")

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for cross product operation.

        For 3D vectors, cross product requires:
        - 6 multiplications
        - 3 subtractions
        Total: 9 FLOPs per cross product
        """
        # Get number of cross products being computed
        num_operations = max(
            1, result.shape[0] if isinstance(result, np.ndarray) else 1
        )
        return 9 * num_operations

class ArccosOperation(BaseOperation):
    """FLOP counter for inverse cosine operations."""

    def __init__(self):
        super().__init__("arccos")

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for inverse cosine operation.

        Arccos is typically implemented using a combination of
        logarithms and square roots. We estimate 10 FLOPs per value
        based on common implementations.
        """
        return 10 * np.size(result)
