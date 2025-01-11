import numpy as np
from typing import Any, Optional

__all__ = [
    "SineOperation",
    "CosineOperation",
    "CrossOperation",
    "ArccosOperation",
    "TangentOperation",
    "ArcTangentOperation",
    "ArcSineOperation",
]

class SineOperation:
    """FLOP counter for sine operations."""

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


class CosineOperation:
    """FLOP counter for cosine operations."""

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


class CrossOperation:
    """FLOP counter for vector cross product operations."""

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

class ArccosOperation:
    """FLOP counter for inverse cosine operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for inverse cosine operation.

        Arccos is typically implemented using a combination of
        logarithms and square roots. We estimate 10 FLOPs per value
        based on common implementations.
        """
        return 10 * np.size(result)


class TangentOperation:
    """FLOP counter for tangent operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for tangent operation.

        Tangent is typically implemented as sin/cos, requiring:
        - 8 FLOPs for sine
        - 8 FLOPs for cosine
        - 1 division
        Total: 17 FLOPs per value
        """

        return 17 * np.size(result)


class ArcTangentOperation:
    """FLOP counter for inverse tangent operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for inverse tangent operation.

        Arctangent is typically implemented using Taylor series or rational approximations.
        We estimate 10 FLOPs per value based on common implementations.
        """

        return 10 * np.size(result)


class ArcSineOperation:
    """FLOP counter for inverse sine operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for inverse sine operation.

        Arcsine is typically implemented using a combination of
        logarithms and square roots. We estimate 10 FLOPs per value
        based on common implementations.
        """
        return 10 * np.size(result)
