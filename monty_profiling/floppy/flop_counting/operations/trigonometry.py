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

        Sine is typically implemented using Taylor series:
        sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...

        Each term requires:
        - Power calculation (2-3 FLOPs)
        - Factorial division (1 FLOP)
        - Addition to sum (1 FLOP)

        With ~7-8 terms for good precision, plus argument reduction,
        we estimate 20 FLOPs per value. This was also chosen to be consistent
        with the log operation, which also uses some form of series expansion.
        """
        return 20 * np.size(result)


class CosineOperation:
    """FLOP counter for cosine operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for cosine operation.

        Cosine is typically implemented using Taylor series:
        cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...

        Each term requires:
        - Power calculation (2-3 FLOPs)
        - Factorial division (1 FLOP)
        - Addition to sum (1 FLOP)

        With ~7-8 terms for good precision, plus argument reduction,
        we estimate 20 FLOPs per value for consistency with sine and log operations.
        """
        return 20 * np.size(result)

class CrossOperation:
    """FLOP counter for vector cross product operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for cross product operation.

        Note: Cross product is only defined for 3D vectors (and 7D, though rarely used).
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

        Using Taylor series: arctan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...
        For 7 terms, counting per element:

        Term 1 (x):      1 FLOP  (copy)
        Term 2 (x³/3):   4 FLOPs (2 for x³, 1 div, 1 add)
        Term 3 (x⁵/5):   6 FLOPs (4 for x⁵, 1 div, 1 add)
        Term 4 (x⁷/7):   8 FLOPs (6 for x⁷, 1 div, 1 add)
        Term 5 (x⁹/9):   10 FLOPs (8 for x⁹, 1 div, 1 add)
        Term 6 (x¹¹/11): 12 FLOPs (10 for x¹¹, 1 div, 1 add)
        Term 7 (x¹³/13): 14 FLOPs (12 for x¹³, 1 div, 1 add)

        Total per element: 55 FLOPs
        """
        return 55 * np.size(result)


class ArcSineOperation:
    """FLOP counter for inverse sine operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for inverse sine operation.

        Arcsine is typically implemented using a combination of
        logarithms and square roots. We estimate 10 FLOPs per value
        based on common implementations.
        """
        return 10 * np.size(result)
