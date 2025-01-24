from typing import Any, Optional

import numpy as np

__all__ = [
    "SineOperation",
    "ArcSineOperation",
    "CosineOperation",
    "ArccosOperation",
    "TangentOperation",
    "ArcTangentOperation",
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


class ArccosOperation:
    """FLOP counter for inverse cosine operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for inverse cosine operation.

        Arccos can be computed using arctan:
        arccos(x) = 2 * arctan(sqrt(1-x)/sqrt(1+x))

        This requires:
        - Two subtractions (1-x, 1+x): 2 FLOPs
        - Two square roots: 20 FLOPs (10 each, using Newton iteration)
        - One division: 1 FLOP
        - One arctan: 20 FLOPs
        - One multiplication by 2: 1 FLOP

        Total per element: 44 FLOPs
        """
        # Handle both scalar and array inputs
        input_value = args[0]
        num_elements = 1 if np.isscalar(input_value) else np.size(input_value)
        return 44 * num_elements


class TangentOperation:
    """FLOP counter for tangent operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for tangent operation.

        Tangent can be computed using Taylor series:
        tan(x) = x + x³/3 + 2x⁵/15 + 17x⁷/315 + ...

        Each term requires:
        - Power calculation (2-3 FLOPs)
        - Coefficient multiplication/division (1-2 FLOPs)
        - Addition to sum (1 FLOP)

        With ~7-8 terms for good precision, plus argument reduction,
        we estimate 20 FLOPs per value for consistency with other trig operations.
        """
        return 20 * np.size(result)


class ArcTangentOperation:
    """FLOP counter for inverse tangent operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for inverse tangent operation.

        Using Taylor series: arctan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...

        Each term requires:
        - Power calculation (2-3 FLOPs)
        - Division by odd number (1 FLOP)
        - Addition to sum (1 FLOP)

        With ~7-8 terms for good precision, plus argument reduction,
        we estimate 20 FLOPs per value for consistency with sine operation.
        """
        return 20 * np.size(result)


class ArcSineOperation:
    """FLOP counter for inverse sine operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for inverse sine operation.

        Arcsine can be computed using arctan:
        arcsin(x) = arctan(x/sqrt(1-x²))

        This requires:
        - One multiplication (x²): 1 FLOP
        - One subtraction (1-x²): 1 FLOP
        - One square root: 10 FLOPs (using Newton iteration)
        - One division: 1 FLOP
        - One arctan: 20 FLOPs

        Total per element: 33 FLOPs
        """
        return 33 * np.size(result)
