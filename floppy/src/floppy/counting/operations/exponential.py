import inspect
from typing import Any

import numpy as np

__all__ = [
    "LogOperation",
    "PowerOperation",
]


class LogOperation:
    """FLOP count for logarithm operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for logarithm operation.

        Each logarithm typically requires ~20-30 FLOPs depending on the implementation
        and desired precision. Common implementations use series expansions or
        iterative methods that involve multiple multiplications and divisions.
        We use a conservative estimate of 20 FLOPs per logarithm.

        Args:
            *args: Input arguments to the operation
            result: Result of the operation

        Returns:
            Number of FLOPs
        """
        # Handle Python scalars by checking the first argument
        if np.isscalar(args[0]) and not isinstance(args[0], np.ndarray):
            return 20
        return 20 * np.size(result)


# All operations now handled
class PowerOperation:
    """FLOP count for power operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for power operation.

        FLOP count depends on the exponent:
        - For integer exponents > 0: Uses repeated multiplication, requiring (exponent-1) FLOPs
        - For integer exponent = 0: No FLOPs (just returns 1)
        - For integer exponent < 0: Same as positive + 1 division
        - For sqrt operations: Uses ~20 FLOPs (specialized sqrt algorithm)
        - For cbrt operations: Uses ~25 FLOPs (specialized cube root algorithm)
        - For reciprocal operations: Uses 1 FLOP (single division)
        - For other fractional exponents: Uses logarithm (~20 FLOPs) and exponential (~20 FLOPs), total ~40 FLOPs

        Args:
            args: (base, exponent) or just (base) for square/sqrt/cbrt/reciprocal operations
            result: Result of the operation
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
                flops_per_element = 20  # Specialized sqrt algorithm
            elif exponent == 1 / 3:  # cbrt case
                flops_per_element = 25  # Specialized cube root algorithm
            elif exponent == -1:  # reciprocal case
                flops_per_element = 1  # Single division
            else:
                flops_per_element = (
                    40  # Approximate FLOPs for other fractional exponents
                )
        else:
            # If exponent is an array, use worst case (fractional exponent)
            flops_per_element = 40

        return n * flops_per_element
