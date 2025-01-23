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


class PowerOperation:
    """FLOP count for power operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for power operation.

        FLOP count depends on the exponent:
        - For integer exponents > 0: Uses repeated multiplication, requiring (exponent-1) FLOPs
        - For integer exponent = 0: No FLOPs (just returns 1)
        - For integer exponent < 0: Same as positive + 1 division
        - For fractional exponents: Uses logarithm (~20 FLOPs) and exponential (~20 FLOPs), or total ~40 FLOPs

        Args:
            args: (base, exponent)
            result: Result of the operation
        """
        base, exponent = args
        n = np.size(base)

        if np.isscalar(exponent):
            if float(exponent).is_integer():
                exp = abs(int(exponent))
                flops_per_element = max(0, exp - 1)  # exp-1 multiplications needed
                if exponent < 0:
                    flops_per_element += 1  # Additional division for negative exponents
            else:
                flops_per_element = 40  # Approximate FLOPs for fractional exponents
        else:
            # If exponent is an array, use worst case (fractional exponent)
            flops_per_element = 40

        return n * flops_per_element
