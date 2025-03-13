from typing import Any, Optional

import numpy as np

__all__ = [
    "ConvolveOperation",
]


class ConvolveOperation:
    """FLOP count for convolution operation.

    This class implements FLOP counting for numpy's convolve operation.
    For each output element, the convolution requires multiplying corresponding
    elements from the input array and the flipped kernel, then summing these products.
    """

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for convolution operation.

        Args:
            *args: Input arrays (array and kernel to convolve)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.convolve parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            For each output element, convolution requires:
            - kernel_size multiplications (one per kernel element)
            - (kernel_size - 1) additions to sum the products
            Total: (2 * kernel_size - 1) * output_size FLOPs
        """
        if len(args) < 2:
            return 0

        array, kernel = args[:2]
        kernel_size = len(kernel)
        output_size = len(result)

        # For each output element:
        # - kernel_size multiplications
        # - (kernel_size - 1) additions
        flops_per_output = 2 * kernel_size - 1

        return flops_per_output * output_size
