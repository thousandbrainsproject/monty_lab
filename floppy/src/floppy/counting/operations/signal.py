# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Optional

import numpy as np

from ..base.protocols import FlopOperation

__all__ = [
    "ConvolveOperation",
]


class ConvolveOperation(FlopOperation):
    """Counts floating point operations (FLOPs) for convolution operations.

    Handles both single array convolutions and batched computations.
    Each output element requires kernel_size multiplications and (kernel_size - 1) additions.

    Example shapes:
        Single: (N,) x (K,) -> (N+K-1,)
        Batched: (B,N) x (K,) -> (B,N+K-1,)

    The convolution computation requires:
    - kernel_size multiplications per output element
    - (kernel_size - 1) additions per output element
    Total: (2 * kernel_size - 1) * output_size FLOPs where:
    - kernel_size is the length of the kernel array
    - output_size is the length of the result array
    - For batched inputs, multiply by batch size

    Reference: Standard convolution algorithm where each output element is computed
    by multiplying corresponding elements from the input array and the flipped kernel,
    then summing these products.
    """

    # Constants for the convolution operation
    MULTS_PER_OUTPUT = 1  # Number of multiplications per output element
    ADDS_PER_OUTPUT = 1  # Number of additions per output element

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing convolution.

        Args:
            *args: Input arrays where the first array contains the input signal
                  and the second array contains the kernel to convolve with.
                  Must be compatible shapes for convolution.
            result: The resulting array from the convolution operation.
                   Used to determine the number of operations computed.
            **kwargs: Additional numpy.convolve parameters (e.g., mode).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Convolution computation:
            - Each output element requires:
                * kernel_size multiplications
                * (kernel_size - 1) additions
            - Total FLOPs per output element = 2 * kernel_size - 1
            - For batched inputs, multiply by batch size
            - The actual number of operations may vary based on convolution mode
        """
        if len(args) < 2:
            return None

        array, kernel = args[:2]
        if not isinstance(array, np.ndarray) or not isinstance(kernel, np.ndarray):
            return None

        kernel_size = len(kernel)
        if kernel_size == 0:
            return 0  # Empty kernel

        output_size = len(result)
        if output_size == 0:
            return 0  # Empty result

        # For each output element:
        # - kernel_size multiplications
        # - (kernel_size - 1) additions
        flops_per_output = (
            kernel_size * self.MULTS_PER_OUTPUT
            + (kernel_size - 1) * self.ADDS_PER_OUTPUT
        )

        return flops_per_output * output_size
