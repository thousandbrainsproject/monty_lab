# flop_counting/operations/matmul.py
import numpy as np
import warnings
from typing import Any, Optional, Tuple
from .base import BaseOperation


class MatmulOperation(BaseOperation):
    """FLOP counter for matrix multiplication operations."""

    def __init__(self):
        super().__init__("matmul")

    def _compute_broadcast_batch_shape(
        self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Compute the broadcasted shape of batch dimensions.

        Args:
            shape1: Shape of first array
            shape2: Shape of second array

        Returns:
            Tuple[int, ...]: Broadcasted batch shape
        """
        batch1 = shape1[:-2] if len(shape1) > 2 else ()
        batch2 = shape2[:-2] if len(shape2) > 2 else ()

        if not batch1 and not batch2:
            return ()

        return np.broadcast_shapes(batch1, batch2)

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """
        Count FLOPs for matrix multiplication with broadcasting support.

        For matrix multiplication C = A @ B:
        - Each element in the result requires one multiplication and one addition per
          inner dimension element
        - Total FLOPs = 2 * M * N * P * batch_size
        where:
            M = rows in result
            N = inner dimension (columns of A, rows of B)
            P = columns in result
            batch_size = product of broadcasted batch dimensions

        Args:
            *args: Input arrays
            result: Result of the operation

        Returns:
            Optional[int]: Number of FLOPs or None if invalid
        """
        try:
            a, b = np.asarray(args[0]), np.asarray(args[1])

            # Handle 1D vector cases
            if a.ndim == 1 and b.ndim == 1:
                # Vector dot product: 2N-1 FLOPs (N multiplications, N-1 additions)
                return 2 * a.shape[0] - 1

            if a.ndim == 1:  # 1D × 2D
                M, N = 1, a.shape[0]
                P = b.shape[1]
            elif b.ndim == 1:  # 2D × 1D
                M, N = a.shape
                P = 1
            else:  # ND × ND
                M = a.shape[-2]
                N = a.shape[-1]  # same as b.shape[-2]
                P = b.shape[-1]

            # Compute broadcasted batch shape
            batch_shape = self._compute_broadcast_batch_shape(a.shape, b.shape)
            batch_size = np.prod(batch_shape) if batch_shape else 1

            # Each element requires N multiplications and N-1 additions
            # For each M×P elements in the result
            return batch_size * M * P * (2 * N - 1)

        except Exception as e:
            warnings.warn(f"Error counting matmul FLOPs: {str(e)}")
            return None
