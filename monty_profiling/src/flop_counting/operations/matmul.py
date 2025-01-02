# flop_counting/operations/matmul.py
import numpy as np
import warnings
from typing import Any, Optional, Tuple
from .base import BaseOperation


class MatmulOperation(BaseOperation):
    """FLOP counter for matrix multiplication operations."""

    def __init__(self):
        super().__init__("matmul")

    def validate_inputs(self, *args: Any) -> bool:
        if not all(isinstance(arg, np.ndarray) for arg in args[:2]):
            return False
        shapes = [arg.shape for arg in args[:2]]
        if len(shapes) < 2:
            return False
        return True

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        if not self.validate_inputs(*args):
            return None

        shapes = [arg.shape for arg in args[:2]]

        # Check for broadcasting
        if shapes[0] != shapes[1]:
            warnings.warn(
                "Broadcasting involved in matmul. FLOP count may be approximate."
            )

        try:
            result_shape = result.shape
            if len(result_shape) < 2:
                return None

            M = result_shape[-2]
            P = result_shape[-1]

            if len(shapes[0]) >= 2 and len(shapes[1]) >= 2:
                N = shapes[0][-1]
                if N != shapes[1][-2]:
                    warnings.warn(
                        f"Invalid matrix dimensions: {shapes[0]} and {shapes[1]}"
                    )
                    return None

                batch_dims = result_shape[:-2]
                batch_count = np.prod(batch_dims) if batch_dims else 1

                return 2 * M * N * P * batch_count
        except (AttributeError, IndexError) as e:
            warnings.warn(f"Error counting matmul FLOPs: {str(e)}")
            return None
