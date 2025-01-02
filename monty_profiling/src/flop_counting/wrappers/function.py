# flop_counting/wrappers/function.py
import numpy as np
from typing import Any
from .base import OperationWrapper


class FunctionWrapper(OperationWrapper):
    """Wrapper for NumPy functions."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = self.operation(*args, **kwargs)

            if self.operation_name == "dot":
                flops = self._count_dot_flops(*args)
                if flops is not None:
                    self.flop_counter.add_flops(flops)

            return result
        except Exception as e:
            warnings.warn(f"Error in {self.operation_name}: {str(e)}")
            raise

    def _count_dot_flops(self, *args: Any) -> Optional[int]:
        if len(args) < 2:
            return None

        a = np.asarray(args[0])
        b = np.asarray(args[1])

        # Vector dot product
        if (a.ndim == 1 or (a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1))) and (
            b.ndim == 1 or (b.ndim == 2 and (b.shape[0] == 1 or b.shape[1] == 1))
        ):
            n = max(a.size, b.size)
            return 2 * n - 1
        # Matrix multiplication
        elif a.ndim >= 2 and b.ndim >= 2:
            return 2 * a.shape[0] * a.shape[1] * b.shape[1]
        return None
