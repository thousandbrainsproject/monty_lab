# flop_counting/wrappers/ufunc.py
from typing import Any, Set
import warnings
from .base import OperationWrapper


class UfuncWrapper(OperationWrapper):
    """Wrapper for NumPy ufuncs."""

    # Map ufunc names to operation names in self._operations
    UFUNC_MAP = {
        # Basic arithmetic
        "add": "add",
        "subtract": "subtract",
        "multiply": "multiply",
        "true_divide": "divide",
        "floor_divide": "floor_divide",
        "mod": "mod",
        "bitwise_and": "bitwise_and",
        "bitwise_or": "bitwise_or",
        # Linear algebra
        "matmul": "matmul",
        "dot": "matmul",  # dot maps to matmul
        "cross": "cross",
        "trace": "trace",
        # Math functions
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "arcsin": "arcsin",
        "arccos": "arccos",
        "arctan": "arctan",
        "log": "log",
        # Statistics
        "mean": "mean",
        "std": "std",
        "var": "var",
        "average": "average",
        "amin": "min",
        "amax": "max",
        "argmin": "argmin",
        "argmax": "argmax",
        # Misc
        "clip": "clip",
        "where": "where",
        "round_": "round",
        "isnan": "isnan",
    }

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.not_supported_list: Set[str] = set()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = self.operation(*args, **kwargs)

            if result is None:
                return result

            original_op_name = getattr(self.operation, "__name__", self.operation_name)
            op_name = self.UFUNC_MAP.get(original_op_name, self.operation_name)

            if op_name in self.flop_counter._operations:
                operation = self.flop_counter._operations[op_name]
                flops = operation.count_flops(*args, result=result)
                if flops is not None:
                    self.flop_counter.add_flops(flops)
            elif op_name not in self.not_supported_list:
                warnings.warn(f"Operation {op_name} not supported for FLOP counting")
                self.not_supported_list.add(op_name)

            return result
        except Exception as e:
            warnings.warn(f"Error in {self.operation_name}: {str(e)}")
            raise
