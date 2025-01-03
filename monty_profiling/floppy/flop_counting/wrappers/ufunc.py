# flop_counting/wrappers/ufunc.py
from typing import Any, Set
import warnings
from .base import OperationWrapper


class UfuncWrapper(OperationWrapper):
    """Wrapper for NumPy ufuncs."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.not_supported_list: Set[str] = set()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = self.operation(*args, **kwargs)

            if result is None:
                return result

            if self.operation_name in self.flop_counter._operations:
                operation = self.flop_counter._operations[self.operation_name]
                flops = operation.count_flops(*args, result=result)
                if flops is not None:
                    self.flop_counter.add_flops(flops)
            elif self.operation_name not in self.not_supported_list:
                warnings.warn(
                    f"Operation {self.operation_name} not supported for FLOP counting"
                )
                self.not_supported_list.add(self.operation_name)

            return result
        except Exception as e:
            warnings.warn(f"Error in {self.operation_name}: {str(e)}")
            raise
