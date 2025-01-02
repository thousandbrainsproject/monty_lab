# flop_counting/operations/base.py
import warnings
from typing import Any, Optional
from ..base import FlopOperation


class BaseOperation(FlopOperation):
    """Base implementation for simple operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        if not self.validate_inputs(*args):
            return None
        return np.size(result) if isinstance(result, np.ndarray) else 1

    def validate_inputs(self, *args: Any) -> bool:
        return len(args) >= 2
