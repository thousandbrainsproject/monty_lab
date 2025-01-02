# flop_counting/wrappers/base.py
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import warnings


class OperationWrapper(ABC):
    """Base class for operation wrappers."""

    def __init__(
        self, operation: Any, flop_counter: "FlopCounter", operation_name: str
    ):
        self.operation = operation
        self.flop_counter = flop_counter
        self.operation_name = operation_name

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self.operation, name)
