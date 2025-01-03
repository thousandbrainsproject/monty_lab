# flop_counting/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class FlopOperation(ABC):
    """Base class for all FLOP counting operations."""

    @abstractmethod
    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for the operation given inputs and result."""
        pass

    @abstractmethod
    def validate_inputs(self, *args: Any) -> bool:
        """Validate that inputs are appropriate for this operation."""
        pass
