import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class FlopOperation:
    """Represents a single FLOP operation or batch of operations"""

    flops: int
    filename: str
    line_no: int
    function_name: str
    timestamp: float


class FlopLogger:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        batch_size: int = 5000,
        log_level: str = "function",
    ):
        self.logger = logger
        self.batch_size = batch_size
        self.log_level = log_level
        self.operation_buffer: List[FlopOperation] = []
        self.function_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.file_counts: Dict[str, int] = defaultdict(int)

    def log_operation(self, operation: FlopOperation) -> None:
        """Add operation to buffer and flush if batch size reached"""
        self.operation_buffer.append(operation)

        # Update aggregated counts
        self.function_counts[(operation.filename, operation.function_name)] += (
            operation.flops
        )
        self.file_counts[operation.filename] += operation.flops

        if len(self.operation_buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered operations based on configured log level"""
        if not self.logger or not self.operation_buffer:
            return

        if self.log_level == "operation":
            self._log_operations()
        elif self.log_level == "function":
            self._log_function_counts()
        else:  # file level
            self._log_file_counts()

        self.operation_buffer.clear()

    def _log_operations(self) -> None:
        """Log individual operations"""
        for op in self.operation_buffer:
            self.logger.debug(
                f"FLOPs: {op.flops} | File: {op.filename} | "
                f"Line: {op.line_no} | Function: {op.function_name}"
            )

    def _log_function_counts(self) -> None:
        """Log aggregated counts by function"""
        for (filename, function_name), count in self.function_counts.items():
            self.logger.debug(
                f"Accumulated FLOPs: {count} | File: {filename} | "
                f"Function: {function_name}"
            )
        self.function_counts.clear()

    def _log_file_counts(self) -> None:
        """Log aggregated counts by file"""
        for filename, count in self.file_counts.items():
            self.logger.debug(f"Accumulated FLOPs: {count} | File: {filename}")
        self.file_counts.clear()
