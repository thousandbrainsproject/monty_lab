import csv
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LogLevel(Enum):
    OPERATION = "operation"
    FUNCTION = "function"
    FILE = "file"


@dataclass
class Operation:
    """Represents a single FLOP operation"""

    flops: int
    filename: str
    line_no: int
    function_name: str
    timestamp: float
    episode: Optional[int] = None
    parent_method: Optional[str] = None
    is_wrapped_method: bool = False


class BaseLogger:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.buffer: List[Operation] = []

    def log_operation(self, operation: Operation) -> None:
        """Add operation to buffer and flush if batch size reached"""
        self.buffer.append(operation)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        raise NotImplementedError("Implement flush() in subclass")


class DetailedLogger(BaseLogger):
    """Handles logging of FLOP operations with detailed information to .log file"""

    def __init__(
        self,
        logger: logging.Logger,
        batch_size: int = 10_000,
        log_level: LogLevel = LogLevel.FUNCTION,
    ):
        super().__init__(batch_size)
        self.logger = logger
        self.log_level = log_level
        self.function_counts = defaultdict(int)
        self.current_file_flops = defaultdict(int)
        self.last_file = None

    def log_operation(self, operation: Operation) -> None:
        """Add operation to buffer and aggregate counts based on log level"""
        self.buffer.append(operation)

        # Aggregate counts for FUNCTION level
        if self.log_level == LogLevel.FUNCTION:
            key = (operation.filename, operation.function_name)
            self.function_counts[key] += operation.flops

        # Similarly for FILE level
        elif self.log_level == LogLevel.FILE:
            # If we switch to a different file, log the previous file's FLOPs
            if self.last_file is not None and operation.filename != self.last_file:
                if self.current_file_flops[self.last_file] > 0:
                    self.logger.debug(
                        f"Accumulated FLOPs: {self.current_file_flops[self.last_file]} | "
                        f"File: {self.last_file}"
                    )
                self.current_file_flops[self.last_file] = 0

            # Add FLOPs to current file's count
            self.current_file_flops[operation.filename] += operation.flops
            self.last_file = operation.filename

        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered operations based on configured log level"""
        if not self.buffer:
            return

        if self.log_level == LogLevel.OPERATION:
            self._log_operations()
        elif self.log_level == LogLevel.FUNCTION:
            self._log_function_counts()
        else:
            self._log_file_counts()

        self.buffer.clear()

    def _log_operations(self) -> None:
        """ Log individual operations.
        Note: This induces a lot of overhead and large file size but will give us the most detailed logs.
        """
        for op in self.buffer:
            self.logger.debug(
                f"FLOPs: {op.flops} | File: {op.filename} | "
                f"Line: {op.line_no} | Function: {op.function_name}"
            )

    def _log_function_counts(self) -> None:
        """Log aggregated counts by function"""
        items = list(self.function_counts.items())

        for (filename, function_name), count in items:
            self.logger.debug(f"Accumulated FLOPs: {count} | Function: {function_name} | File: {filename}")
        self.function_counts.clear()

    def _log_file_counts(self) -> None:
        """Log aggregated counts by file"""
        items = list(self.current_file_flops.items())
        for filename, count in items:
            if count > 0:
                self.logger.debug(f"Accumulated FLOPs: {count} | File: {filename}")
        self.current_file_flops.clear()
        self.last_file = None


class CSVLogger(BaseLogger):
    def __init__(self, filepath: str, batch_size: int = 10_000, max_rows: int = 10_000):
        self.base_filepath = filepath
        self.max_rows = max_rows
        self.total_rows = 0
        self.current_file_number = 0
        self.current_filepath = self._get_filepath()
        super().__init__(batch_size)
        self._initialize_csv()

    def _get_filepath(self):
        path = Path(self.base_filepath)
        return str(path.parent / f"{path.stem}_{self.current_file_number}{path.suffix}")

    def _initialize_csv(self) -> None:
        Path(self.current_filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.current_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "episode", "method", "flops", "filename", "line_no", "parent_method"])

    def flush(self) -> None:
        if not self.buffer:
            return

        with open(self.current_filepath, "a", newline="") as f:
            writer = csv.writer(f)
            for op in self.buffer:
                writer.writerow([op.timestamp, op.episode, op.function_name, op.flops, op.filename, op.line_no, op.parent_method])
        self.total_rows += len(self.buffer)
        if self.total_rows >= self.max_rows:
            self.current_file_number += 1
            self.total_rows = 0
            self.current_filepath = self._get_filepath()
            self._initialize_csv()
        self.buffer.clear()

class LogManager:
    def __init__(self, detailed_logger: Optional[DetailedLogger] = None, csv_logger: Optional[CSVLogger] = None):
        self.detailed_logger = detailed_logger
        self.csv_logger = csv_logger

    def log_operation(self, operation: Operation) -> None:
        if self.detailed_logger:
            self.detailed_logger.log_operation(operation)
        if self.csv_logger and operation.is_wrapped_method:
            self.csv_logger.log_operation(operation)

    def flush(self) -> None:
        if self.detailed_logger:
            self.detailed_logger.flush()
        if self.csv_logger:
            self.csv_logger.flush()
