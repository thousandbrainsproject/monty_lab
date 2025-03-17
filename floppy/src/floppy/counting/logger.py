# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class LogLevel(Enum):
    OPERATION = "operation"
    FUNCTION = "function"
    FILE = "file"


@dataclass
class FlopLogEntry:
    """Represents a single FLOP logging entry with metadata about where and when FLOPs were counted"""

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
        self.buffer: List[FlopLogEntry] = []

    def log_operation(self, operation: FlopLogEntry) -> None:
        """Add operation to buffer and flush if batch size reached"""
        self.buffer.append(operation)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        raise NotImplementedError("Implement flush() in subclass")


class DetailedLogger(BaseLogger):
    """Handles logging of FLOP operations with detailed information to .log file.

    This logger provides three different levels of granularity for logging FLOP operations:

    1. OPERATION (LogLevel.OPERATION):
       - Logs each individual FLOP operation separately
       - Includes details: FLOPs count, file, line number, and function name
       - Most detailed but highest overhead and largest log file size

    2. FUNCTION (LogLevel.FUNCTION):
       - Aggregates FLOP counts at the function level
       - Groups operations by filename and function name
       - Provides a balance between detail and performance
       - Logs accumulated FLOPs per function when buffer is flushed

    3. FILE (LogLevel.FILE):
       - Aggregates FLOP counts at the file level
       - Tracks total FLOPs per file
       - Most efficient in terms of storage and performance
       - Logs accumulated FLOPs whenever switching to a different file

    Args:
        logger (logging.Logger): The logger instance to use for output
        batch_size (int, optional): Number of operations to buffer before flushing. Defaults to 10,000.
        log_level (LogLevel, optional): The granularity level for logging. Defaults to LogLevel.FUNCTION.
    """

    def __init__(
        self,
        logger: logging.Logger,
        batch_size: int = 10_000,
        log_level: LogLevel = LogLevel.FUNCTION,
    ) -> None:
        super().__init__(batch_size)
        self.logger = logger
        self.log_level = log_level
        self.function_counts = defaultdict(int)
        self.current_file_flops = defaultdict(int)
        self.last_file = None

    def log_operation(self, operation: FlopLogEntry) -> None:
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
        """Log individual operations.
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
    """Logs FLOP operations to CSV files with automatic file rotation.

    This logger writes operation data to CSV files, creating new files when the maximum
    number of rows is reached. Each CSV file contains columns for timestamp, episode,
    method name, FLOP count, filename, line number, and parent method.

    The CSV files are named using the pattern: {base_name}_{file_number}.csv

    Args:
        filepath (str): Base filepath for the CSV files. The file number will be inserted
            before the extension (e.g., "flops_0.csv", "flops_1.csv")
        batch_size (int, optional): Number of operations to buffer before writing to file.
            Defaults to 10,000.
        max_rows (int, optional): Maximum number of rows per CSV file before rotating to
            a new file. Defaults to 10,000.
    """

    def __init__(self, filepath: str, batch_size: int = 10_000, max_rows: int = 10_000):
        self.base_filepath = filepath
        self.max_rows = max_rows
        self.total_rows = 0
        self.current_file_number = 0
        self.current_filepath = self._get_filepath()
        super().__init__(batch_size)
        self._initialize_csv()

    def _get_filepath(self) -> str:
        """Generate the current CSV filepath based on the file number.

        Returns:
            str: The complete filepath including the current file number.
        """
        path = Path(self.base_filepath)
        return str(path.parent / f"{path.stem}_{self.current_file_number}{path.suffix}")

    def _initialize_csv(self) -> None:
        """Initialize a new CSV file with header row.

        Creates the necessary directory structure if it doesn't exist and writes
        the column headers to the new CSV file.
        """
        Path(self.current_filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.current_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "episode", "method", "flops", "filename", "line_no", "parent_method"])

    def flush(self) -> None:
        """Write buffered operations to the current CSV file.

        If the number of rows in the current file exceeds max_rows after writing,
        a new CSV file will be created for subsequent operations. Each row contains:
        - timestamp: When the operation occurred
        - episode: Training episode number (if applicable)
        - method: Name of the function where FLOPs were counted
        - flops: Number of floating point operations
        - filename: Source file containing the operation
        - line_no: Line number where the operation occurred
        - parent_method: Name of the parent function (if applicable)
        """
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
    """Manages multiple loggers for FLOP operations with different output formats.

    The LogManager coordinates between a DetailedLogger for human-readable logging output
    and a CSVLogger for machine-readable data collection. It can handle either or both types
    of loggers simultaneously.

    The DetailedLogger writes to a .log file with configurable granularity levels
    (operation, function, or file level), while the CSVLogger writes structured data
    to CSV files with automatic file rotation.

    Note: When using both loggers, only wrapped method operations (is_wrapped_method=True)
    are written to the CSV logger, while all operations are sent to the DetailedLogger.

    Args:
        detailed_logger (Optional[DetailedLogger]): Logger for human-readable output with
            configurable granularity. If None, no detailed logging is performed.
        csv_logger (Optional[CSVLogger]): Logger for CSV output with automatic file
            rotation. If None, no CSV logging is performed.

    Example:
        >>> detailed_logger = DetailedLogger(logger, log_level=LogLevel.FUNCTION)
        >>> csv_logger = CSVLogger("flops.csv", max_rows=10000)
        >>> log_manager = LogManager(detailed_logger, csv_logger)
        >>> log_manager.log_operation(operation)  # Logs to both outputs if appropriate
    """

    def __init__(self, detailed_logger: Optional[DetailedLogger] = None, csv_logger: Optional[CSVLogger] = None):
        self.detailed_logger = detailed_logger
        self.csv_logger = csv_logger

    def log_operation(self, operation: FlopLogEntry) -> None:
        """Log a FLOP operation to the configured loggers.

        The operation is always sent to the DetailedLogger if one is configured.
        For the CSVLogger, only operations with is_wrapped_method=True are logged.

        Args:
            operation (FlopLogEntry): The FLOP operation to log, containing details such as
                flop count, filename, line number, function name, etc.
        """
        if self.detailed_logger:
            self.detailed_logger.log_operation(operation)
        if self.csv_logger and operation.is_wrapped_method:
            self.csv_logger.log_operation(operation)

    def flush(self) -> None:
        """Flush any buffered operations in both loggers.

        This ensures all pending operations are written to their respective outputs.
        Should be called when you want to ensure all data is persisted, typically
        at the end of a logging session or before program termination.
        """
        if self.detailed_logger:
            self.detailed_logger.flush()
        if self.csv_logger:
            self.csv_logger.flush()
