# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import inspect
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Tuple

from floppy.counting.base import FlopCounter
from floppy.counting.logger import CSVLogger, DetailedLogger, FlopLogEntry, LogManager


class MontyFlopTracer:
    """A specialized tracer for monitoring and logging FLOP operations in Monty-based experiments.

    The MontyFlopTracer wraps key methods in Monty-based experiments to track floating point
    operations (FLOPs) across different components of the system. It provides detailed insights
    into computational complexity and performance characteristics of various experiment stages.

    Key Features:
        - Comprehensive method wrapping for Monty, Experiment, DataLoader, and MotorSystem classes
        - Hierarchical tracking of method calls and their FLOP counts
        - Detailed CSV and optional debug logging of operations
        - Episode-based tracking for experiment progression
        - Thread-safe operation counting
        - Support for enabling/disabling FLOP monitoring dynamically

    The tracer automatically wraps methods of interest and maintains a stack of active
    method calls to properly attribute FLOPs to their source. It can generate both
    high-level CSV summaries and detailed debug logs of FLOP operations.

    Attributes:
        experiment_name (str): Name of the experiment being traced
        monty: Instance of the Monty class being monitored
        experiment: Instance of the Experiment class being monitored
        results_dir (str): Directory for storing trace results
        total_flops (int): Total number of FLOPs counted
        current_episode (int): Current episode number being processed
        flop_counter (FlopCounter): Counter instance for tracking FLOPs
        log_manager (LogManager): Manager for logging FLOP operations

    Examples:
        >>> # Initialize tracer with experiment components
        >>> tracer = MontyFlopTracer(
        ...     experiment_name="my_experiment",
        ...     monty_instance=monty,
        ...     experiment_instance=experiment,
        ...     results_dir="~/results",
        ...     detailed_logging=True
        ... )
        >>>
        >>> # Methods are automatically wrapped and traced
        >>> experiment.run_episode()  # FLOPs will be counted and logged
        >>>
        >>> # Disable monitoring when done
        >>> tracer.disable_monty_flop_monitoring()
    """

    def __init__(
        self,
        experiment_name: str,
        monty_instance: Any,
        experiment_instance: Any,
        results_dir: str,
        detailed_logging: bool = False,
        detailed_logger_kwargs: Optional[Dict[str, Any]] = None,
        csv_logger_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a MontyFlopTracer instance.

        This constructor sets up FLOP tracing for a Monty-based experiment by wrapping key methods
        of the provided component instances. It initializes logging infrastructure and prepares
        the tracer to monitor computational operations across the experiment lifecycle.

        Args:
            experiment_name: Unique identifier for the experiment being traced.
            monty_instance: Instance of the Monty class whose methods will be traced.
            experiment_instance: Instance of the Experiment class managing the experiment flow.
            results_dir: Path to directory where trace results will be saved.
            detailed_logging: Whether to enable detailed debug logging of operations.
                When True, generates a separate log file with method-level FLOP details.
            detailed_logger_kwargs: Optional configuration parameters for the detailed logger.
                See DetailedLogger class for available options.
            csv_logger_kwargs: Optional configuration parameters for the CSV logger.
                See CSVLogger class for available options.

        Raises:
            ValueError: If results_dir is not provided.
        """
        if not results_dir:
            raise ValueError("results_dir must be provided")

        self.experiment_name = experiment_name
        self.monty = monty_instance
        self.experiment = experiment_instance
        self.results_dir = results_dir

        self._initialize_log_manager(
            detailed_logging, detailed_logger_kwargs, csv_logger_kwargs
        )
        self.flop_counter = FlopCounter(
            log_manager=self.log_manager,
            include_paths=["tbp.monty"],
            skip_paths=[
                "site-packages",
                "numpy",
                "scipy",
                "habitat_sim",
            ],
        )

        self.total_flops = 0
        self.current_episode = 0
        self._method_stack = []
        self._active_counter = False
        self._current_flops_stack = []

        self._original_monty_methods = self._collect_monty_methods()
        self._original_experiment_methods = self._collect_experiment_methods()
        self.enable_monty_flop_monitoring()

    def enable_monty_flop_monitoring(self) -> None:
        """Enable FLOP counting for key Monty algorithmic components.

        This method wraps specific methods in Monty that are critical for the
        Demonstrating Monty Capabilities paper.
        """
        for method_name, (
            original_method,
            full_name,
        ) in self._original_monty_methods.items():
            wrapped_method = self._create_wrapper(
                method_name, original_method, full_name
            )
            setattr(self.monty, method_name, wrapped_method)

        for method_name, (
            original_method,
            full_name,
        ) in self._original_experiment_methods.items():
            wrapped_method = self._create_wrapper(
                method_name, original_method, full_name
            )
            setattr(self.experiment, method_name, wrapped_method)

    def disable_monty_flop_monitoring(self) -> None:
        """Disable FLOP counting and restore original Monty methods.

        This method unwraps all previously monitored methods, restoring them to their
        original implementations.

        Returns:
            None
        """
        for method_name, (original_method, _) in self._original_monty_methods.items():
            setattr(self.monty, method_name, original_method)
        for method_name, (
            original_method,
            _,
        ) in self._original_experiment_methods.items():
            setattr(self.experiment, method_name, original_method)

    def reset(self) -> None:
        """Reset all FLOP counters to their initial state.

        Note:
            This does not affect any logged data that has already been written to files.
        """
        self.total_flops = 0
        self.flop_counter.flops = 0

    def _initialize_log_manager(
        self,
        detailed_logging: bool,
        detailed_logger_kwargs: Optional[Dict[str, Any]] = None,
        csv_logger_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the logging infrastructure for FLOP counting.

        Sets up both CSV and optional detailed logging for FLOP operations. Creates necessary
        directories and log files with timestamps for unique identification.

        Args:
            detailed_logging: If True, enables detailed debug-level logging of operations
                in addition to CSV logging.
            detailed_logger_kwargs: Optional configuration dictionary for the DetailedLogger.
                If None, default settings will be used.
            csv_logger_kwargs: Optional configuration dictionary for the CSVLogger.
                If None, default settings will be used.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(self.results_dir).expanduser().resolve()
        csv_path = (
            self.results_dir / f"flop_traces_{self.experiment_name}_{timestamp}.csv"
        )

        csv_logger = CSVLogger(csv_path, **csv_logger_kwargs)

        detailed_logger = None
        if detailed_logging:
            log_path = (
                self.results_dir
                / f"detailed_flops_{self.experiment_name}_{timestamp}.log"
            )
            logger = logging.getLogger(f"detailed_flops_{self.experiment_name}")

            # Disable output to console
            logger.propagate = False
            logger.handlers.clear()

            file_handler = logging.FileHandler(str(log_path))
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)
            detailed_logger = DetailedLogger(logger=logger, **detailed_logger_kwargs)

        self.log_manager = LogManager(
            detailed_logger=detailed_logger, csv_logger=csv_logger
        )

    def _collect_monty_methods(self) -> Dict[str, Tuple[Callable[..., Any], str]]:
        """Collect Monty methods that need to be wrapped for FLOP counting.

        Each collected method is paired with its fully qualified name for logging purposes
        as Monty and Experiment can have same method names.

        Returns:
            Dict[str, Tuple[Callable[..., Any], str]]: A dictionary mapping method names to tuples.
            Each tuple contains:
                - The original method (Callable)
                - The fully qualified method name (str) used for logging

        Example return value:
            {
                "_matching_step": (<bound method>, "monty._matching_step"),
                "_exploratory_step": (<bound method>, "monty._exploratory_step"),
                ...
            }
        """
        return {
            "_matching_step": (self.monty._matching_step, "monty._matching_step"),
            "_exploratory_step": (
                self.monty._exploratory_step,
                "monty._exploratory_step",
            ),
            "_step_learning_modules": (
                self.monty._step_learning_modules,
                "monty._step_learning_modules",
            ),
        }

    def _collect_experiment_methods(self) -> Dict[str, Tuple[Callable[..., Any], str]]:
        """Collect Experiment methods that need to be wrapped for FLOP counting.

        Returns:
            Dict[str, Tuple[Callable[..., Any], str]]: A dictionary mapping method names to tuples.
            Each tuple contains:
                - The original method (Callable)
                - The fully qualified method name (str) used for logging

        Example return value:
            {
                "run_episode": (<bound method>, "experiment.run_episode"),
                "pre_epoch": (<bound method>, "experiment.pre_epoch"),
                ...
            }
        """
        return {
            "run_episode": (self.experiment.run_episode, "experiment.run_episode"),
            "pre_epoch": (self.experiment.pre_epoch, "experiment.pre_epoch"),
            "pre_episode": (self.experiment.pre_episode, "experiment.pre_episode"),
            "pre_step": (self.experiment.pre_step, "experiment.pre_step"),
            "post_step": (self.experiment.post_step, "experiment.post_step"),
            "post_episode": (self.experiment.post_episode, "experiment.post_episode"),
            "run_epoch": (self.experiment.run_epoch, "experiment.run_epoch"),
        }

    @contextmanager
    def _method_context(self, method_name: str) -> Generator[None, None, None]:
        """Context manager for tracking the hierarchical call stack of monitored methods.

        This context manager maintains a stack of method names to track the execution hierarchy
        of monitored methods in the Monty framework. It's used to understand which methods
        are calling other methods and to properly attribute FLOP counts in nested calls.

        Args:
            method_name: The fully qualified name of the method (e.g., 'experiment.run_episode')
                       being added to the method stack.

        Yields:
            None: Control is yielded back to the wrapped code block.
        """
        self._method_stack.append(method_name)
        try:
            yield
        finally:
            self._method_stack.pop()

    def _create_wrapper(
        self, method_name: str, original_method: Callable[..., Any], full_name: str
    ) -> Callable[..., Any]:
        """Create a wrapper that tracks FLOP counts and method execution hierarchy for Monty framework methods.

        This wrapper provides high-level method tracing by:
        1. Managing the FLOP counter's activation state for top-level method calls
        2. Tracking the method call hierarchy using a stack
        3. Recording method-specific FLOP counts and their attribution
        4. Logging detailed method execution information including caller context
        5. Handling episode counting for experiment progression

        Args:
            method_name: The name of the method being wrapped (e.g., 'run_episode')
            original_method: The original method implementation to be wrapped
            full_name: The fully qualified name of the method (e.g., 'experiment.run_episode')
                      used for accurate logging and method hierarchy tracking

        Returns:
            A wrapped version of the method that provides FLOP counting and execution tracing
            while maintaining the original method's functionality and signature.

        Note:
            This wrapper works in conjunction with FlopCounter but operates at a different level:
            - FlopCounter tracks individual NumPy operations and their FLOP counts
            - This wrapper aggregates those FLOP counts and attributes them to specific
              methods in the Monty framework's execution flow
        """

        def wrapped(*args, **kwargs):
            is_outer_call = not self._active_counter

            if is_outer_call:
                self._active_counter = True
                self.flop_counter.flops = 0

            caller_frame = inspect.currentframe().f_back
            caller_name = caller_frame.f_code.co_name if caller_frame else None

            start_flops = self.flop_counter.flops
            self._current_flops_stack.append(start_flops)

            with self._method_context(full_name):  # Use full_name here
                # Only use flop_counter context manager for outer calls
                if is_outer_call:
                    with self.flop_counter:
                        result = original_method(*args, **kwargs)
                else:
                    result = original_method(*args, **kwargs)

            method_flops = self.flop_counter.flops - self._current_flops_stack.pop()

            if is_outer_call:
                self._active_counter = False
                self.total_flops += self.flop_counter.flops

            # Log the operation
            filename = inspect.getfile(original_method)
            line_no = inspect.getsourcelines(original_method)[1]

            operation = FlopLogEntry(
                flops=method_flops,
                filename=filename,
                line_no=line_no,
                function_name=full_name,
                timestamp=time.time(),
                parent_method=caller_name,
                episode=self.current_episode,
                is_wrapped_method=True,
            )
            self.log_manager.log_operation(operation)

            # Increment episode counter after run_episode completes
            if method_name == "run_episode":
                self.current_episode += 1

            return result

        return wrapped
