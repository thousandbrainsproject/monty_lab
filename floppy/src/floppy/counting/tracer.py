import csv
import inspect
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.floppy.counting.counter import FlopCounter


@dataclass
class MethodTrace:
    """Dataclass for storing method trace information."""

    timestamp: float
    episode: int
    method_name: str
    flops: int
    parent_method: Optional[str] = None


class MontyFlopTracer:
    """Tracks FLOPs for Monty class methods."""

    def __init__(
        self,
        experiment_name,
        monty_instance,
        experiment_instance,
        train_dataloader_instance,
        eval_dataloader_instance,
        motor_system_instance,
        log_dir=None,
        detailed_logging=True,
    ):
        self.experiment_name = experiment_name
        self.monty = monty_instance
        self.experiment = experiment_instance
        self.train_dataloader = train_dataloader_instance
        self.eval_dataloader = eval_dataloader_instance
        self.motor_system = motor_system_instance
        self.detailed_logging = detailed_logging

        # Setup logging first
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = (
            Path(log_dir or f"~/tbp/monty_lab/floppy/results/counting")
            .expanduser()
            .resolve()
        )
        if not self.log_dir.parent.exists():
            self.log_dir.parent.mkdir(parents=True, exist_ok=True)

        self.log_path = (
            self.log_dir / f"flop_traces_{self.experiment_name}_{timestamp}.csv"
        )

        if detailed_logging:
            # Create a logger with a unique name
            self.logger = logging.getLogger(f"flop_tracer_{timestamp}")
            self.logger.setLevel(logging.DEBUG)
            # Prevent propagation to root logger to avoid duplicate logs
            self.logger.propagate = False

            # Create file handler only (remove console handler)
            file_handler = logging.FileHandler(
                self.log_path.parent
                / f"detailed_flops_{self.experiment_name}_{timestamp}.log"
            )

            # Create formatter
            formatter = logging.Formatter("%(asctime)s | %(message)s")
            file_handler.setFormatter(formatter)

            # Add only file handler to logger
            self.logger.addHandler(file_handler)

        # Initialize FlopCounter with the logger
        self.flop_counter = FlopCounter(
            logger=self.logger if detailed_logging else None
        )

        self.total_flops = 0
        self.current_episode = 0
        self._method_stack = []
        self._active_counter = False
        self._current_flops_stack = []
        self._call_stack = []  # Track actual call stack, not just wrapped methods

        self._initialize_csv()
        self._original_monty_methods = self._collect_monty_methods()
        self._original_experiment_methods = self._collect_experiment_methods()
        if self.train_dataloader is not None:
            self._original_train_dataloader_methods = (
                self._collect_train_dataloader_methods()
            )
        if self.eval_dataloader is not None:
            self._original_eval_dataloader_methods = (
                self._collect_eval_dataloader_methods()
            )
        if self.motor_system is not None:
            self._original_motor_system_methods = self._collect_motor_system_methods()
        self._wrap_methods()

    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        # Create parent directories if they don't exist
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Write headers
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "episode", "method", "flops"])

    def _log_trace(self, trace: MethodTrace):
        """Log a method trace to the CSV file."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    trace.timestamp,
                    trace.episode,
                    trace.method_name,
                    trace.flops,
                    trace.parent_method,
                ]
            )

    def _collect_monty_methods(self):
        """Collect methods that need to be wrapped."""
        return {
            "step": (self.monty.step, "monty.step"),
            "_matching_step": (self.monty._matching_step, "monty._matching_step"),
            "_exploratory_step": (
                self.monty._exploratory_step,
                "monty._exploratory_step",
            ),
        }

    def _collect_experiment_methods(self):
        """Collect methods that need to be wrapped."""
        return {
            "run_episode": (self.experiment.run_episode, "experiment.run_episode"),
            "pre_epoch": (self.experiment.pre_epoch, "experiment.pre_epoch"),
            "pre_episode": (self.experiment.pre_episode, "experiment.pre_episode"),
            "pre_step": (self.experiment.pre_step, "experiment.pre_step"),
            "post_step": (self.experiment.post_step, "experiment.post_step"),
            "post_episode": (self.experiment.post_episode, "experiment.post_episode"),
        }

    def _collect_train_dataloader_methods(self):
        return {
            "pre_episode": (
                self.train_dataloader.pre_episode,
                "train_dataloader.pre_episode",
            ),
        }

    def _collect_eval_dataloader_methods(self):
        return {
            "pre_episode": (
                self.eval_dataloader.pre_episode,
                "eval_dataloader.pre_episode",
            ),
            "get_good_view_with_patch_refinement": (
                self.eval_dataloader.get_good_view_with_patch_refinement,
                "eval_dataloader.get_good_view_with_patch_refinement",
            ),
            "get_good_view": (
                self.eval_dataloader.get_good_view,
                "eval_dataloader.get_good_view",
            ),
        }

    def _collect_motor_system_methods(self):
        return {
            "orient_to_object": (
                self.motor_system.orient_to_object,
                "motor_system.orient_to_object",
            ),
            "move_close_enough": (
                self.motor_system.move_close_enough,
                "motor_system.move_close_enough",
            ),
        }

    @contextmanager
    def _method_context(self, method_name: str):
        """Context manager for tracking method hierarchy."""
        self._method_stack.append(method_name)
        try:
            yield
        finally:
            self._method_stack.pop()

    def _create_wrapper(
        self, method_name: str, original_method: Callable, full_name: str
    ) -> Callable:
        """Create a wrapper for the given method."""

        def wrapped(*args, **kwargs):
            is_outer_call = not self._active_counter

            if is_outer_call:
                self._active_counter = True
                self.flop_counter.flops = 0

            # Get the actual caller from the call stack
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

            trace = MethodTrace(
                timestamp=time.time(),
                episode=self.current_episode,
                method_name=full_name,  # Use full_name here
                flops=method_flops,
                parent_method=caller_name,
            )
            self._log_trace(trace)
            if self.detailed_logging:
                logging.debug(f"Logged trace: {trace}")

            # Increment episode counter after run_episode completes
            if method_name == "run_episode":
                self.current_episode += 1

            return result

        return wrapped

    def _wrap_methods(self) -> None:
        """Wrap methods to count FLOPs."""
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

        if self.train_dataloader is not None:
            for method_name, (
                original_method,
                full_name,
            ) in self._original_train_dataloader_methods.items():
                wrapped_method = self._create_wrapper(
                    method_name, original_method, full_name
                )
                setattr(self.train_dataloader, method_name, wrapped_method)

        if self.eval_dataloader is not None:
            for method_name, (
                original_method,
                full_name,
            ) in self._original_eval_dataloader_methods.items():
                wrapped_method = self._create_wrapper(
                    method_name, original_method, full_name
                )
                setattr(self.eval_dataloader, method_name, wrapped_method)

        if self.motor_system is not None:
            for method_name, (
                original_method,
                full_name,
            ) in self._original_motor_system_methods.items():
                wrapped_method = self._create_wrapper(
                    method_name, original_method, full_name
                )
                setattr(self.motor_system, method_name, wrapped_method)

    def unwrap(self):
        """Restore original methods."""
        for method_name, original_method in self._original_monty_methods.items():
            setattr(self.monty, method_name, original_method)
        for method_name, original_method in self._original_experiment_methods.items():
            setattr(self.experiment, method_name, original_method)

        if self.train_dataloader is not None:
            for (
                method_name,
                original_method,
            ) in self._original_train_dataloader_methods.items():
                setattr(self.train_dataloader, method_name, original_method)

        if self.eval_dataloader is not None:
            for (
                method_name,
                original_method,
            ) in self._original_eval_dataloader_methods.items():
                setattr(self.eval_dataloader, method_name, original_method)

        if self.motor_system is not None:
            for (
                method_name,
                original_method,
            ) in self._original_motor_system_methods.items():
                setattr(self.motor_system, method_name, original_method)

    def reset(self):
        """Reset FLOP counters."""
        self.total_flops = 0
        self.flop_counter.flops = 0
