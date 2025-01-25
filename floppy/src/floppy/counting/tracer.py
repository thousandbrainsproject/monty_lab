import csv
import inspect
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from floppy.counting.counter import FlopCounter
from floppy.counting.logger import CSVLogger, DetailedLogger, LogManager, Operation


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
        results_dir: Optional[str] = None,
        detailed_logging: bool = False,
        detailed_logger_kwargs: Optional[Dict[str, Any]] = None,
        csv_logger_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_name = experiment_name
        self.monty = monty_instance
        self.experiment = experiment_instance
        self.train_dataloader = train_dataloader_instance
        self.eval_dataloader = eval_dataloader_instance
        self.motor_system = motor_system_instance
        self.results_dir = results_dir

        self._initialize_log_manager(
            detailed_logging, detailed_logger_kwargs, csv_logger_kwargs
        )
        self.flop_counter = FlopCounter(logger=self.log_manager)

        self.total_flops = 0
        self.current_episode = 0
        self._method_stack = []
        self._active_counter = False
        self._current_flops_stack = []

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

    def _initialize_log_manager(
        self,
        detailed_logging: bool,
        detailed_logger_kwargs: Optional[Dict[str, Any]],
        csv_logger_kwargs: Optional[Dict[str, Any]],
    ):
        """Initialize the log manager."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.results_dir = (
            Path(self.results_dir or f"~/tbp/monty_lab/floppy/results/counting")
            .expanduser()
            .resolve()
        )
        csv_path = (
            self.results_dir / f"flop_traces_{self.experiment_name}_{timestamp}.csv"
        )

        csv_logger = CSVLogger(csv_path, **csv_logger_kwargs)

        detailed_logger = None
        if self.detailed_logging:
            log_path = (
                self.results_dir
                / f"detailed_flops_{self.experiment_name}_{timestamp}.log"
            )
            logger = logging.getLogger(f"detailed_flops_{self.experiment_name}")
            file_handler = logging.FileHandler(str(log_path))
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)
            detailed_logger = DetailedLogger(logger=logger, **detailed_logger_kwargs)

        self.log_manager = LogManager(
            detailed_logger=detailed_logger, csv_logger=csv_logger
        )

    def _collect_monty_methods(self):
        """Collect methods that need to be wrapped."""
        return {
            # "step": (self.monty.step, "monty.step"),
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

    def _collect_experiment_methods(self):
        """Collect methods that need to be wrapped."""
        return {
            "run_episode": (self.experiment.run_episode, "experiment.run_episode"),
            "pre_epoch": (self.experiment.pre_epoch, "experiment.pre_epoch"),
            "pre_episode": (self.experiment.pre_episode, "experiment.pre_episode"),
            "pre_step": (self.experiment.pre_step, "experiment.pre_step"),
            "post_step": (self.experiment.post_step, "experiment.post_step"),
            "post_episode": (self.experiment.post_episode, "experiment.post_episode"),
            "run_epoch": (self.experiment.run_epoch, "experiment.run_epoch"),
        }

    def _collect_train_dataloader_methods(self):
        return {
            # "pre_episode": (
            #     self.train_dataloader.pre_episode,
            #     "train_dataloader.pre_episode",
            # ),
        }

    def _collect_eval_dataloader_methods(self):
        return {
            # "pre_episode": (
            #     self.eval_dataloader.pre_episode,
            #     "eval_dataloader.pre_episode",
            # ),
            # "get_good_view_with_patch_refinement": (
            #     self.eval_dataloader.get_good_view_with_patch_refinement,
            #     "eval_dataloader.get_good_view_with_patch_refinement",
            # ),
            # "get_good_view": (
            #     self.eval_dataloader.get_good_view,
            #     "eval_dataloader.get_good_view",
            # ),
        }

    def _collect_motor_system_methods(self):
        return {
            # "orient_to_object": (
            #     self.motor_system.orient_to_object,
            #     "motor_system.orient_to_object",
            # ),
            # "move_close_enough": (
            #     self.motor_system.move_close_enough,
            #     "motor_system.move_close_enough",
            # ),
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

            # Log the operation
            frame = inspect.currentframe()
            filename = frame.f_code.co_filename
            line_no = frame.f_lineno

            operation = Operation(
                flops=method_flops,
                filename=filename,
                line_no=line_no,
                function_name=full_name,
                timestamp=time.time(),
                parent_method=caller_name,
                episode=self.current_episode,
            )
            self.log_manager.log_operation(operation)

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
