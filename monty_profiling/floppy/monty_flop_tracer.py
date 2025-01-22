from .flop_counting.counter import FlopCounter
import csv
from pathlib import Path
import time
from typing import Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import inspect


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
        self, experiment_name, monty_instance, experiment_instance, log_path=None
    ):
        self.experiment_name = experiment_name
        self.monty = monty_instance
        self.experiment = experiment_instance
        self.flop_counter = FlopCounter()
        self.total_flops = 0
        self.current_episode = 0
        self._method_stack = []
        self._active_counter = False
        self._current_flops_stack = []
        self._call_stack = []  # Track actual call stack, not just wrapped methods

        # Setup logging
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = (
            Path(
                log_path
                or f"~/tbp/monty_lab/monty_profiling/results/flop_traces_{self.experiment_name}_{timestamp}.csv"
            )
            .expanduser()
            .resolve()
        )
        self._initialize_csv()
        self._original_monty_methods = self._collect_monty_methods()
        self._original_experiment_methods = self._collect_experiment_methods()
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
        print(f"Logged trace: {trace}")

    def _collect_monty_methods(self):
        """Collect methods that need to be wrapped."""
        return {
            # "step": self.monty.step,
            "_matching_step": self.monty._matching_step,
            # "_exploratory_step": self.monty._exploratory_step,
        }

    def _collect_experiment_methods(self):
        """Collect methods that need to be wrapped."""
        print("Collecting experiment methods...")
        print(f"run_episode type: {type(self.experiment.run_episode)}")
        print(f"run_episode: {self.experiment.run_episode}")
        return {
            "run_episode": self.experiment.run_episode,
            "pre_episode": self.experiment.pre_episode,
            # "experiment.pre_step": self.experiment.pre_step,
            # "experiment.post_step": self.experiment.post_step,
            "post_episode": self.experiment.post_episode,
        }

    @contextmanager
    def _method_context(self, method_name: str):
        """Context manager for tracking method hierarchy."""
        self._method_stack.append(method_name)
        try:
            yield
        finally:
            self._method_stack.pop()

    def _create_wrapper(self, method_name: str, original_method: Callable) -> Callable:
        """Create a wrapper for the given method."""

        def wrapped(*args, **kwargs):
            print(f"\nEntering wrapped {method_name}")  # Debug print
            is_outer_call = not self._active_counter

            if is_outer_call:
                print(f"This is an outer call for {method_name}")  # Debug print
                self._active_counter = True
                self.flop_counter.flops = 0

            # Get the actual caller from the call stack
            caller_frame = inspect.currentframe().f_back
            caller_name = caller_frame.f_code.co_name if caller_frame else None
            print(f"Caller for {method_name}: {caller_name}")  # Debug print

            start_flops = self.flop_counter.flops
            self._current_flops_stack.append(start_flops)

            with self._method_context(method_name):
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
                method_name=method_name,
                flops=method_flops,
                parent_method=caller_name,
            )
            self._log_trace(trace)
            print(f"Exiting wrapped {method_name}")  # Debug print

            return result

        return wrapped

    def _wrap_methods(self) -> None:
        """Wrap methods to count FLOPs."""
        print("\nWrapping methods...")
        for method_name, original_method in self._original_monty_methods.items():
            print(f"Wrapping Monty method: {method_name}")
            wrapped_method = self._create_wrapper(method_name, original_method)
            setattr(self.monty, method_name, wrapped_method)

        for method_name, original_method in self._original_experiment_methods.items():
            print(f"Wrapping Experiment method: {method_name}")
            wrapped_method = self._create_wrapper(method_name, original_method)
            print(f"Setting {method_name} on experiment")
            setattr(self.experiment, method_name, wrapped_method)
            print(
                f"After wrapping, {method_name} is now: {getattr(self.experiment, method_name)}"
            )

    def unwrap(self):
        """Restore original methods."""
        for method_name, original_method in self._original_methods.items():
            target = self.monty if hasattr(self.monty, method_name) else self.experiment
            setattr(target, method_name, original_method)

    def reset(self):
        """Reset FLOP counters."""
        self.total_flops = 0
        self.flop_counter.flops = 0
