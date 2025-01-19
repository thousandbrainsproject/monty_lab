from .flop_counting.counter import FlopCounter
import csv
from pathlib import Path
import time
from typing import Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import inspect
import types


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

    def __init__(self, monty_instance, experiment_instance, log_path=None):
        self.monty = monty_instance
        self.experiment = experiment_instance
        self.flop_counter = FlopCounter()
        self.total_flops = 0
        self.current_episode = 0
        self._method_stack = []

        # Setup logging
        self.log_path = (
            Path(log_path or "~/tbp/monty_lab/monty_profiling/results/flop_traces.csv")
            .expanduser()
            .resolve()
        )
        self._initialize_csv()
        self._original_methods = self._collect_methods()
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
                [trace.timestamp, trace.episode, trace.method_name, trace.flops]
            )

    def _collect_methods(self):
        """Collect methods that need to be wrapped."""
        return {
            "monty._matching_step": self.monty._matching_step,
            "monty._exploratory_step": self.monty._exploratory_step,
            # "experiment.run_episode": self.experiment.run_episode,
            # "experiment.pre_episode": self.experiment.pre_episode,
            # "experiment.pre_step": self.experiment.pre_step,
            # "monty.step": self.monty.step,
            # "experiment.post_step": self.experiment.post_step,
            # "experiment.post_episode": self.experiment.post_episode,
        }

    @contextmanager
    def _method_context(self, method_name: str):
        """Context manager for tracking method hierarchy."""
        parent_method = self._method_stack[-1] if self._method_stack else None
        self._method_stack.append(method_name)
        start_flops = self.flop_counter.flops

        try:
            yield parent_method
        finally:
            method_flops = self.flop_counter.flops - start_flops
            self._method_stack.pop()

            trace = MethodTrace(
                timestamp=time.time(),
                episode=self.current_episode,
                method_name=method_name,
                flops=method_flops,
                parent_method=parent_method,
            )
            self._log_trace(trace)

    def _create_wrapper(self, method_name: str, original_method: Callable) -> Callable:
        """Create a wrapper for the given method."""

        # Get method's signature
        sig = inspect.signature(original_method)

        def wrapped(*args, **kwargs):
            with self._method_context(method_name):
                with self.flop_counter:
                    return original_method(*args, **kwargs)

        # Copy the original method's signature and docstring
        wrapped.__signature__ = sig
        wrapped.__doc__ = original_method.__doc__
        wrapped.__wrapped__ = original_method

        # Re-bind to the original instance if necessary
        if inspect.ismethod(original_method):
            return types.MethodType(wrapped, original_method.__self__)
        return wrapped

    def _wrap_methods(self) -> None:
        """Wrap methods to count FLOPs."""
        for method_key, original in self._original_methods.items():
            # Extract target class or instance
            if method_key.startswith("monty."):
                target = self.monty
                method_name = method_key.split(".", 1)[1]
            elif method_key.startswith("experiment."):
                target = self.experiment
                method_name = method_key.split(".", 1)[1]
            else:
                continue

            current_method = getattr(target, method_name, None)
            if current_method and getattr(current_method, "__wrapped__", False):
                # Skip already-wrapped methods
                continue

            # Create a wrapper with the original function/method
            wrapped_method = self._create_wrapper(method_name, original)

            # Handle binding for instance methods
            if hasattr(
                target, method_name
            ):  # This ensures the method exists on the target
                if inspect.ismethod(current_method):
                    # Re-bind to the target instance
                    wrapped_method = types.MethodType(wrapped_method, target)

            # Set the wrapped method directly
            setattr(target, method_name, wrapped_method)

    def unwrap(self):
        """Restore original methods."""
        for method_name, original_method in self._original_methods.items():
            target = self.monty if hasattr(self.monty, method_name) else self.experiment
            setattr(target, method_name, original_method)

    def reset(self):
        """Reset FLOP counters."""
        self.total_flops = 0
        self.flop_counter.flops = 0
