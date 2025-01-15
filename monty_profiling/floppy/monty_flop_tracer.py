from .flop_counting.counter import FlopCounter
import csv
from pathlib import Path
import time

class MontyFlopTracer:
    """Tracks FLOPs for Monty class methods."""

    def __init__(self, monty_instance, experiment_instance, log_path=None):
        self.monty = monty_instance
        self.experiment = experiment_instance
        self.flop_counter = FlopCounter()
        self.total_flops = 0

        # Setup logging with proper path resolution
        self.log_path = (
            Path(log_path or "~/tbp/monty_lab/monty_profiling/results/flop_traces.csv")
            .expanduser()
            .resolve()
        )
        self.episode_start_time = None
        self._setup_csv()
        self._original_methods = {
            # Monty methods
            "_matching_step": self.monty._matching_step,
            "_exploratory_step": self.monty._exploratory_step,
            "post_episode": self.monty.post_episode,
            # MontyExperiment methods
            "run_episode": self.experiment.run_episode,
            "pre_episode": self.experiment.pre_episode,
            "pre_step": self.experiment.pre_step,
            "step": self.monty.step,
            "post_step": self.experiment.post_step,
            "post_episode": self.experiment.post_episode,
        }
        self._wrap_methods()

    def _setup_csv(self):
        """Initialize CSV file with headers."""
        # Create parent directories if they don't exist
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Write headers
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "episode", "method", "flops"])
        self.current_episode = 0

    def _log_flops(self, method_name, flops):
        """Log FLOP counts to CSV file."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), self.current_episode, method_name, flops])

    def _wrap_methods(self):
        """Wrap key Monty methods to count FLOPs."""

        def wrap_method(method_name, target_obj):
            original = self._original_methods[method_name]

            def wrapped(*args, **kwargs):
                self.flop_counter.flops = 0

                with self.flop_counter:
                    result = original(*args, **kwargs)

                step_flops = self.flop_counter.flops
                self._log_flops(method_name, step_flops)
                self.total_flops += step_flops
                return result

            setattr(target_obj, method_name, wrapped)

        # Special wrapper for run_episode to track total FLOPs independently
        def wrapped_run_episode(*args, **kwargs):
            self.current_episode += 1
            episode_start_flops = self.total_flops
            result = self._original_methods["run_episode"](*args, **kwargs)
            episode_total = self.total_flops - episode_start_flops
            self._log_flops("run_episode_total", episode_total)
            return result

        # Wrap Monty methods
        wrap_method("_matching_step", self.monty)
        wrap_method("_exploratory_step", self.monty)

        # Wrap MontyExperiment methods
        experiment_methods = [
            "pre_episode",
            "pre_step",
            "step",
            "post_step",
            "post_episode",
        ]
        for method in experiment_methods:
            wrap_method(method, self.experiment)

        # Set the special run_episode wrapper
        setattr(self.experiment, "run_episode", wrapped_run_episode)

    def unwrap(self):
        """Restore original methods."""
        for method_name, original in self._original_methods.items():
            setattr(self.monty, method_name, original)

    def reset(self):
        """Reset FLOP counters."""
        self.total_flops = 0
        self.flop_counter.flops = 0


# Helper function to easily add FLOP tracking to any Monty instance
def add_flop_tracking(monty_instance, experiment_instance):
    """Adds FLOP tracking to both Monty and MontyExperiment instances."""
    return MontyFlopTracer(monty_instance, experiment_instance)
