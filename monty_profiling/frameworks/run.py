# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import logging
import time
from tbp.monty.frameworks.run import (
    merge_args,
    print_config,
    run,
    create_cmd_parser,
    config_to_dict,
)
from monty_flops.src.monty_flop_tracer import add_flop_tracking
from typing import Dict, Any


def wrap_monty_with_flops(monty_cls, experiment_cls):
    """Wraps both Monty and Experiment classes to add FLOP tracking."""
    original_setup = experiment_cls.setup_experiment

    def wrapped_setup(self, config):
        """Wrap the setup_experiment to initialize counters first."""
        # Initialize counters before doing anything else
        self.init_counters()
        # Call original setup
        original_setup(self, config)

        flop_tracker = add_flop_tracking(self.model, self)
        self.model.flop_tracker = flop_tracker
        self.flop_tracker = flop_tracker

    # Wrap the experiment's setup and Monty's init
    experiment_cls.setup_experiment = wrapped_setup

    return monty_cls, experiment_cls


def run_with_flops(exp_config: Dict[str, Any]):
    """Runs an experiment with FLOP tracking by modifying both classes."""
    # Get both classes
    original_monty_class = exp_config["monty_config"].get("monty_class")
    original_experiment_class = exp_config.get("experiment_class")

    if original_monty_class is None:
        raise ValueError("No monty_class found in monty_config")
    if original_experiment_class is None:
        raise ValueError("No experiment_class found in exp_config")

    # Wrap both classes
    wrapped_monty, wrapped_experiment = wrap_monty_with_flops(
        original_monty_class, original_experiment_class
    )

    # Update config with wrapped classes
    exp_config["monty_config"]["monty_class"] = wrapped_monty
    exp_config["experiment_class"] = wrapped_experiment

    # Run with the modified config
    result = run(exp_config)
    return result


def flop_main(all_configs, experiments=None):
    """Main function that runs experiments with FLOP counting enabled."""
    cmd_args = None
    if not experiments:
        cmd_parser = create_cmd_parser(all_configs=all_configs)
        cmd_args = cmd_parser.parse_args()
        experiments = cmd_args.experiments
        if cmd_args.quiet_habitat_logs:
            os.environ["MAGNUM_LOG"] = "quiet"
            os.environ["HABITAT_SIM_LOG"] = "quiet"

    for experiment in experiments:
        exp = all_configs[experiment]
        exp_config = merge_args(exp, cmd_args)
        exp_config = config_to_dict(exp_config)

        # Update run_name and output dir with experiment name
        if not exp_config["logging_config"]["run_name"]:
            exp_config["logging_config"]["run_name"] = experiment
        exp_config["logging_config"]["output_dir"] = os.path.join(
            exp_config["logging_config"]["output_dir"],
            exp_config["logging_config"]["run_name"],
        )

        # If we are not running in parallel, this should always be False
        exp_config["logging_config"]["log_parallel_wandb"] = False

        # Print config if requested
        if cmd_args is not None and cmd_args.print_config:
            print_config(exp_config)
            continue

        # Create output directory
        os.makedirs(exp_config["logging_config"]["output_dir"], exist_ok=True)

        # Run the experiment with FLOP tracking
        start_time = time.time()
        run_with_flops(exp_config)
        logging.info(f"Done running {experiment} in {time.time() - start_time} seconds")