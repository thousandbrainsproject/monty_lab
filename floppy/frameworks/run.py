# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import os
import time
from argparse import Namespace
from typing import Any, Dict, List, Optional, Type

from tbp.monty.frameworks.run import (
    config_to_dict,
    create_cmd_parser,
    merge_args,
    print_config,
    run,
)

from floppy.counting.logger import LogLevel
from floppy.counting.tracer import MontyFlopTracer
from frameworks.models.evidence_matching import FlopCountingEvidenceGraphLM
from frameworks.models.goal_state_generation import (
    FlopCountingEvidenceGoalStateGenerator,
)


def wrap_experiment_with_flops(experiment_cls: Type, run_name: str) -> Type:
    """Modifies Monty experiment class to enable FLOP counting.

    This function modifies the experiment class's setup_experiment method by:
    1. Creating a modified setup method that replaces standard learning modules
       with FLOP-counting versions (FlopCountingEvidenceGraphLM and
       FlopCountingEvidenceGoalStateGenerator)
    2. Initializing a MontyFlopTracer to track FLOP counts
    3. Assigning the MontyFlopTracer's counter to each learning module

    Args:
        experiment_cls: The experiment class to be modified
        run_name (str): Name of the experiment run, used by the MontyFlopTracer

    Returns:
        Type: The modified experiment class with FLOP counting capabilities
    """
    original_setup = experiment_cls.setup_experiment

    def wrapped_setup(self, config: Dict[str, Any]) -> None:
        """Modified setup method that adds FLOP counting capabilities.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing experiment settings
        """
        modified_config = config.copy()
        # Replace standard learning modules with FLOP-counting versions
        for lm_key in modified_config["monty_config"]["learning_module_configs"]:
            lm_config = modified_config["monty_config"]["learning_module_configs"][
                lm_key
            ]
            if lm_config["learning_module_class"] == "EvidenceGraphLM":
                lm_config["learning_module_class"] = FlopCountingEvidenceGraphLM
                lm_config["learning_module_args"]["gsg_class"] = (
                    FlopCountingEvidenceGoalStateGenerator
                )

        original_setup(self, modified_config)

        # Extract Floppy-specific configurations
        floppy_config = modified_config.get("floppy_config", {})
        results_dir = floppy_config.get("results_dir", "")
        detailed_logging = floppy_config.get("detailed_logging", False)

        # Configure logging parameters
        detailed_logger_kwargs = {
            "batch_size": floppy_config.get("detailed_batch_size", 10000),
            "log_level": LogLevel[(floppy_config.get("log_level", "FUNCTION"))],
        }
        csv_logger_kwargs = {
            "batch_size": floppy_config.get("csv_batch_size", 1000),
        }

        # Initialize FLOP tracer with experiment components
        flop_tracer = MontyFlopTracer(
            experiment_name=run_name,
            monty_instance=self.model,
            experiment_instance=self,
            results_dir=results_dir,
            detailed_logging=detailed_logging,
            detailed_logger_kwargs=detailed_logger_kwargs,
            csv_logger_kwargs=csv_logger_kwargs,
        )

        # Share the same FLOP counter across all learning modules
        one_true_flop_counter = flop_tracer.flop_counter
        for lm in self.model.learning_modules:
            if isinstance(lm, FlopCountingEvidenceGraphLM):
                lm.flop_counter = one_true_flop_counter
                if hasattr(lm, "gsg") and isinstance(
                    lm.gsg, FlopCountingEvidenceGoalStateGenerator
                ):
                    lm.gsg.flop_counter = one_true_flop_counter
        self.flop_tracer = flop_tracer

    experiment_cls.setup_experiment = wrapped_setup
    return experiment_cls


def run_with_flops(exp_config: Dict[str, Any]) -> Any:
    """Runs an experiment with FLOP counting enabled.

    Args:
        exp_config (Dict[str, Any]): Experiment configuration dictionary containing
            experiment class and parameters.

    Returns:
        Any: Result of the experiment execution.

    Raises:
        ValueError: If no experiment_class is found in exp_config.
    """
    original_experiment_class = exp_config.get("experiment_class")

    if original_experiment_class is None:
        raise ValueError("No experiment_class found in exp_config")

    run_name = exp_config["logging_config"]["run_name"]
    wrapped_experiment = wrap_experiment_with_flops(original_experiment_class, run_name)

    exp_config["experiment_class"] = wrapped_experiment
    return run(exp_config)


def flop_main(
    all_configs: Dict[str, Any], experiments: Optional[List[str]] = None
) -> None:
    """Main function that runs experiments with FLOP counting enabled.

    This function handles command-line argument parsing, experiment configuration,
    and execution of experiments with FLOP counting capabilities.

    Args:
        all_configs (Dict[str, Any]): Dictionary containing all available experiment
            configurations.
        experiments (Optional[List[str]], optional): List of experiment names to run.
            If None, experiments will be selected via command-line arguments.
            Defaults to None.
    """
    cmd_args: Optional[Namespace] = None
    if not experiments:
        # Set up command-line argument parser with FLOP-specific options
        cmd_parser = create_cmd_parser(experiments=all_configs.keys())
        cmd_parser.add_argument(
            "--detailed_logging",
            action="store_true",
            help="Enable detailed logging of FLOP operations",
        )
        cmd_parser.add_argument(
            "--log_level",
            type=str,
            choices=["FILE", "FUNCTION", "OPERATION"],
            default="FUNCTION",
            help="Level of detail for FLOP logging (if detailed_logging is enabled)",
        )
        cmd_parser.add_argument(
            "--detailed_batch_size",
            type=int,
            default=100000,
            help="Batch size for detailed logger",
        )
        cmd_parser.add_argument(
            "--csv_batch_size",
            type=int,
            default=10000,
            help="Batch size for CSV logger",
        )
        cmd_args = cmd_parser.parse_args()
        experiments = cmd_args.experiments

        # Configure habitat logging based on command-line arguments
        if cmd_args.quiet_habitat_logs:
            os.environ["MAGNUM_LOG"] = "quiet"
            os.environ["HABITAT_SIM_LOG"] = "quiet"

    # Process each experiment in the list
    for experiment in experiments:
        exp = all_configs[experiment]
        exp_config = merge_args(exp, cmd_args)
        exp_config = config_to_dict(exp_config)

        # Update experiment naming and output directory configuration
        if not exp_config["logging_config"]["run_name"]:
            exp_config["logging_config"]["run_name"] = experiment
        exp_config["logging_config"]["output_dir"] = os.path.join(
            exp_config["logging_config"]["output_dir"],
            f"{exp_config['logging_config']['run_name']}_floppy",
        )

        # Configure Floppy-specific settings
        exp_config["floppy_config"] = {
            "results_dir": exp_config["logging_config"]["output_dir"],
            "detailed_logging": cmd_args.detailed_logging if cmd_args else False,
            "log_level": cmd_args.log_level.upper() if cmd_args else "FUNCTION",
            "detailed_batch_size": cmd_args.detailed_batch_size if cmd_args else 10000,
            "csv_batch_size": cmd_args.csv_batch_size if cmd_args else 100,
        }
        # Disable parallel wandb logging for non-parallel execution
        exp_config["logging_config"]["log_parallel_wandb"] = False

        # Print configuration if requested
        if cmd_args is not None and cmd_args.print_config:
            print_config(exp_config)
            continue

        # Create output directory and run the experiment
        os.makedirs(exp_config["logging_config"]["output_dir"], exist_ok=True)

        start_time = time.time()
        run_with_flops(exp_config)
        logging.info(f"Done running {experiment} in {time.time() - start_time} seconds")
