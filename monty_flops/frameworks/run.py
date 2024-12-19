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

from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM
from frameworks.models.evidence_matching import FlopCounterEvidenceGraphLM
from playground.flop_counter import FlopCounter
from frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionFlopsExperiment,
)


def inject_flop_counter(config):
    """Replace EvidenceGraphLM with FLOPCounterEvidenceGraphLM in any config."""
    # if "learning_module_configs" in config["monty_config"]:
    #     lm_configs = config["monty_config"]["learning_module_configs"]

    #     for lm_config_key in lm_configs.keys():  # e.g. dict_keys(['learning_module_0'])
    #         specific_lm_config = lm_configs[lm_config_key]
    #         if specific_lm_config["learning_module_class"] == EvidenceGraphLM:
    #             specific_lm_config["learning_module_class"] = FlopCounterEvidenceGraphLM

    # Replace MontyExperiment class with MontyFlopsExperiment
    config["experiment_class"] = MontyObjectRecognitionFlopsExperiment
    return config


def flop_main(all_configs, experiments=None):
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
        exp_config = merge_args(exp, cmd_args)  # TODO: is this really even necessary?
        exp_config = config_to_dict(exp_config)

        # Update run_name and output dir with experiment name
        # NOTE: wandb args are further processed in monty_experiment
        if not exp_config["logging_config"]["run_name"]:
            exp_config["logging_config"]["run_name"] = experiment
        exp_config["logging_config"]["output_dir"] = os.path.join(
            exp_config["logging_config"]["output_dir"],
            exp_config["logging_config"]["run_name"],
        )
        # If we are not running in parallel, this should always be False
        exp_config["logging_config"]["log_parallel_wandb"] = False

        ######## INJECT FLOP COUNTER ########
        exp_config = inject_flop_counter(exp_config)

        # Print config, including udpates to run name
        if cmd_args is not None:
            if cmd_args.print_config:
                print_config(exp_config)
                continue

        os.makedirs(exp_config["logging_config"]["output_dir"], exist_ok=True)
        start_time = time.time()
        run(exp_config)

        logging.info(f"Done running {experiment} in {time.time() - start_time} seconds")
