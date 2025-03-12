# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 8: Flops Comparison.

This module defines the following experiments:
 - `dist_agent_1lm_randrot_nohyp_x_percent_5p`
 - `dist_agent_1lm_randrot_nohyp_x_percent_10p`
 - `dist_agent_1lm_randrot_nohyp_x_percent_20p`
 - `dist_agent_1lm_randrot_nohyp_x_percent_40p`
 - `dist_agent_1lm_randrot_nohyp_x_percent_60p`
 - `dist_agent_1lm_randrot_nohyp_x_percent_80p`
 - `dist_agent_1lm_randrot_x_percent_5p`
 - `dist_agent_1lm_randrot_x_percent_10p`
 - `dist_agent_1lm_randrot_x_percent_20p`
 - `dist_agent_1lm_randrot_x_percent_40p`
 - `dist_agent_1lm_randrot_x_percent_60p`
 - `dist_agent_1lm_randrot_x_percent_80p`

Experiments use:
 - 77 objects
 - 5 random rotations
 - No sensor noise
 - No voting

The main output measure is accuracy and FLOPs as a function of x-percent threshold and whether
hypothesis testing is used.
"""

import copy

from .fig5_rapid_inference_with_voting import (
    dist_agent_1lm_randrot_noise,  # With hypothesis testing
)
from .fig6_rapid_inference_with_model_based_policies import (
    dist_agent_1lm_randrot_noise_nohyp,
)  # No hypothesis testing


def update_x_percent_threshold_in_config(
    template: dict,
    x_percent_threshold: int,
    evidence_update_threshold: str = "80%",
) -> dict:
    """Update the x_percent threshold in the config.
    This function modifies the config in-place.

    Args:
        template (dict): The config to update.
        x_percent_threshold (int): The percentage of the threshold to update.
        evidence_update_threshold (str): The evidence update threshold to set.

    Returns:
        dict: The updated config.
    """
    config = copy.deepcopy(template)

    # Update the x_percent_threshold
    lm_config_dict = config["monty_config"].learning_module_configs
    lm_config_dict["learning_module_0"]["learning_module_args"][
        "x_percent_threshold"
    ] = x_percent_threshold

    # Set the evidence update to "80%"
    lm_config_dict["learning_module_0"]["learning_module_args"][
        "evidence_update_threshold"
    ] = evidence_update_threshold

    # Update the logging run name
    config[
        "logging_config"
    ].run_name = f"{config['logging_config'].run_name}_x_percent_{x_percent_threshold}"

    return config


################
# Base Configs #
################
"""Creates two base configurations:

1. dist_agent_1lm_randrot_nohyp:
   - Based on dist_agent_1lm_randrot_noise_nohyp
   - Disables sensor noise
   - No hypothesis testing
   
2. dist_agent_1lm_randrot:
   - Based on dist_agent_1lm_randrot_noise
   - Disables sensor noise
   - Includes hypothesis testing

Both configurations serve as templates for the various x-percent threshold experiments
that follow. The main difference between them is the presence/absence of hypothesis
testing functionality.
"""

# Define dist_agent_1lm_randrot_nohyp
dist_agent_1lm_randrot_nohyp = copy.deepcopy(dist_agent_1lm_randrot_noise_nohyp)
for sm_dict in dist_agent_1lm_randrot_nohyp[
    "monty_config"
].sensor_module_configs.values():
    sm_args = sm_dict["sensor_module_args"]
    if sm_args["sensor_module_id"] == "view_finder":
        continue
    sm_args["noise_params"] = {}  # Set noise_param to empty dictionary to remove noise
dist_agent_1lm_randrot_nohyp["logging_config"].run_name = "dist_agent_1lm_randrot_nohyp"

# Define dist_agent_1lm_randrot
dist_agent_1lm_randrot = copy.deepcopy(dist_agent_1lm_randrot_noise)
for sm_dict in dist_agent_1lm_randrot["monty_config"].sensor_module_configs.values():
    sm_args = sm_dict["sensor_module_args"]
    if sm_args["sensor_module_id"] == "view_finder":
        continue
    sm_args["noise_params"] = {}  # Set noise_param to empty dictionary to remove noise
dist_agent_1lm_randrot["logging_config"].run_name = "dist_agent_1lm_randrot"

#####################################################################
# No Hypothesis Testing Configs with different x percent thresholds #
#####################################################################
dist_agent_1lm_randrot_nohyp_x_percent_5p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp, 5
)
dist_agent_1lm_randrot_nohyp_x_percent_10p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp, 10
)
dist_agent_1lm_randrot_nohyp_x_percent_20p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp, 20
)
dist_agent_1lm_randrot_nohyp_x_percent_40p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp, 40
)
dist_agent_1lm_randrot_nohyp_x_percent_60p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp, 60
)
dist_agent_1lm_randrot_nohyp_x_percent_80p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp, 80
)

##################################################################
# Hypothesis Testing Configs with different x percent thresholds #
##################################################################
dist_agent_1lm_randrot_x_percent_5p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot, 5
)
dist_agent_1lm_randrot_x_percent_10p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot, 10
)
dist_agent_1lm_randrot_x_percent_20p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot, 20
)
dist_agent_1lm_randrot_x_percent_40p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot, 40
)
dist_agent_1lm_randrot_x_percent_60p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot, 60
)
dist_agent_1lm_randrot_x_percent_80p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot, 80
)


CONFIGS = {
    "dist_agent_1lm_randrot_nohyp_x_percent_5p": dist_agent_1lm_randrot_nohyp_x_percent_5p,
    "dist_agent_1lm_randrot_nohyp_x_percent_10p": dist_agent_1lm_randrot_nohyp_x_percent_10p,
    "dist_agent_1lm_randrot_nohyp_x_percent_20p": dist_agent_1lm_randrot_nohyp_x_percent_20p,
    "dist_agent_1lm_randrot_nohyp_x_percent_40p": dist_agent_1lm_randrot_nohyp_x_percent_40p,
    "dist_agent_1lm_randrot_nohyp_x_percent_60p": dist_agent_1lm_randrot_nohyp_x_percent_60p,
    "dist_agent_1lm_randrot_nohyp_x_percent_80p": dist_agent_1lm_randrot_nohyp_x_percent_80p,
    "dist_agent_1lm_randrot_x_percent_5p": dist_agent_1lm_randrot_x_percent_5p,
    "dist_agent_1lm_randrot_x_percent_10p": dist_agent_1lm_randrot_x_percent_10p,
    "dist_agent_1lm_randrot_x_percent_20p": dist_agent_1lm_randrot_x_percent_20p,
    "dist_agent_1lm_randrot_x_percent_40p": dist_agent_1lm_randrot_x_percent_40p,
    "dist_agent_1lm_randrot_x_percent_60p": dist_agent_1lm_randrot_x_percent_60p,
    "dist_agent_1lm_randrot_x_percent_80p": dist_agent_1lm_randrot_x_percent_80p,
}
