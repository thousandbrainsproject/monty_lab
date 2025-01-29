# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 7: Flops Comparison.
This module defines the following experiments:
 - `dist_agent_1lm_randrot_nohyp_x_percent_5p`
 - `dist_agent_1lm_randrot_nohyp_x_percent_10p`
 - `dist_agent_1lm_randrot_nohyp_x_percent_20p`
 - `dist_agent_1lm_randrot_nohyp_x_percent_30p`
 - `dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all`

Experiments use:
 - 77 objects
 - 5 random rotations
 - No sensor noise*
 - No hypothesis testing*
 - No voting

 The main output measure is accuracy and FLOPs as a function of x-percent threshold.
"""
# TODO: Set use_multithreading to False for thread-safety
# TODO: Turn off Monty logging in Monty Configs
# TODO: Turn off Python logging in Monty Configs
# TODO: Add tests *with hypothesis testing*
# TODO: Add pretraining experiment with 77 objects and 1 rotation

import copy

from .fig5_rapid_inference_with_model_based_policies import (
    dist_agent_1lm_randrot_noise_nohyp,
)


def update_x_percent_threshold_in_config(
    config, x_percent_threshold, evidence_update_threshold="x_percent_threshold"
):
    """Update the x_percent threshold in the config.
    This function modifies the config in-place.

    Args:
        config (dict): The config to update.
        x_percent_threshold (float): The percentage of the threshold to update.
        evidence_update_threshold (str): How to decide which hypotheses should be updated.
            In [int, float, 'mean', 'median', 'all', 'x_percent_threshold'].

    Returns:
        dict: The updated config.
    """
    # Update the run name
    config[
        "logging_config"
    ].run_name = f"dist_agent_1lm_randrot_nohyp_x_percent_{x_percent_threshold}p"

    # Update the x_percent_threshold
    lm_config_dict = config["monty_config"].learning_module_configs
    lm_config_dict["learning_module_0"]["learning_module_args"][
        "x_percent_threshold"
    ] = x_percent_threshold

    # Update the string value for evidence_update_threshold
    lm_config_dict["learning_module_0"]["learning_module_args"][
        "evidence_update_threshold"
    ] = evidence_update_threshold
    return config


dist_agent_1lm_randrot_nohyp = copy.deepcopy(dist_agent_1lm_randrot_noise_nohyp)
for sm_dict in dist_agent_1lm_randrot_nohyp[
    "monty_config"
].sensor_module_configs.values():
    sm_args = sm_dict["sensor_module_args"]
    if sm_args["sensor_module_id"] == "view_finder":
        continue
    sm_args["noise_params"] = {}  # Set noise_param to empty dictionary to remove noise

dist_agent_1lm_randrot_nohyp_x_percent_5p = copy.deepcopy(
    dist_agent_1lm_randrot_noise_nohyp
)
dist_agent_1lm_randrot_nohyp_x_percent_5p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp_x_percent_5p, 5, "x_percent_threshold"
)

dist_agent_1lm_randrot_nohyp_x_percent_10p = copy.deepcopy(
    dist_agent_1lm_randrot_noise_nohyp
)
dist_agent_1lm_randrot_nohyp_x_percent_10p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp_x_percent_10p, 10, "x_percent_threshold"
)

dist_agent_1lm_randrot_nohyp_x_percent_20p = copy.deepcopy(
    dist_agent_1lm_randrot_noise_nohyp
)
dist_agent_1lm_randrot_nohyp_x_percent_20p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp_x_percent_20p, 20, "x_percent_threshold"
)

dist_agent_1lm_randrot_nohyp_x_percent_30p = copy.deepcopy(
    dist_agent_1lm_randrot_noise_nohyp
)
dist_agent_1lm_randrot_nohyp_x_percent_30p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp_x_percent_30p, 30, "x_percent_threshold"
)

dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all = copy.deepcopy(
    dist_agent_1lm_randrot_noise_nohyp
)
dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all = (
    update_x_percent_threshold_in_config(
        dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all,
        30,
        "all",
    )
)
dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all[
    "logging_config"
].run_name = "dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all"

CONFIGS = {
    "dist_agent_1lm_randrot_nohyp_x_percent_5p": dist_agent_1lm_randrot_nohyp_x_percent_5p,
    "dist_agent_1lm_randrot_nohyp_x_percent_10p": dist_agent_1lm_randrot_nohyp_x_percent_10p,
    "dist_agent_1lm_randrot_nohyp_x_percent_20p": dist_agent_1lm_randrot_nohyp_x_percent_20p,
    "dist_agent_1lm_randrot_nohyp_x_percent_30p": dist_agent_1lm_randrot_nohyp_x_percent_30p,
    "dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all": dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all,
}
