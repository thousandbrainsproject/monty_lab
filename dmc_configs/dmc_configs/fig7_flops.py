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

import copy
from .fig4_rapid_inference_with_voting import dist_agent_1lm_randrot_noise_nohyp


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


dist_agent_1lm_randrot_no_noise_nohyp = copy.deepcopy(
    dist_agent_1lm_randrot_noise_nohyp
)
# TODO: Remove sensor noise
# Or can I start from fig6's dist_agent_1lm_ranrot_nohyp_1rot_trained? (Though I think I need 14 rot?)


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

CONFIGS = {
    "dist_agent_1lm_randrot_nohyp_x_percent_5p": dist_agent_1lm_randrot_nohyp_x_percent_5p,
    "dist_agent_1lm_randrot_nohyp_x_percent_10p": dist_agent_1lm_randrot_nohyp_x_percent_10p,
    "dist_agent_1lm_randrot_nohyp_x_percent_20p": dist_agent_1lm_randrot_nohyp_x_percent_20p,
    "dist_agent_1lm_randrot_nohyp_x_percent_30p": dist_agent_1lm_randrot_nohyp_x_percent_30p,
    "dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all": dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all,
}
