"""
Implements `dist_agent_1lm_randrot_nohyp_x_percent_20p`
"""

import copy
from dmc_configs.dmc_eval_experiments import dist_agent_1lm_nohyp_randrot


def update_x_percent_threshold_in_config(
    config, x_percent_threshold, evidence_update_threshold="x_percent_threshold"
):
    """Update the x_percent threshold in the config.

    Args:
        config (dict): The config to update.
        x_percent_threshold (float): The percentage of the threshold to update.

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
        "evidence_update_threshold"
    ] = x_percent_threshold

    # Update the string value for evidence_update_threshold
    lm_config_dict["learning_module_0"]["learning_module_args"][
        "evidence_update_threshold"
    ] = evidence_update_threshold
    return config


dist_agent_1lm_randrot_nohyp_x_percent_5p = copy.deepcopy(dist_agent_1lm_nohyp_randrot)
dist_agent_1lm_randrot_nohyp_x_percent_5p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp_x_percent_5p, 0.05, "x_percent_threshold"
)

dist_agent_1lm_randrot_nohyp_x_percent_10p = copy.deepcopy(dist_agent_1lm_nohyp_randrot)
dist_agent_1lm_randrot_nohyp_x_percent_10p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp_x_percent_10p, 0.1, "x_percent_threshold"
)

dist_agent_1lm_randrot_nohyp_x_percent_20p = copy.deepcopy(dist_agent_1lm_nohyp_randrot)
dist_agent_1lm_randrot_nohyp_x_percent_20p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp_x_percent_20p, 0.2, "x_percent_threshold"
)

dist_agent_1lm_randrot_nohyp_x_percent_30p = copy.deepcopy(dist_agent_1lm_nohyp_randrot)
dist_agent_1lm_randrot_nohyp_x_percent_30p = update_x_percent_threshold_in_config(
    dist_agent_1lm_randrot_nohyp_x_percent_30p, 0.3, "x_percent_threshold"
)

dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all = copy.deepcopy(
    dist_agent_1lm_nohyp_randrot
)
dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all = (
    update_x_percent_threshold_in_config(
        dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all,
        0.3,
        "all",
    )
)
