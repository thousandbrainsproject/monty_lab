# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 5: Rapid Inference with Model-Based Policies.

This module defines the following experiments:
 - `dist_agent_1lm_randrot_noise_nohyp`
 - `dist_agent_1lm_randrot_noise_moderatehyp`

 Experiments use:
 - 77 objects
 - Sensor noise and 5 random rotations
 - No voting
 - Varying levels of hypothesis-testing

"""

import copy

from .fig4_rapid_inference_with_voting import dist_agent_1lm_randrot_noise

# No hypothesis-testing
dist_agent_1lm_randrot_noise_nohyp = copy.deepcopy(dist_agent_1lm_randrot_noise)
dist_agent_1lm_randrot_noise_nohyp[
    "logging_config"
].run_name = "dist_agent_1lm_randrot_noise_nohyp"
dist_agent_1lm_randrot_noise_nohyp[
    "monty_config"
].motor_system_config.motor_system_args.use_goal_state_driven_actions = False

# Moderate hypothesis-testing
dist_agent_1lm_randrot_noise_moderatehyp = copy.deepcopy(dist_agent_1lm_randrot_noise)
dist_agent_1lm_randrot_noise_moderatehyp[
    "logging_config"
].run_name = "dist_agent_1lm_randrot_noise_moderatehyp"
lm_config = dist_agent_1lm_randrot_noise_moderatehyp[
    "monty_config"
].learning_module_configs["learning_module_0"]
gsg_args = lm_config["learning_module_args"]["gsg_args"]
gsg_args["elapsed_steps_factor"] = 20
gsg_args["min_post_goal_success_steps"] = 10

CONFIGS = {
    "dist_agent_1lm_randrot_noise_nohyp": dist_agent_1lm_randrot_noise_nohyp,
    "dist_agent_1lm_randrot_noise_moderatehyp": dist_agent_1lm_randrot_noise_moderatehyp,
}
