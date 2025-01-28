# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 9: Structured Object Representations

This module defines the following experiments:
 - `dist_agent_1lm_randrot_noise_10simobj`

 Experiments use:
 - 10 similar objects (but using the 77-object pretrained model)
 - 5 random rotations
 - Sensor noise
 - Hypothesis-testing policy active
 - No voting
 - SELECTIVE evidence logging
"""

import copy

from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
    EvalEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.environments.ycb import SIMILAR_OBJECTS
from tbp.monty.frameworks.loggers.monty_handlers import DetailedJSONHandler

from .common import DMC_RESULTS_DIR
from .fig4_rapid_inference_with_voting import dist_agent_1lm_randrot_noise

dist_agent_1lm_randrot_noise_10simobj = copy.deepcopy(dist_agent_1lm_randrot_noise)
dist_agent_1lm_randrot_noise_10simobj["logging_config"] = EvalEvidenceLMLoggingConfig(
    output_dir=str(DMC_RESULTS_DIR),
    run_name="dist_agent_1lm_randrot_noise_10simobj",
    wandb_group="dmc",
    monty_log_level="SELECTIVE",
)
# dist_agent_1lm_randrot_noise_10simobj["logging_config"].monty_handlers.append(
# DetailedJSONHandler,
# )
dist_agent_1lm_randrot_noise_10simobj[
    "eval_dataloader_args"
].object_names = SIMILAR_OBJECTS

CONFIGS = {
    "dist_agent_1lm_randrot_noise_10simobj": dist_agent_1lm_randrot_noise_10simobj,
}
