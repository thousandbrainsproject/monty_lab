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
 - `surf_agent_1lm_randrot_noise`
 - `surf_agent_1lm_randrot_noise_nohyp`
 - `dist_agent_1lm_randrot_noise_nohyp`
 - `dist_agent_1lm_randrot_noise_moderatehyp` # still using?

 Experiments use:
 - 77 objects
 - Sensor noise and 5 random rotations
 - No voting
 - Varying levels of hypothesis-testing

"""

from copy import deepcopy

from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    SurfaceAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.simulators.habitat.configs import (
    SurfaceViewFinderMountHabitatDatasetArgs,
)

from .common import (
    DMC_PRETRAIN_DIR,
    MAX_EVAL_STEPS,
    MAX_TOTAL_STEPS,
    MIN_EVAL_STEPS,
    RANDOM_ROTATIONS_5,
    DMCEvalLoggingConfig,
    get_surf_lm_config,
    get_surf_motor_config,
    get_surf_patch_config,
    get_view_finder_config,
    make_randrot_noise_variant,
)
from .fig4_rapid_inference_with_voting import dist_agent_1lm_randrot_noise

# Surface agent
surf_agent_1lm = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(DMC_PRETRAIN_DIR / "surf_agent_1lm/pretrained"),
        n_eval_epochs=len(RANDOM_ROTATIONS_5),
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
    ),
    logging_config=DMCEvalLoggingConfig(run_name="surf_agent_1lm"),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyArgs(min_eval_steps=MIN_EVAL_STEPS),
        sensor_module_configs=dict(
            sensor_module_0=get_surf_patch_config(),
            sensor_module_1=get_view_finder_config(),
        ),
        learning_module_configs=dict(
            learning_module_0=get_surf_lm_config(),
        ),
        motor_system_config=get_surf_motor_config(),
    ),
    # Set up environment.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=RANDOM_ROTATIONS_5),
    ),
    # Configure dummy train dataloader. Required but not used.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=["mug"],
        object_init_sampler=PredefinedObjectInitializer(),
    ),
)

# Surface agent: Standard hypothesis-testing
surf_agent_1lm_randrot_noise = make_randrot_noise_variant(surf_agent_1lm)

# Surface agent: No hypothesis-testing
surf_agent_1lm_randrot_noise_nohyp = deepcopy(surf_agent_1lm_randrot_noise)
surf_agent_1lm_randrot_noise_nohyp[
    "logging_config"
].run_name = "surf_agent_1lm_randrot_noise_nohyp"
surf_agent_1lm_randrot_noise_nohyp[
    "monty_config"
].motor_system_config.motor_system_args.use_goal_state_driven_actions = False

# Distant agent: No hypothesis-testing
dist_agent_1lm_randrot_noise_nohyp = deepcopy(dist_agent_1lm_randrot_noise)
dist_agent_1lm_randrot_noise_nohyp[
    "logging_config"
].run_name = "dist_agent_1lm_randrot_noise_nohyp"
dist_agent_1lm_randrot_noise_nohyp[
    "monty_config"
].motor_system_config.motor_system_args.use_goal_state_driven_actions = False

# Distant agent: Moderate hypothesis-testing
dist_agent_1lm_randrot_noise_moderatehyp = deepcopy(dist_agent_1lm_randrot_noise)
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
    "surf_agent_1lm_randrot_noise": surf_agent_1lm_randrot_noise,
    "surf_agent_1lm_randrot_noise_nohyp": surf_agent_1lm_randrot_noise_nohyp,
    # "dist_agent_1lm_randrot_noise_moderatehyp": dist_agent_1lm_randrot_noise_moderatehyp,
}

