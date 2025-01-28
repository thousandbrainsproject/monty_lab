# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 8: Multi-Modal Transfer

This module defines the following experiments:
 - `touch_agent_1lm_randrot_noise`
 - `dist_on_touch_1lm_randrot_noise`
 - `touch_on_dist_1lm_randrot_noise`

 Experiments use:
 - 77 objects
 - 5 random rotations
 - Sensor noise
 - No voting
"""

import copy

from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    SurfaceAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    SurfaceViewFinderMountHabitatDatasetArgs,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching import (
    MontyForEvidenceGraphMatching,
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

# `touch_agent_1lm`: a morphology-only model.
touch_agent_1lm = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(DMC_PRETRAIN_DIR / "touch_agent_1lm/pretrained"),
        n_eval_epochs=len(RANDOM_ROTATIONS_5),
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
    ),
    logging_config=DMCEvalLoggingConfig(run_name="touch_agent_1lm"),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyArgs(min_eval_steps=MIN_EVAL_STEPS),
        sensor_module_configs=dict(
            sensor_module_0=get_surf_patch_config(color=False),
            sensor_module_1=get_view_finder_config(),
        ),
        learning_module_configs=dict(
            learning_module_0=get_surf_lm_config(color=False),
        ),
        motor_system_config=get_surf_motor_config(),
    ),
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

# - Add noise and use 5 random rotations.
touch_agent_1lm_randrot_noise = make_randrot_noise_variant(touch_agent_1lm)


# `dist_on_touch_1lm_randrot_noise`: a distant agent using the touch agent's
# pretrained model.
dist_on_touch_1lm_randrot_noise = copy.deepcopy(dist_agent_1lm_randrot_noise)

# - Update the run name.
dist_on_touch_1lm_randrot_noise[
    "logging_config"
].run_name = "dist_on_touch_1lm_randrot_noise"

# - Set the model path to the touch agent's pretrained model.
dist_on_touch_1lm_randrot_noise["experiment_args"].model_name_or_path = str(
    DMC_PRETRAIN_DIR / "touch_agent_1lm/pretrained"
)
# - Tell the LM not to try and use the sensor's color data for graph matching
#   since the model has no color data stored.
lm_configs = dist_on_touch_1lm_randrot_noise["monty_config"].learning_module_configs
lm_args = lm_configs["learning_module_0"]["learning_module_args"]
lm_args["tolerances"]["patch"].pop("hsv")
lm_args["feature_weights"]["patch"].pop("hsv")


# `touch_on_dist_1lm_randrot_noise`: a touch agent using the distant agent's
# pretrained model.
touch_on_dist_1lm_randrot_noise = copy.deepcopy(touch_agent_1lm_randrot_noise)

# - Update the run name.
touch_on_dist_1lm_randrot_noise[
    "logging_config"
].run_name = "touch_on_dist_1lm_randrot_noise"

# - Set the model path to the distant agent's pretrained model.
touch_on_dist_1lm_randrot_noise["experiment_args"].model_name_or_path = str(
    DMC_PRETRAIN_DIR / "dist_agent_1lm/pretrained"
)


CONFIGS = {
    "touch_agent_1lm_randrot_noise": touch_agent_1lm_randrot_noise,
    "dist_on_touch_1lm_randrot_noise": dist_on_touch_1lm_randrot_noise,
    "touch_on_dist_1lm_randrot_noise": touch_on_dist_1lm_randrot_noise,
}
