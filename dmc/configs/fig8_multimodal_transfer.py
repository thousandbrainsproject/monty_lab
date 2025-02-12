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
 - `dist_agent_1lm_randrot_noise_10distinctobj`
 - `touch_agent_1lm_randrot_noise_10distinctobj`
 - `dist_on_touch_1lm_randrot_noise_10distinctobj`
 - `touch_on_dist_1lm_randrot_noise_10distinctobj`

 Experiments use:
 - Test on 10 distinct objects, trained on 10 distinct objects.
 - 5 random rotations
 - Sensor noise
 - No voting
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
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
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

# `touch_agent_1lm_10distinctobj`: a morphology-only model.
touch_agent_1lm_10distinctobj = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(
            DMC_PRETRAIN_DIR / "touch_agent_1lm_10distinctobj/pretrained"
        ),
        n_eval_epochs=len(RANDOM_ROTATIONS_5),
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
    ),
    logging_config=DMCEvalLoggingConfig(run_name="touch_agent_1lm_10distinctobj"),
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
        object_names=DISTINCT_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=RANDOM_ROTATIONS_5),
    ),
    # Configure dummy train dataloader. Required but not used.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=["mug"],
        object_init_sampler=PredefinedObjectInitializer(),
    ),
)

# Distant agent
dist_agent_1lm_randrot_noise_10distinctobj = deepcopy(dist_agent_1lm_randrot_noise)
dist_agent_1lm_randrot_noise_10distinctobj[
    "logging_config"
].run_name = "dist_agent_1lm_randrot_noise_10distinctobj"
dist_agent_1lm_randrot_noise_10distinctobj["experiment_args"].model_name_or_path = str(
    DMC_PRETRAIN_DIR / "dist_agent_1lm_10distinctobj/pretrained"
)
dist_agent_1lm_randrot_noise_10distinctobj[
    "eval_dataloader_args"
].object_names = DISTINCT_OBJECTS


# Touch agent
touch_agent_1lm_randrot_noise_10distinctobj = make_randrot_noise_variant(
    touch_agent_1lm_10distinctobj
)
touch_agent_1lm_randrot_noise_10distinctobj[
    "logging_config"
].run_name = "touch_agent_1lm_randrot_noise_10distinctobj"


# Distant agent w/ touch pretrained model
dist_on_touch_1lm_randrot_noise_10distinctobj = deepcopy(
    dist_agent_1lm_randrot_noise_10distinctobj
)
dist_on_touch_1lm_randrot_noise_10distinctobj[
    "logging_config"
].run_name = "dist_on_touch_1lm_randrot_noise_10distinctobj"
dist_on_touch_1lm_randrot_noise_10distinctobj[
    "experiment_args"
].model_name_or_path = str(
    DMC_PRETRAIN_DIR / "touch_agent_1lm_10distinctobj/pretrained"
)
# - Tell the LM not to use the sensor's color data for graph matching
#   since the model has no color data stored.
lm_configs = dist_on_touch_1lm_randrot_noise_10distinctobj[
    "monty_config"
].learning_module_configs
lm_args = lm_configs["learning_module_0"]["learning_module_args"]
lm_args["tolerances"]["patch"].pop("hsv")
lm_args["feature_weights"]["patch"].pop("hsv")


# Touch agent w/ distant agent pretrained model
touch_on_dist_1lm_randrot_noise_10distinctobj = deepcopy(
    touch_agent_1lm_randrot_noise_10distinctobj
)
touch_on_dist_1lm_randrot_noise_10distinctobj[
    "logging_config"
].run_name = "touch_on_dist_1lm_randrot_noise_10distinctobj"
touch_on_dist_1lm_randrot_noise_10distinctobj[
    "experiment_args"
].model_name_or_path = str(DMC_PRETRAIN_DIR / "dist_agent_1lm_10distinctobj/pretrained")


CONFIGS = {
    "dist_agent_1lm_randrot_noise_10distinctobj": dist_agent_1lm_randrot_noise_10distinctobj,
    "touch_agent_1lm_randrot_noise_10distinctobj": touch_agent_1lm_randrot_noise_10distinctobj,
    "dist_on_touch_1lm_randrot_noise_10distinctobj": dist_on_touch_1lm_randrot_noise_10distinctobj,
    "touch_on_dist_1lm_randrot_noise_10distinctobj": touch_on_dist_1lm_randrot_noise_10distinctobj,
}
