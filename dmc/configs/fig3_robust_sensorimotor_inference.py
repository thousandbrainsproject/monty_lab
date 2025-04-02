# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 3: Robust Sensorimotor Inference.

This module defines the following experiments:
 - `dist_agent_1lm`
 - `dist_agent_1lm_noise`
 - `dist_agent_1lm_randrot_all`
 - `dist_agent_1lm_randrot_all_noise`

 Experiments use:
 - 77 objects
 - 14 rotations
 - Goal-state-driven/hypothesis-testing policy active
 - A single LM (no voting)

NOTE: random rotation variants use the random object initializer and 14 rotations.
`dist_agent_1lm_randrot_noise` which uses the 5 predefined "random" rotations
is defined in `fig5_rapid_inference_with_voting.py`.
"""

from copy import deepcopy

from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    PatchAndViewMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMountHabitatDatasetArgs,
)

from .common import (
    DMC_PRETRAIN_DIR,
    MAX_EVAL_STEPS,
    MAX_TOTAL_STEPS,
    MIN_EVAL_STEPS,
    DMCEvalLoggingConfig,
    get_eval_lm_config,
    get_eval_motor_config,
    get_eval_patch_config,
    get_view_finder_config,
    make_noise_variant,
)

# - 14 Rotation used during training (cube faces + corners)
TEST_ROTATIONS = get_cube_face_and_corner_views_rotations()


dist_agent_1lm = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(DMC_PRETRAIN_DIR / "dist_agent_1lm/pretrained"),
        n_eval_epochs=len(TEST_ROTATIONS),
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
    ),
    logging_config=DMCEvalLoggingConfig(run_name="dist_agent_1lm"),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyArgs(min_eval_steps=MIN_EVAL_STEPS),
        sensor_module_configs=dict(
            sensor_module_0=get_eval_patch_config("dist"),
            sensor_module_1=get_view_finder_config(),
        ),
        learning_module_configs=dict(
            learning_module_0=get_eval_lm_config("dist"),
        ),
        motor_system_config=get_eval_motor_config("dist"),
    ),
    # Set up environment.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=TEST_ROTATIONS),
    ),
)

# Noisy/random rotation variants
# ------------------------------------------------------------------------------

# - Noisy sensor variant
dist_agent_1lm_noise = make_noise_variant(dist_agent_1lm)

# - Random rotation variant (14 random rotations)
dist_agent_1lm_randrot_all = deepcopy(dist_agent_1lm)
dist_agent_1lm_randrot_all["logging_config"].run_name = "dist_agent_1lm_randrot_all"
dist_agent_1lm_randrot_all[
    "eval_dataloader_args"
].object_init_sampler = RandomRotationObjectInitializer()

# - Random rotation variant (14 random rotations) and sensor noise
dist_agent_1lm_randrot_all_noise = make_noise_variant(
    dist_agent_1lm_randrot_all, run_name="dist_agent_1lm_randrot_all_noise"
)

CONFIGS = {
    "dist_agent_1lm": dist_agent_1lm,
    "dist_agent_1lm_noise": dist_agent_1lm_noise,
    "dist_agent_1lm_randrot_all": dist_agent_1lm_randrot_all,
    "dist_agent_1lm_randrot_all_noise": dist_agent_1lm_randrot_all_noise,
}
