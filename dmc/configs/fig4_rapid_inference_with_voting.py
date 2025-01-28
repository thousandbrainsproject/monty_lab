# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Configs for Figure 4: Rapid Inference with Voting.

This module defines the following single-LM config:
 - `dist_agent_1lm_randrot_noise`
 This is distinct from `dist_agent_1lm_randrot_all_noise` used in figure 3 in that it
 uses 5 random rotations instead of 14.

This module also defines defines the following multi-LM configs where half the number
of LMs must match before terminating an episode:
 - `dist_agent_2lm_half_lms_match_randrot_noise`
 - `dist_agent_4lm_half_lms_match_randrot_noise`
 - `dist_agent_8lm_half_lms_match_randrot_noise`
 - `dist_agent_16lm_half_lms_match_randrot_noise`

 In addition, there are variants where the minimum number of LMs that must match is
 fixed to 2:
 - `dist_agent_2lm_fixed_min_lms_match_randrot_noise`
 - `dist_agent_4lm_fixed_min_lms_match_randrot_noise`
 - `dist_agent_8lm_fixed_min_lms_match_randrot_noise`
 - `dist_agent_16lm_fixed_min_lms_match_randrot_noise`

All of these experiments use:
 - 77 objects
 - Goal-state-driven/hypothesis-testing policy active
 - Sensor noise and 5 (predefined) random rotations
 - Voting over 1, 2, 4, 8, or 16 LMs

"""

from copy import deepcopy

from tbp.monty.frameworks.config_utils.config_args import make_multi_lm_monty_config
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching import MontyForEvidenceGraphMatching
from tbp.monty.simulators.habitat.configs import (
    make_multi_sensor_habitat_dataset_args,
)

from .common import (
    DMC_PRETRAIN_DIR,
    MAX_EVAL_STEPS,
    MAX_TOTAL_STEPS,
    MIN_EVAL_STEPS,
    RANDOM_ROTATIONS_5,
    DMCEvalLoggingConfig,
    get_dist_lm_config,
    get_dist_motor_config,
    get_dist_patch_config,
    make_randrot_noise_variant,
)
from .fig3_robust_sensorimotor_inference import dist_agent_1lm

# - Set up arguments for `make_multi_lm_monty_config`. The following arguments
# are used for all multi-LM configs.
mlm_learning_module_config = get_dist_lm_config()
mlm_sensor_module_config = get_dist_patch_config()
mlm_motor_system_config = get_dist_motor_config()
mlm_monty_config_args = {
    "monty_class": MontyForEvidenceGraphMatching,
    "learning_module_class": mlm_learning_module_config["learning_module_class"],
    "learning_module_args": mlm_learning_module_config["learning_module_args"],
    "sensor_module_class": mlm_sensor_module_config["sensor_module_class"],
    "sensor_module_args": mlm_sensor_module_config["sensor_module_args"],
    "motor_system_class": mlm_motor_system_config.motor_system_class,
    "motor_system_args": mlm_motor_system_config.motor_system_args,
    "monty_args": dict(min_eval_steps=MIN_EVAL_STEPS),
}

"""
Create the Multi-LM configs. Note that we are setting the predefined rotations
in these configs, but we will use `make_randrot_noise_variant` anyway to handle
setting run names the way we want.
"""

dist_agent_2lm_half_lms_match = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(DMC_PRETRAIN_DIR / "dist_agent_2lm/pretrained"),
        n_eval_epochs=len(RANDOM_ROTATIONS_5),
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
        min_lms_match=1,
    ),
    logging_config=DMCEvalLoggingConfig(run_name="dist_agent_2lm_half_lms_match"),
    monty_config=make_multi_lm_monty_config(2, **mlm_monty_config_args),
    # Set up environment.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_multi_sensor_habitat_dataset_args(2),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=RANDOM_ROTATIONS_5),
    ),
    # Configure dummy train dataloader. Required but not used.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=["mug"],
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
    ),
)

dist_agent_4lm_half_lms_match = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(DMC_PRETRAIN_DIR / "dist_agent_4lm/pretrained"),
        n_eval_epochs=len(RANDOM_ROTATIONS_5),
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
        min_lms_match=2,
    ),
    logging_config=DMCEvalLoggingConfig(run_name="dist_agent_4lm_half_lms_match"),
    monty_config=make_multi_lm_monty_config(4, **mlm_monty_config_args),
    # Set up environment.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_multi_sensor_habitat_dataset_args(4),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=RANDOM_ROTATIONS_5),
    ),
    # Configure dummy train dataloader. Required but not used.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=["mug"],
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
    ),
)

dist_agent_8lm_half_lms_match = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(DMC_PRETRAIN_DIR / "dist_agent_8lm/pretrained"),
        n_eval_epochs=len(RANDOM_ROTATIONS_5),
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
        min_lms_match=4,
    ),
    logging_config=DMCEvalLoggingConfig(run_name="dist_agent_8lm_half_lms_match"),
    monty_config=make_multi_lm_monty_config(8, **mlm_monty_config_args),
    # Set up environment.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_multi_sensor_habitat_dataset_args(8),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=RANDOM_ROTATIONS_5),
    ),
    # Configure dummy train dataloader. Required but not used.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=["mug"],
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
    ),
)

dist_agent_16lm_half_lms_match = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(DMC_PRETRAIN_DIR / "dist_agent_16lm/pretrained"),
        n_eval_epochs=len(RANDOM_ROTATIONS_5),
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
        min_lms_match=8,
    ),
    logging_config=DMCEvalLoggingConfig(run_name="dist_agent_16lm_half_lms_match"),
    monty_config=make_multi_lm_monty_config(16, **mlm_monty_config_args),
    # Set up environment.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_multi_sensor_habitat_dataset_args(16),
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

# ==== The single-LM config ====
dist_agent_1lm_randrot_noise = make_randrot_noise_variant(dist_agent_1lm)

# ==== Variants where min_lms_match=int(num_lms/2) ====

# - 2 LMs
dist_agent_2lm_half_lms_match_randrot_noise = make_randrot_noise_variant(
    dist_agent_2lm_half_lms_match
)

# - 4 LMs
dist_agent_4lm_half_lms_match_randrot_noise = make_randrot_noise_variant(
    dist_agent_4lm_half_lms_match
)

# - 8 LMs
dist_agent_8lm_half_lms_match_randrot_noise = make_randrot_noise_variant(
    dist_agent_8lm_half_lms_match
)

# - 16 LMs
dist_agent_16lm_half_lms_match_randrot_noise = make_randrot_noise_variant(
    dist_agent_16lm_half_lms_match
)

# ==== Variants where min_lms_match=min(num_lms, 2) ====

# - 2 LMs
dist_agent_2lm_fixed_min_lms_match_randrot_noise = deepcopy(
    dist_agent_2lm_half_lms_match_randrot_noise
)
dist_agent_2lm_fixed_min_lms_match_randrot_noise["experiment_args"].min_lms_match = 2
dist_agent_2lm_fixed_min_lms_match_randrot_noise["logging_config"].run_name = (
    "dist_agent_2lm_fixed_min_lms_match_randrot_noise"
)

# - 4 LMs
dist_agent_4lm_fixed_min_lms_match_randrot_noise = deepcopy(
    dist_agent_4lm_half_lms_match_randrot_noise
)
dist_agent_4lm_fixed_min_lms_match_randrot_noise["experiment_args"].min_lms_match = 2
dist_agent_4lm_fixed_min_lms_match_randrot_noise["logging_config"].run_name = (
    "dist_agent_4lm_fixed_min_lms_match_randrot_noise"
)

# - 8 LMs
dist_agent_8lm_fixed_min_lms_match_randrot_noise = deepcopy(
    dist_agent_8lm_half_lms_match_randrot_noise
)
dist_agent_8lm_fixed_min_lms_match_randrot_noise["experiment_args"].min_lms_match = 2
dist_agent_8lm_fixed_min_lms_match_randrot_noise["logging_config"].run_name = (
    "dist_agent_8lm_fixed_min_lms_match_randrot_noise"
)

# - 16 LMs
dist_agent_16lm_fixed_min_lms_match_randrot_noise = deepcopy(
    dist_agent_16lm_half_lms_match_randrot_noise
)
dist_agent_16lm_fixed_min_lms_match_randrot_noise["experiment_args"].min_lms_match = 2
dist_agent_16lm_fixed_min_lms_match_randrot_noise["logging_config"].run_name = (
    "dist_agent_16lm_fixed_min_lms_match_randrot_noise"
)


CONFIGS = {
    "dist_agent_1lm_randrot_noise": dist_agent_1lm_randrot_noise,
    "dist_agent_2lm_half_lms_match_randrot_noise": dist_agent_2lm_half_lms_match_randrot_noise,
    "dist_agent_4lm_half_lms_match_randrot_noise": dist_agent_4lm_half_lms_match_randrot_noise,
    "dist_agent_8lm_half_lms_match_randrot_noise": dist_agent_8lm_half_lms_match_randrot_noise,
    "dist_agent_16lm_half_lms_match_randrot_noise": dist_agent_16lm_half_lms_match_randrot_noise,
    "dist_agent_2lm_fixed_min_lms_match_randrot_noise": dist_agent_2lm_fixed_min_lms_match_randrot_noise,
    "dist_agent_4lm_fixed_min_lms_match_randrot_noise": dist_agent_4lm_fixed_min_lms_match_randrot_noise,
    "dist_agent_8lm_fixed_min_lms_match_randrot_noise": dist_agent_8lm_fixed_min_lms_match_randrot_noise,
    "dist_agent_16lm_fixed_min_lms_match_randrot_noise": dist_agent_16lm_fixed_min_lms_match_randrot_noise,
}
