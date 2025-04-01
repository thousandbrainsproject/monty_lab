# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Supervised pretraining experiments.

This module defines a suite of supervised pretraining experiments. The following is
a list of pretraining experiments and the models they produce:
 - `pretrain_dist_agent_1lm` -> `dist_agent_1lm`
 - `pretrain_surf_agent_1lm` -> `surf_agent_1lm`
 - `pretrain_dist_agent_2lm` -> `dist_agent_2lm`
 - `pretrain_dist_agent_4lm` -> `dist_agent_4lm`
 - `pretrain_dist_agent_8lm` -> `dist_agent_8lm`
 - `pretrain_dist_agent_16lm` -> `dist_agent_16lm`

All of these models are trained on 77 YCB objects with 14 rotations each (cube face
and corners).

This module also defines a set of function that return default configs for learning
modules, sensor modules, and motor systems specific to pretraining experiments. They
may be useful for other modules that define pretraining experiments but should not
be used for eval experiments. The config 'getter'functions defined here are
 - `get_pretrain_lm_config`
 - `get_pretrain_patch_config`
 - `get_pretrain_motor_config`

Names and logger args have follow certain rules for consistency and to help
avoid accidental conflicts:
 - Model names follow the pattern `{AGENT_TYPE}_agent_{NUM_LMS}lm`, where `AGENT_TYPE`
   is  `"dist"` or `"surf"`.
 - The experiment key is `pretrain_{MODEL_NAME}` (e.g., `pretrain_dist_agent_1lm`). By
    'experiment key', we mean the key used to run the experiment.

"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigCurvatureInformedSurface,
    MotorSystemConfigNaiveScanSpiral,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    SurfaceAndViewMontyConfig,
    get_cube_face_and_corner_views_rotations,
    make_multi_lm_monty_config,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.sensor_modules import (
    HabitatDistantPatchSM,
    HabitatSurfacePatchSM,
)
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
    make_multi_sensor_habitat_dataset_args,
)

from .common import DMC_PRETRAIN_DIR, get_view_finder_config

# Specify default here
# - Experiment args
NUM_EXPLORATORY_STEPS_DIST = 500
NUM_EXPLORATORY_STEPS_SURF = 1000

# - Define 14 training rotations. Views from enclosing cube faces plus its corners.
TRAIN_ROTATIONS = get_cube_face_and_corner_views_rotations()


@dataclass
class DMCPretrainLoggingConfig(PretrainLoggingConfig):
    output_dir: str = str(DMC_PRETRAIN_DIR)


"""
Config "getter" Functions for Pretraining Experiments
-----------------------------------------------------
"""


def get_pretrain_lm_config(agent_type: str) -> Dict[str, Any]:
    """Get a learning module config for pretraining experiments.

    This function returns a learning module config that uses default settings for
    pretraining experiments. For experiments with distant agents, use
    `agent_type="dist"`, and for experiments with surface agents, use `agent_type="surf"`.
    Settings are identical between "dist" and "surf" modes except for a
    distance threshold parameter that controls how close points in the learned
    graph can be. For distant agents, this is set to 0.001 meters; for surface
    agents, it is set to 0.0001 meters.

    Args:
        agent_type: The type of agent this LM will be connected to. Must be "dist"
          or "surf".

    Returns:
        A dictionary with two items:
          - "learning_module_class": The EvidenceGraphLM class.
          - "learning_module_args": A dictionary of arguments for the EvidenceGraphLM
            class.

    """
    if agent_type == "dist":
        graph_delta_distance_threshold = 0.001
    elif agent_type == "surf":
        graph_delta_distance_threshold = 0.0001
    else:
        raise ValueError(f"Invalid agent_type: {agent_type}. Must be 'dist' or 'surf'.")

    return dict(
        learning_module_class=DisplacementGraphLM,
        learning_module_args=dict(
            k=10,
            match_attribute="displacement",
            tolerance=np.ones(3) * 0.0001,
            graph_delta_thresholds={
                "patch": dict(
                    distance=graph_delta_distance_threshold,
                    pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                    principal_curvatures_log=[1, 1],
                    hsv=[0.1, 1, 1],
                )
            },
        ),
    )


def get_pretrain_patch_config(agent_type: str) -> Dict[str, Any]:
    """Get default distant patch config for pretraining.

    Provided as a convenience for handling sensor ID names and excluding
    color-related features from the config.

    Args:
        sensor_module_id (str): Identifier for the sensor module. Defaults to "patch".
        color (bool): Whether to include color features. Defaults to True.

    Returns:
        dict: Configuration dictionary for the surface patch sensor module.
    """

    if agent_type == "dist":
        return dict(
            sensor_module_class=HabitatDistantPatchSM,
            sensor_module_args=dict(
                sensor_module_id="patch",
                features=[
                    # morphological features
                    "pose_vectors",
                    "pose_fully_defined",
                    "on_object",
                    "principal_curvatures",
                    "principal_curvatures_log",
                    "gaussian_curvature",
                    "mean_curvature",
                    "gaussian_curvature_sc",
                    "mean_curvature_sc",
                    "object_coverage",
                    # non-morphological features
                    "rgba",
                    "hsv",
                ],
                save_raw_obs=False,
            ),
        )
    elif agent_type == "surf":
        return dict(
            sensor_module_class=HabitatSurfacePatchSM,
            sensor_module_args=dict(
                sensor_module_id="patch",
                features=[
                    # morphological features
                    "pose_vectors",
                    "pose_fully_defined",
                    "on_object",
                    "object_coverage",
                    "min_depth",
                    "mean_depth",
                    "principal_curvatures",
                    "principal_curvatures_log",
                    "gaussian_curvature",
                    "mean_curvature",
                    "gaussian_curvature_sc",
                    "mean_curvature_sc",
                    # non-morphological features
                    "rgba",
                    "hsv",
                ],
                save_raw_obs=False,
            ),
        )
    else:
        raise ValueError(f"Invalid agent_type: {agent_type}. Must be 'dist' or 'surf'.")


def get_pretrain_motor_config(agent_type: str) -> dataclass:
    """Get default distant motor config for pretraining.

    Returns:
        A dataclass with two attributes:
          - motor_system_class: The MotorSystemConfigNaiveScanSpiral class
            for distant agents, or the
            MotorSystemConfigCurvatureInformedSurface class for surface agents.
          - motor_system_args: A dictionary of arguments for the MotorSystemConfig
            class.

    """
    if agent_type == "dist":
        return MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=5)
        )
    elif agent_type == "surf":
        return MotorSystemConfigCurvatureInformedSurface()
    else:
        raise ValueError(f"Invalid agent_type: {agent_type}. Must be 'dist' or 'surf'.")


"""
1-LM models
--------------------------------------------------------------------------------
"""

pretrain_dist_agent_1lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(TRAIN_ROTATIONS),
        do_eval=False,
    ),
    logging_config=DMCPretrainLoggingConfig(run_name="dist_agent_1lm"),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=NUM_EXPLORATORY_STEPS_DIST),
        learning_module_configs=dict(
            learning_module_0=get_pretrain_lm_config("dist"),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_pretrain_patch_config("dist"),
            sensor_module_1=get_view_finder_config(),
        ),
        motor_system_config=get_pretrain_motor_config("dist"),
    ),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=TRAIN_ROTATIONS),
    ),
)

pretrain_surf_agent_1lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(TRAIN_ROTATIONS),
        do_eval=False,
    ),
    logging_config=DMCPretrainLoggingConfig(run_name="surf_agent_1lm"),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(
            num_exploratory_steps=NUM_EXPLORATORY_STEPS_SURF
        ),
        learning_module_configs=dict(
            learning_module_0=get_pretrain_lm_config("surf"),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_pretrain_patch_config("surf"),
            sensor_module_1=get_view_finder_config(),
        ),
        motor_system_config=get_pretrain_motor_config("surf"),
    ),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=TRAIN_ROTATIONS),
    ),
)

"""
Multi-LM Experiments
--------------------------------------------------------------------------------
"""
# - Set up arguments for `make_multi_lm_monty_config`. Use the prefix `mlm_` to indicate
# that these are arguments for the multi-LM experiments.
# - The following set of arguments are reused for all multi-LM configs.
mlm_learning_module_config = get_pretrain_lm_config("dist")
mlm_sensor_module_config = get_pretrain_patch_config("dist")
mlm_motor_system_config = get_pretrain_motor_config("dist")
mlm_monty_config_args = {
    "monty_class": MontyForGraphMatching,
    "learning_module_class": mlm_learning_module_config["learning_module_class"],
    "learning_module_args": mlm_learning_module_config["learning_module_args"],
    "sensor_module_class": mlm_sensor_module_config["sensor_module_class"],
    "sensor_module_args": mlm_sensor_module_config["sensor_module_args"],
    "motor_system_class": mlm_motor_system_config.motor_system_class,
    "motor_system_args": mlm_motor_system_config.motor_system_args,
    "monty_args": dict(num_exploratory_steps=NUM_EXPLORATORY_STEPS_DIST),
}

# - Template for multi-LM configs.
template_multi_lm_config = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(TRAIN_ROTATIONS),
        do_eval=False,
    ),
    dataset_class=ED.EnvironmentDataset,
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=TRAIN_ROTATIONS),
    ),
)

pretrain_dist_agent_2lm = deepcopy(template_multi_lm_config)
pretrain_dist_agent_2lm.update(
    dict(
        logging_config=DMCPretrainLoggingConfig(run_name="dist_agent_2lm"),
        monty_config=make_multi_lm_monty_config(2, **mlm_monty_config_args),
        dataset_args=make_multi_sensor_habitat_dataset_args(2),
    )
)

pretrain_dist_agent_4lm = deepcopy(template_multi_lm_config)
pretrain_dist_agent_4lm.update(
    dict(
        logging_config=DMCPretrainLoggingConfig(run_name="dist_agent_4lm"),
        monty_config=make_multi_lm_monty_config(4, **mlm_monty_config_args),
        dataset_args=make_multi_sensor_habitat_dataset_args(4),
    )
)

pretrain_dist_agent_8lm = deepcopy(template_multi_lm_config)
pretrain_dist_agent_8lm.update(
    dict(
        logging_config=DMCPretrainLoggingConfig(run_name="dist_agent_8lm"),
        monty_config=make_multi_lm_monty_config(8, **mlm_monty_config_args),
        dataset_args=make_multi_sensor_habitat_dataset_args(8),
    )
)

pretrain_dist_agent_16lm = deepcopy(template_multi_lm_config)
pretrain_dist_agent_16lm.update(
    dict(
        logging_config=DMCPretrainLoggingConfig(run_name="dist_agent_16lm"),
        monty_config=make_multi_lm_monty_config(16, **mlm_monty_config_args),
        dataset_args=make_multi_sensor_habitat_dataset_args(16),
    )
)


"""
Make configs discoverable
--------------------------------------------------------------------------------
"""

CONFIGS = {
    "pretrain_dist_agent_1lm": pretrain_dist_agent_1lm,
    "pretrain_surf_agent_1lm": pretrain_surf_agent_1lm,
    "pretrain_dist_agent_2lm": pretrain_dist_agent_2lm,
    "pretrain_dist_agent_4lm": pretrain_dist_agent_4lm,
    "pretrain_dist_agent_8lm": pretrain_dist_agent_8lm,
    "pretrain_dist_agent_16lm": pretrain_dist_agent_16lm,
}