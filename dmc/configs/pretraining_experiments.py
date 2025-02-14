# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Supervised pretraining experiments.

This module defines a suite of supervised pretraining experiments. The core models
that experiments produce are:
 - `dist_agent_1lm`
 - `dist_agent_1lm_10distinctobj`
 - `surf_agent_1lm`
 - `surf_agent_1lm_10distinctobj`
 - `touch_agent_1lm`
 - `touch_agent_1lm_10distinctobj`
 - `dist_agent_2lm`
 - `dist_agent_4lm`
 - `dist_agent_8lm`
 - `dist_agent_16lm`

All of these models are trained on 77 YCB objects with 14 rotations each (cube face
and corners) except those with the `10distinctobj` suffix which are trained on the
10-distinct object dataset. The `touch` model is a surface agent without access to
color information.

This module performs some config finalization which does a few useful things:
 - Adds required (but unused) `eval_dataloader_class` and `eval_dataloader_args`.
 - Sets the logging config's `output_dir` to `PRETRAIN_DIR`.
 - Sets `do_eval` to `False` for all experiments.
 - Checks that no two configs would have the same output directory.

This module also defines a set of function that return default configs for learning
modules, sensor modules, and motor systems specific to pretraining experiments. They
may be useful for other modules that define pretraining experiments but should not
be used for eval experiments. Some functions take an optional `color` argument, where
setting it to `False` returns a sensor or learning module suitable for touch-only models.

The config 'getter'functions defined here are
 - `get_dist_lm_config`
 - `get_surf_lm_config`
 - `get_dist_patch_config`
 - `get_surf_patch_config`
 - `get_view_finder_config`
 - `get_dist_motor_config`
 - `get_surf_motor_config`

Names and logger args have follow certain rules for consistency and to help
avoid accidental conflicts:
 - Model names follow the pattern `{SENSOR}_agent_{NUM_LMS}lm`, where `SENSOR` is
   one of `dist`, `surf`, or `touch`.
 - The experiment key is `pretrain_{MODEL_NAME}` (e.g., `pretrain_dist_agent_1lm`). By
    'experiment key', I mean the key used to identify the config in `CONFIGS`.
 - The logging config's `run_name` is the model name.
 - The logging config's `output_dir` is `PRETRAIN_DIR`.

"""

import copy
from pathlib import Path

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
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS, SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatDistantPatchSM,
    HabitatSurfacePatchSM,
)
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
    make_multi_sensor_habitat_dataset_args,
)

from .common import DMC_PRETRAIN_DIR

# Specify default here
# - Experiment args
NUM_EXPLORATORY_STEPS_DIST = 500
NUM_EXPLORATORY_STEPS_SURF = 1000

# - Define 14 training rotations. Views from enclosing cube faces plus its corners.
TRAIN_ROTATIONS = get_cube_face_and_corner_views_rotations()


"""
Config "getter" functions
--------------------------------------------------------------------------------
"""


def get_dist_lm_config(
    sensor_module_id: str = "patch",
    color: bool = True,
) -> dict:
    """Get configuration for a displacement learning module.

    Convenience function that helps with sensor module IDs (particularly in
    a multi-sensor/LM configuration) and excluding color from graphs.

    Args:
        sensor_module_id (str): Identifier for the sensor module. Defaults to "patch".
        color (bool): Whether to include color-related features. Defaults to True.

    Returns:
        dict: Configuration dictionary for the displacement learning module.

    """
    out = dict(
        learning_module_class=DisplacementGraphLM,
        learning_module_args=dict(
            k=5,
            match_attribute="displacement",
            tolerance=np.ones(3) * 0.0001,
            graph_delta_thresholds={
                sensor_module_id: dict(
                    distance=0.001,  # 1 mm 0.01 on ycb v9
                    pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                    principal_curvatures_log=[1, 1],
                    hsv=[0.1, 1, 1],
                )
            },
        ),
    )
    if not color:
        out["learning_module_args"]["graph_delta_thresholds"][sensor_module_id].pop(
            "hsv"
        )
    return out


def get_surf_lm_config(
    sensor_module_id: str = "patch",
    color: bool = True,
) -> dict:
    """Get configuration for a displacement learning module.

    Convenience function that helps with sensor module IDs (particularly in
    a multi-sensor/LM configuration) and excluding color from graphs.

    Args:
        sensor_module_id (str): Identifier for the sensor module. Defaults to "patch".
        color (bool): Whether to include color-related features. Defaults to True.

    Returns:
        dict: Configuration dictionary for the displacement learning module.

    """
    out = dict(
        learning_module_class=DisplacementGraphLM,
        learning_module_args=dict(
            k=5,
            match_attribute="displacement",
            tolerance=np.ones(3) * 0.0001,
            graph_delta_thresholds={
                sensor_module_id: dict(
                    distance=0.01,
                    pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                    principal_curvatures_log=[1, 1],
                    hsv=[0.1, 1, 1],
                )
            },
        ),
    )
    if not color:
        out["learning_module_args"]["graph_delta_thresholds"][sensor_module_id].pop(
            "hsv"
        )
    return out


def get_dist_patch_config(sensor_module_id: str = "patch", color: bool = True) -> dict:
    """Get default distant patch config for pretraining.

    Provided as a convenience for handling sensor ID names and excluding
    color-related features from the config.

    Args:
        sensor_module_id (str): Identifier for the sensor module. Defaults to "patch".
        color (bool): Whether to include color features. Defaults to True.

    Returns:
        dict: Configuration dictionary for the surface patch sensor module.
    """
    out = dict(
        sensor_module_class=HabitatDistantPatchSM,
        sensor_module_args=dict(
            sensor_module_id=sensor_module_id,
            features=[
                # morphological features (necessarry)
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
                # non-morphological features (optional)
                "rgba",
                "hsv",
            ],
            save_raw_obs=True,
        ),
    )
    if not color:
        out["sensor_module_args"]["features"].remove("rgba")
        out["sensor_module_args"]["features"].remove("hsv")
    return out


def get_surf_patch_config(sensor_module_id: str = "patch", color: bool = True) -> dict:
    """Get default surface patch config for pretraining.

    Provided as a convenience for handling sensor ID names and excluding
    color-related features from the config.

    Args:
        sensor_module_id (str): Identifier for the sensor module. Defaults to "patch".
        color (bool): Whether to include color features. Defaults to True.

    Returns:
        dict: Configuration dictionary for the surface patch sensor module.
    """
    out = dict(
        sensor_module_class=HabitatSurfacePatchSM,
        sensor_module_args=dict(
            sensor_module_id=sensor_module_id,
            features=[
                # morphological features (necessarry)
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
                # non-morphological features (optional)
                "rgba",
                "hsv",
            ],
            save_raw_obs=True,
        ),
    )
    if not color:
        out["sensor_module_args"]["features"].remove("rgba")
        out["sensor_module_args"]["features"].remove("hsv")
    return out


def get_view_finder_config() -> dict:
    """Get default config for view finder.

    Returns:
        dict: Configuration dictionary for the view finder sensor module.

    """
    return dict(
        sensor_module_class=DetailedLoggingSM,
        sensor_module_args=dict(
            sensor_module_id="view_finder",
            save_raw_obs=True,
        ),
    )


def get_dist_motor_config(step_size: int = 5) -> MotorSystemConfigNaiveScanSpiral:
    """Get default distant motor config for pretraining.

    Returns:
        MotorSystemConfigNaiveScanSpiral: Configuration for the motor system for use
        with a distant agent.

    """
    return MotorSystemConfigNaiveScanSpiral(
        motor_system_args=make_naive_scan_policy_config(step_size=step_size)
    )


def get_surf_motor_config() -> MotorSystemConfigCurvatureInformedSurface:
    """Get default surface motor config for pretraining.

    Returns:
        MotorSystemConfigCurvatureInformedSurface: Configuration for the motor system
        for use with a surface agent.

    """
    return MotorSystemConfigCurvatureInformedSurface()


"""
Functions used for generating experiment variants.
--------------------------------------------------------------------------------
"""


def make_10distinctobj_variant(template: dict) -> dict:
    """Make a 10-distinct object variant of a config.

    The config returned is a copy of `template` with the following changes:
    - The `object_names` in the `train_dataloader_args` is set to `DISTINCT_OBJECTS`.
    - The logging config's `run_name` is appended with "_10distinctobj".

    Returns:
        dict: Copy of `template` config that trains on DISTINCT_OBJECTS dataset.
            The logging config's `run_name` is appended with "_10distinctobj".

    """
    config = copy.deepcopy(template)
    run_name = f"{config['logging_config'].run_name}_10distinctobj"
    config["logging_config"].run_name = run_name
    config["train_dataloader_args"].object_names = DISTINCT_OBJECTS
    return config


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
    logging_config=PretrainLoggingConfig(run_name="dist_agent_1lm"),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=NUM_EXPLORATORY_STEPS_DIST),
        learning_module_configs=dict(
            learning_module_0=get_dist_lm_config(),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_dist_patch_config(),
            sensor_module_1=get_view_finder_config(),
        ),
        motor_system_config=get_dist_motor_config(),
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
    logging_config=PretrainLoggingConfig(run_name="surf_agent_1lm"),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(
            num_exploratory_steps=NUM_EXPLORATORY_STEPS_SURF
        ),
        learning_module_configs=dict(
            learning_module_0=get_surf_lm_config(),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_surf_patch_config(),
            sensor_module_1=get_view_finder_config(),
        ),
        motor_system_config=get_surf_motor_config(),
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

pretrain_touch_agent_1lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(TRAIN_ROTATIONS),
        do_eval=False,
    ),
    logging_config=PretrainLoggingConfig(run_name="touch_agent_1lm"),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(
            num_exploratory_steps=NUM_EXPLORATORY_STEPS_SURF
        ),
        learning_module_configs=dict(
            learning_module_0=get_surf_lm_config(color=False),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_surf_patch_config(color=False),
            sensor_module_1=get_view_finder_config(),
        ),
        motor_system_config=get_surf_motor_config(),
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

# Make 10distinctobj variants
pretrain_dist_agent_1lm_10distinctobj = make_10distinctobj_variant(
    pretrain_dist_agent_1lm
)
pretrain_surf_agent_1lm_10distinctobj = make_10distinctobj_variant(
    pretrain_surf_agent_1lm
)
pretrain_touch_agent_1lm_10distinctobj = make_10distinctobj_variant(
    pretrain_touch_agent_1lm
)

"""
Setup for Multi-LM Experiments
--------------------------------------------------------------------------------
"""
# - Set up arguments for `make_multi_lm_monty_config`. Use the prefix `mlm_` to indicate
# that these are arguments for the multi-LM experiments.
# - The following set of arguments are reused for all multi-LM configs.
mlm_learning_module_config = get_dist_lm_config()
mlm_sensor_module_config = get_dist_patch_config()
mlm_motor_system_config = get_dist_motor_config()
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


"""
2 LMs
--------------------------------------------------------------------------------
"""

pretrain_dist_agent_2lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(TRAIN_ROTATIONS),
        do_eval=False,
    ),
    logging_config=PretrainLoggingConfig(run_name="dist_agent_2lm"),
    monty_config=make_multi_lm_monty_config(2, **mlm_monty_config_args),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_multi_sensor_habitat_dataset_args(2),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=TRAIN_ROTATIONS),
    ),
)

"""
4 LMs
--------------------------------------------------------------------------------
"""

pretrain_dist_agent_4lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(TRAIN_ROTATIONS),
        do_eval=False,
    ),
    logging_config=PretrainLoggingConfig(run_name="dist_agent_4lm"),
    monty_config=make_multi_lm_monty_config(4, **mlm_monty_config_args),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_multi_sensor_habitat_dataset_args(4),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=TRAIN_ROTATIONS),
    ),
)

"""
8 LMs
--------------------------------------------------------------------------------
"""

pretrain_dist_agent_8lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(TRAIN_ROTATIONS),
        do_eval=False,
    ),
    logging_config=PretrainLoggingConfig(run_name="dist_agent_8lm"),
    monty_config=make_multi_lm_monty_config(8, **mlm_monty_config_args),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_multi_sensor_habitat_dataset_args(8),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=TRAIN_ROTATIONS),
    ),
)

"""
16 LMs
--------------------------------------------------------------------------------
"""

pretrain_dist_agent_16lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(TRAIN_ROTATIONS),
    ),
    logging_config=PretrainLoggingConfig(run_name="dist_agent_16lm"),
    monty_config=make_multi_lm_monty_config(16, **mlm_monty_config_args),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_multi_sensor_habitat_dataset_args(16),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=TRAIN_ROTATIONS),
    ),
)


"""
Finalize configs
--------------------------------------------------------------------------------
"""

CONFIGS = {
    "pretrain_dist_agent_1lm": pretrain_dist_agent_1lm,
    "pretrain_surf_agent_1lm": pretrain_surf_agent_1lm,
    "pretrain_touch_agent_1lm": pretrain_touch_agent_1lm,
    "pretrain_dist_agent_1lm_10distinctobj": pretrain_dist_agent_1lm_10distinctobj,
    "pretrain_surf_agent_1lm_10distinctobj": pretrain_surf_agent_1lm_10distinctobj,
    "pretrain_touch_agent_1lm_10distinctobj": pretrain_touch_agent_1lm_10distinctobj,
    "pretrain_dist_agent_2lm": pretrain_dist_agent_2lm,
    "pretrain_dist_agent_4lm": pretrain_dist_agent_4lm,
    "pretrain_dist_agent_8lm": pretrain_dist_agent_8lm,
    "pretrain_dist_agent_16lm": pretrain_dist_agent_16lm,
}

# Perform sanity checks and
_output_paths = []
for exp in CONFIGS.values():
    # Add dummy eval dataloader. Required but not used.
    exp["eval_dataloader_class"] = ED.InformedEnvironmentDataLoader
    exp["eval_dataloader_args"] = EnvironmentDataloaderPerObjectArgs(
        object_names=["mug"],
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
    )
    # Configure output directory..
    exp["logging_config"].output_dir = str(DMC_PRETRAIN_DIR)

    # Make sure eval is disabled.
    exp["experiment_args"].do_eval = False

    # CHECK: output path must be unique.
    _path = Path(exp["logging_config"].output_dir) / exp["logging_config"].run_name
    assert _path not in _output_paths
    _output_paths.append(_path)

del _output_paths, _path
