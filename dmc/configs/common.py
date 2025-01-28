# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    MotorSystemConfigInformedGoalStateDriven,
    ParallelEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
)
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
)

# - Path Settings
DMC_ROOT_DIR = Path(os.environ.get("DMC_ROOT_DIR", "~/tbp/results/dmc")).expanduser()
DMC_PRETRAIN_DIR = DMC_ROOT_DIR / "pretrained_models"
DMC_RESULTS_DIR = DMC_ROOT_DIR / "results"

# - Common Parameters
MAX_TOTAL_STEPS = 10_000
MIN_EVAL_STEPS = 20
MAX_EVAL_STEPS = 500

# - 5 Random Rotations
RANDOM_ROTATIONS_5 = [
    [19, 339, 301],
    [196, 326, 225],
    [68, 100, 252],
    [256, 284, 218],
    [259, 193, 172],
]

"""
Custom classes
"""


@dataclass
class DMCEvalLoggingConfig(ParallelEvidenceLMLoggingConfig):
    """Logging config with DMC-specific output directory and wandb group.

    This config also drops the reproduce episode handler which is included
    as a default handler in `ParallelEvidenceLMLoggingConfig`.
    """

    output_dir: str = str(DMC_RESULTS_DIR)
    wandb_group: str = "dmc"
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
        ]
    )


"""
Config "Getter" Functions for Evaluation Experiments.
"""

def get_dist_lm_config(
    sensor_module_id: str = "patch",
    max_nneighbors: int = 5,
    color: bool = True,
) -> dict:
    """Get default distant evidence learning module config for evaluation.

    Args:
        sensor_module_id: ID of the sensor module this LM is associated with.
        max_nneighbors: Maximum number of neighbors to consider when matching features.
        color: Whether to include color (HSV) features in matching.

    Returns:
        dict: Learning module configuration with EvidenceGraphLM class and arguments
              including matching tolerances, feature weights, and goal state settings.
    """
    out = dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,  # =1cm
            tolerances={
                sensor_module_id: {
                    "hsv": np.array([0.1, 0.2, 0.2]),
                    "principal_curvatures_log": np.ones(2),
                }
            },
            feature_weights={
                sensor_module_id: {
                    "hsv": np.array([1, 0.5, 0.5]),
                }
            },
            # use_multithreading=False,
            # Most likely hypothesis needs to have 20% more evidence than the others
            # to be considered certain enough to trigger a terminal condition (match).
            x_percent_threshold=20,
            # look at 10 closest features stored in the search radius at most.
            max_nneighbors=max_nneighbors,
            # Update all hypotheses with evidence > x_percent_threshold (faster)
            evidence_update_threshold="x_percent_threshold",
            # NOTE: Currently not used when loading pretrained graphs.
            max_graph_size=0.3,  # 30cm
            num_model_voxels_per_dim=100,
            # Goal state generator which is used for model-based action suggestions.
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=dict(
                # Tolerance(s) when determining goal-state success
                goal_tolerances=dict(
                    location=0.015,  # distance in meters
                ),
                elapsed_steps_factor=10,
                # Number of necessary steps for a hypothesis goal-state to be considered
                min_post_goal_success_steps=5,
                desired_object_distance=0.03,
            ),
        ),
    )
    if not color:
        out["learning_module_args"]["tolerances"][sensor_module_id].pop("hsv")
        out["learning_module_args"]["feature_weights"][sensor_module_id].pop("hsv")

    return out


def get_surf_lm_config(
    sensor_module_id: str = "patch",
    max_nneighbors: int = 5,
    color: bool = True,
) -> dict:
    """Get default surface evidence learning module config.

    Args:
        sensor_module_id: ID of the sensor module this LM receives input from.
        max_nneighbors: Maximum number of neighbors to consider when matching features.
        color: Whether to include color (HSV) features.

    Returns:
        dict: Learning module config dictionary containing class and args.
    """
    out = dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,  # =1cm
            tolerances={
                sensor_module_id: {
                    "hsv": np.array([0.1, 0.2, 0.2]),
                    "principal_curvatures_log": np.ones(2),
                }
            },
            feature_weights={
                sensor_module_id: {
                    "hsv": np.array([1, 0.5, 0.5]),
                }
            },
            # Most likely hypothesis needs to have 20% more evidence than the others
            # to be considered certain enough to trigger a terminal condition (match).
            x_percent_threshold=20,
            # look at 10 closest features stored in the search radius at most.
            max_nneighbors=max_nneighbors,
            # Update all hypotheses with evidence > x_percent_threshold (faster)
            evidence_update_threshold="x_percent_threshold",
            # NOTE: Currently not used when loading pretrained graphs.
            max_graph_size=0.3,  # 30cm
            num_model_voxels_per_dim=100,
            # Goal state generator which is used for model-based action suggestions.
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=dict(
                # Tolerance(s) when determining goal-state success
                goal_tolerances=dict(
                    location=0.015,  # distance in meters
                ),
                elapsed_steps_factor=10,
                # Number of necessary steps for a hypothesis goal-state to be considered
                min_post_goal_success_steps=5,
                desired_object_distance=0.025,
            ),
        ),
    )
    if not color:
        out["learning_module_args"]["tolerances"][sensor_module_id].pop("hsv")
        out["learning_module_args"]["feature_weights"][sensor_module_id].pop("hsv")

    return out


def get_dist_patch_config(
    sensor_module_id: str = "patch",
    color: bool = True,
) -> dict:
    """Get default feature-change sensor module config for distant agent.

    Args:
        sensor_module_id (str, optional): ID for the sensor module. Defaults to "patch".
        color (bool, optional): Whether to include color features. Defaults to True.

    Returns:
        dict: Configuration dictionary containing:
            - sensor_module_class: The FeatureChangeSM class
            - sensor_module_args: Dict of arguments including features list,
              delta thresholds, and other sensor module settings
    """
    out = dict(
        sensor_module_class=FeatureChangeSM,
        sensor_module_args=dict(
            sensor_module_id=sensor_module_id,
            features=[
                # morphological features (necessarry)
                "pose_vectors",
                "pose_fully_defined",
                # non-morphological features (optional)
                "on_object",
                "principal_curvatures_log",
                "hsv",
            ],
            delta_thresholds={
                "on_object": 0,
                "distance": 0.01,
            },
            surf_agent_sm=False,
            save_raw_obs=False,
        ),
    )
    if not color:
        out["sensor_module_args"]["features"].remove("hsv")
    return out


def get_surf_patch_config(
    sensor_module_id: str = "patch",
    color: bool = True,
) -> dict:
    """Get default feature-change sensor module config for surface agent.

    Args:
        sensor_module_id (str, optional): ID for the sensor module. Defaults to "patch".
        color (bool, optional): Whether to include color features. Defaults to True.

    Returns:
        dict: Configuration dictionary containing:
            - sensor_module_class: The FeatureChangeSM class
            - sensor_module_args: Dict of arguments including features list,
              delta thresholds, and other sensor module settings
    """
    out = dict(
        sensor_module_class=FeatureChangeSM,
        sensor_module_args=dict(
            sensor_module_id=sensor_module_id,
            features=[
                # morphological features (necessarry)
                "pose_vectors",
                "pose_fully_defined",
                "on_object",
                # non-morphological features (optional)
                "object_coverage",
                "min_depth",
                "mean_depth",
                "principal_curvatures",
                "principal_curvatures_log",
                "hsv",
            ],
            delta_thresholds={
                "on_object": 0,
                "distance": 0.01,
            },
            surf_agent_sm=True,
            save_raw_obs=False,
        ),
    )
    if not color:
        out["sensor_module_args"]["features"].remove("hsv")

    return out


def get_view_finder_config() -> dict:
    """Get default view finder sensor module config for evaluation.

    The view finder sensor module is used to log detailed observations during
    evaluation. It uses the DetailedLoggingSM class with minimal configuration - just
    setting the sensor module ID and disabling raw observation saving.

    Returns:
        dict: Configuration dictionary containing:
            - sensor_module_class: The DetailedLoggingSM class
            - sensor_module_args: Dict with sensor_module_id and save_raw_obs settings
    """
    return dict(
        sensor_module_class=DetailedLoggingSM,
        sensor_module_args=dict(
            sensor_module_id="view_finder",
            save_raw_obs=False,
        ),
    )


def get_dist_motor_config() -> MotorSystemConfigInformedGoalStateDriven:
    """Get default distant motor config for evaluation.

    Returns:
        MotorSystemConfigInformedGoalStateDriven: Motor system configuration for
            distant agents that uses goal states to drive actions.
    """
    return MotorSystemConfigInformedGoalStateDriven()


def get_surf_motor_config() -> MotorSystemConfigCurInformedSurfaceGoalStateDriven:
    """Get default surface motor config for evaluation.

    Returns:
        MotorSystemConfigCurInformedSurfaceGoalStateDriven: Motor system configuration
            for surface agents that uses curvature-informed goal states to drive
            actions.
    """
    return MotorSystemConfigCurInformedSurfaceGoalStateDriven()


"""
Functions used for generating experiment variants.
--------------------------------------------------------------------------------
"""


def add_sensor_noise(
    config: dict,
    color: bool = True,
    pose_vectors: float = 2.0,
    hsv: float = 0.1,
    principal_curvatures_log: float = 0.1,
    pose_fully_defined: float = 0.01,
    location: float = 0.002,
) -> None:
    """Add default sensor noise to an experiment config in-place.

    Applies noise parameters to all sensor modules except the view finder. The
    `color` parameter controls whether to add 'hsv' noise. Set this to `False` for
    touch experiments and experiments using the pretrained touch model.

    Args:
        config: Experiment config to add sensor noise to.

    Returns:
        None: Modifies the input config in-place.
    """
    noise_params = {
        "pose_vectors": pose_vectors,
        "hsv": hsv,
        "principal_curvatures_log": principal_curvatures_log,
        "pose_fully_defined": pose_fully_defined,
        "location": location,
    }
    if not color:
        noise_params.pop("hsv")

    for sm_dict in config["monty_config"].sensor_module_configs.values():
        sm_args = sm_dict["sensor_module_args"]
        if sm_args["sensor_module_id"] == "view_finder":
            continue
        sm_args["noise_params"] = noise_params


def make_noise_variant(template: dict, color: bool = True) -> dict:
    """Create an experiment config with added sensor noise.

    Args:
        template: Experiment config to copy.

    Returns:
        dict: Copy of `template` with added sensor noise and with the
          "_noise" suffix appended to the logging config's `run_name`.

    Raises:
        ValueError: If experiment config does not have a run name.

    """
    config = copy.deepcopy(template)
    run_name = config["logging_config"].run_name
    if not run_name:
        raise ValueError("Experiment must have a run name to make a noisy version.")

    config["logging_config"].run_name = f"{run_name}_noise"
    add_sensor_noise(config, color=color)

    return config


def make_randrot_all_variant(template: dict) -> dict:
    """Create an config with a random object rotations.

    Args:
        template: Experiment config to copy.

    Returns:
        dict: Copy of `template` with a random rotation object initializer and the
          "_randrot" suffix appended to the logging config's `run_name`.

    Raises:
        ValueError: If experiment config does not have a run name.
    """
    config = copy.deepcopy(template)
    run_name = config["logging_config"].run_name
    if not run_name:
        raise ValueError(
            "Experiment must have a run name to make a random rotation version."
        )
    config["logging_config"].run_name = f"{run_name}_randrot_all"
    config[
        "eval_dataloader_args"
    ].object_init_sampler = RandomRotationObjectInitializer()

    return config


def make_randrot_variant(template: dict) -> dict:
    """Create an experiment config using the 5 predefined "random" rotations.

    Args:
        template: Experiment config to copy.

    Returns:
        dict: Copy of `template` with a PredefinedObjectInitializer set to
        use the 5 predefined rotations. Add the "_randrot" suffix to the
        logging config's `run_name`.
    Raises:
        ValueError: If experiment config does not have a run name.
    """
    config = copy.deepcopy(template)
    run_name = config["logging_config"].run_name
    if not run_name:
        raise ValueError(
            "Experiment must have a run name to make a random rotation version."
        )
    config["logging_config"].run_name = f"{run_name}_randrot"

    # Set eval dataloader args.
    config["eval_dataloader_args"].object_init_sampler = PredefinedObjectInitializer(
        rotations=RANDOM_ROTATIONS_5
    )

    # Update the number of epochs.
    config["experiment_args"].n_eval_epochs = len(RANDOM_ROTATIONS_5)

    return config


def make_randrot_noise_variant(template: dict, color: bool = True) -> dict:
    """Creates a variant of an experiment with both random rotations and sensor noise.

    Args:
        template: Dictionary containing experiment configuration.
        noise_params: Dictionary of noise parameters to add to sensor modules.
            Defaults to DEFAULT_NOISE_PARAMS.
        color: Whether to add noise to color features. Defaults to True.

    Returns:
        dict: Copy of `template` with sensor noise and a random rotation object
            initializer. The logging config's `run_name` has the original run name
            plus the suffix "_randrot_noise".
    """
    run_name = template["logging_config"].run_name
    config = make_randrot_variant(template)
    config = make_noise_variant(config, color=color)
    config["logging_config"].run_name = f"{run_name}_randrot_noise"

    return config


def make_randrot_all_noise_variant(template: dict, color: bool = True) -> dict:
    """Creates a variant of an experiment with both random rotations and sensor noise.

    Args:
        template: Dictionary containing experiment configuration.
        noise_params: Dictionary of noise parameters to add to sensor modules.
            Defaults to DEFAULT_NOISE_PARAMS.
        color: Whether to add noise to color features. Defaults to True.

    Returns:
        dict: Copy of `template` with sensor noise and a random rotation object
            initializer. The logging config's `run_name` has the original run name
            plus the suffix "_randrot_all_noise".
    """
    run_name = template["logging_config"].run_name
    config = make_randrot_all_variant(template)
    config = make_noise_variant(config, color=color)
    config["logging_config"].run_name = f"{run_name}_randrot_all_noise"

    return config