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
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    EvalEvidenceLMLoggingConfig,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    MotorSystemConfigInformedGoalStateDriven,
    ParallelEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
    ReproduceEpisodeHandler,
)
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
)
from tbp.monty.frameworks.utils.logging_utils import maybe_rename_existing_file

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


"""
Config "Getter" Functions for Evaluation Experiments.
"""

def get_dist_lm_config(
    sensor_module_id: str = "patch",
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
            # Update all hypotheses with evidence > 80% of max evidence (faster)
            evidence_update_threshold="80%",
            # Look at 5 closest features stored in the search radius at most.
            max_nneighbors=5,
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
            # look at 5 closest features stored in the search radius at most.
            max_nneighbors=5,
            # Update all hypotheses with evidence > 80% of max evidence (faster)
            evidence_update_threshold="80%",
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


"""
Logging
--------------------------------------------------------------------------------
"""


class SelectiveEvidenceHandler(DetailedJSONHandler):
    """Detailed Logger that only saves evidence LM data and limited sensor data.

    Saves the following LM data:
     - current_mlh
     - evidences
     - lm_processed_steps
     - possible_locations
     - possible_rotations
     - possible_matches
     - symmetry_evidence
     - symmetric_locations
     - symmetric_rotations

    For sensor modules, only data is saved for steps where an LM has processed data.

    This class extends `DetailedJSONHandler` by breaking up the logic of
    `report_episode` into two parts:
     - `init_buffer_data`: Initialize the buffer data dict.
     - `save`: Save the buffer data to a file.

    This is intended to make it easier for subclasses to modify the data saved
    by overriding `init_buffer_data` or dropping buffer data after its initialized
    during `report_episode`.

    This class also can take a `selective_handler_args` which can be used to exclude
    certain items from the stored data. For example,
    ```
    selector_handler_args = {"exclude": ["SM_0", "SM_1"]}
    ```
    will exclude data for `SM_0` and `SM_1` entirely. Supply `selective_handler_args`
    by setting the `selective_handler_args` attribute in a logging config.
    """

    def __init__(self, selective_handler_args: Optional[Mapping] = None):
        super().__init__()
        self.handler_args = (
            dict(selective_handler_args) if selective_handler_args else {}
        )

    def report_episode(
        self,
        data: Mapping,
        output_dir: str,
        episode: int,
        mode: str = "train",
        **kwargs,
    ):
        """Report episode data.

        Args:
            data (dict): Data to report. Contains keys "BASIC" and "DETAILED".
            output_dir (str): Directory to save the report.
            episode (int): Episode number within the epoch.
            mode (str): Either "train" or "eval".
            **kwargs: Additional keyword arguments.

        Changed name to report episode since we are currently running with
        reporting and flushing exactly once per episode.
        """
        # Initialize buffer data, using only certain LM data and only sensor data
        # for steps where an LM has processed data.
        episode_total, buffer_data = self.init_buffer_data(
            data, episode, mode, **kwargs
        )

        # Save data.
        self.save(episode_total, buffer_data, output_dir)

    def save(self, episode_total: int, buffer_data: Mapping, output_dir: str) -> None:
        """Save data to a file.

        Args:
            data (Mapping): Data to save.
            output_dir (str): Directory to save the data.
        """
        save_stats_path = os.path.join(output_dir, "detailed_run_stats.json")
        maybe_rename_existing_file(save_stats_path, ".json", self.report_count)
        with open(save_stats_path, "a") as f:
            json.dump({episode_total: buffer_data}, f, cls=BufferEncoder)
            f.write(os.linesep)

        print("Stats appended to " + save_stats_path)
        self.report_count += 1

    def init_buffer_data(
        self,
        data: Mapping,
        episode: int,
        mode: str,
        **kwargs,
    ) -> Tuple[int, Mapping]:
        """Initialize the output data dict.

        Args:
            data (Mapping): Data from the episode.
            episode (int): Episode number.
            mode (str): Either "train" or "eval".

        Returns:
            Tuple[int, Mapping]: The episode number and the data to save.
        """

        # Get basic and detailed data.
        if mode == "train":
            episode_total = kwargs["train_episodes_to_total"][episode]
        elif mode == "eval":
            episode_total = kwargs["eval_episodes_to_total"][episode]
        detailed = data["DETAILED"][episode_total]

        buffer_data = dict()

        # Add LM data.
        lm_ids = [key for key in detailed if key.startswith("LM")]
        for lm_id in lm_ids:
            lm_dict = {
                "current_mlh": detailed[lm_id]["current_mlh"],
                "evidences": detailed[lm_id]["evidences"],
                "lm_processed_steps": detailed[lm_id]["lm_processed_steps"],
                "possible_locations": detailed[lm_id]["possible_locations"],
                "possible_rotations": detailed[lm_id]["possible_rotations"],
                "possible_matches": detailed[lm_id]["possible_matches"],
                "symmetry_evidence": detailed[lm_id]["symmetry_evidence"],
                "symmetric_locations": detailed[lm_id]["symmetric_locations"],
                "symmetric_rotations": detailed[lm_id]["symmetric_rotations"],
            }
            buffer_data[lm_id] = lm_dict

        # Add SM data, but only where LMs have processed data.
        sm_ids = [key for key in detailed if key.startswith("SM")]
        lm_processed_steps = self.find_lm_processed_steps(detailed)
        for sm_id in sm_ids:
            sm_dict = dict()
            for name in [
                "raw_observations",
                "processed_observations",
                "sm_properties",
            ]:
                if name in detailed[sm_id]:
                    lst = [detailed[sm_id][name][step] for step in lm_processed_steps]
                    sm_dict[name] = lst
            buffer_data[sm_id] = sm_dict

        # Handle excludes.
        exclude = self.handler_args.get("exclude", [])
        for key in exclude:
            buffer_data.pop(key, None)

        # Finalize output data.
        return episode_total, buffer_data

    def find_lm_processed_steps(self, detailed: Mapping) -> np.ndarray:
        """Find steps where any LM has processed data.

        Args:
            detailed (Mapping): Data from a single episode.

        Returns:
            np.ndarray: Int array of indices where at least one LM processed data.
        """
        lm_ids = [key for key in detailed if key.startswith("LM")]
        if len(lm_ids) == 1:
            return np.argwhere(detailed[lm_ids[0]]["lm_processed_steps"]).squeeze()

        n_monty_steps = len(detailed[lm_ids[0]]["lm_processed_steps"])
        lm_processed_steps = np.zeros(n_monty_steps, dtype=bool)
        for step in range(n_monty_steps):
            processed = [detailed[key]["lm_processed_steps"][step] for key in lm_ids]
            lm_processed_steps[step] = any(processed)
        return np.argwhere(lm_processed_steps).squeeze()

    def close(self):
        pass


@dataclass
class SelectiveEvidenceLoggingConfig(EvalEvidenceLMLoggingConfig):
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            SelectiveEvidenceHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_handlers: list = field(default_factory=list)
    monty_log_level: str = "DETAILED"
    selective_handler_args: dict = field(default_factory=dict)


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