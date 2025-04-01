# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    EvalEvidenceLMLoggingConfig,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    MotorSystemConfigInformedGoalStateDriven,
    ParallelEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
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

# - Evaluation Parameters
MAX_TOTAL_STEPS = 10_000
MIN_EVAL_STEPS = 20
MAX_EVAL_STEPS = 500

# - 5 Predefined Random Rotations
RANDOM_ROTATIONS_5 = [
    [19, 339, 301],
    [196, 326, 225],
    [68, 100, 252],
    [256, 284, 218],
    [259, 193, 172],
]


"""
Config Functions for Creating/Modifying Evaluation Experiments Configs
----------------------------------------------------------------------
Note: See `pretraining_experiments.py` for functions that return configs
suitable for pretraining experiments.

"""


def get_eval_lm_config(agent_type: str) -> Dict[str, Any]:
    """Create a learning module config for evaluation experiments.

    This function returns a learning module config that uses default settings for
    evaluation experiments. For experiments with distant agents, use
    `agent_type="dist"`; for experiments with surface agents, use `agent_type="surf"`.

    Settings are identical between "dist" and "surf" modes except
    `desired_object_distance` -- a parameter of the goal state generator that
    determines how far the agent should be from a point on the object's surface
    after performing a hypothesis-driven jump. For distant agents, this is set to
    0.03 meters; for surface agents, it is set to 0.025 meters.

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
        desired_object_distance = 0.03
    elif agent_type == "surf":
        desired_object_distance = 0.025
    else:
        raise ValueError(f"Invalid agent_type: {agent_type}. Must be 'dist' or 'surf'.")

    return dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            # Specify graph matching thresholds and tolerances.
            max_match_distance=0.01,  # 1 cm
            tolerances={
                "patch": {
                    "hsv": np.array([0.1, 0.2, 0.2]),
                    "principal_curvatures_log": np.ones(2),
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0.5, 0.5]),
                }
            },
            # Update all hypotheses with evidence > 80% of max evidence.
            evidence_update_threshold="80%",
            x_percent_threshold=20,
            # Look at `n` closest points stored in the search radius (at most).
            max_nneighbors=10,
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
                # How for the agent should from a point on the object's surface after
                # performing a hypothesis-driven jump.
                desired_object_distance=desired_object_distance,
            ),
        ),
    )


def get_eval_patch_config(agent_type: str) -> Dict[str, Any]:
    """Create a sensor module config for evaluation experiments.

    This function returns a sensor module config that uses default settings for
    evaluation experiments. For experiments with distant agents, use
    `agent_type="dist"`; for experiments with surface agents, use `agent_type="surf"`.
    Both modes use a `FeatureChangeSM` with identical settings.

    Args:
        agent_type: The type of agent this SM will be connected to. Must be "dist"
          or "surf".

    Returns:
        A dictionary with two items:
          - "sensor_module_class": The FeatureChangeSM class.
          - "sensor_module_args": A dictionary of arguments for the FeatureChangeSM
            class.
    """
    if agent_type == "dist":
        surf_agent_sm = False
    elif agent_type == "surf":
        surf_agent_sm = True
    else:
        raise ValueError(f"Invalid agent_type: {agent_type}. Must be 'dist' or 'surf'.")

    return dict(
        sensor_module_class=FeatureChangeSM,
        sensor_module_args=dict(
            sensor_module_id="patch",
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
            surf_agent_sm=surf_agent_sm,
            save_raw_obs=False,
        ),
    )


def get_view_finder_config() -> Dict[str, Any]:
    """Create a sensor module config for a view-finder.

    A view finder is used to move the agent onto the object at the start of an episode,
    or after a goal-state driven jump. It is also used to log observations for
    post-hoc visualization. The returned config is suitable for both pretraining and
    evaluation experiments.

    Returns:
        A dictionary with two items:
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


def get_eval_motor_config(agent_type: str) -> dataclass:
    """Create a motor system config for evaluation experiments.

    This function returns a motor system config that uses default settings for
    evaluation experiments. For experiments with distant agents, use
    `agent_type="dist"`, and for experiments with surface agents, use
    `agent_type="surf"`.

    Args:
        agent_type: The type of agent this motor system will be connected to. Must be
          "dist" or "surf".

    Returns:
        A dataclass with two attributes:
          - motor_system_class: The MotorSystemConfigInformedGoalStateDriven class
            for distant agents, or the
            MotorSystemConfigCurInformedSurfaceGoalStateDriven class for surface agents.
          - motor_system_args: A dictionary of arguments for the MotorSystemConfig
            class.
    """
    if agent_type == "dist":
        return MotorSystemConfigInformedGoalStateDriven()
    elif agent_type == "surf":
        return MotorSystemConfigCurInformedSurfaceGoalStateDriven()
    else:
        raise ValueError(f"Invalid agent_type: {agent_type}. Must be 'dist' or 'surf'.")


"""
Functions for generating variations of existing configs.
--------------------------------------------------------
"""


def add_sensor_noise(config: Dict[str, Any]) -> None:
    """Add sensor noise to an experiment config. Modifies the config in-place.

    Applies noise parameters to all sensor modules except the view finder.

    Args:
        config: Experiment config to add sensor noise to.
    """
    noise_params = {
        "pose_vectors": 2.0,
        "hsv": 0.1,
        "principal_curvatures_log": 0.1,
        "pose_fully_defined": 0.01,
        "location": 0.002,
    }

    for sm_dict in config["monty_config"].sensor_module_configs.values():
        sm_args = sm_dict["sensor_module_args"]
        if sm_args["sensor_module_id"] == "view_finder":
            continue
        sm_args["noise_params"] = noise_params


def make_noise_variant(
    template: Dict[str, Any],
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a copy of an experiment config with added sensor noise.

    Args:
        template: Experiment config to copy.
        run_name: Name of the new experiment or `None`. By default, this
          function will add "_noise" to the template config's run name (if possible),
          but the run name can be specified directly via this parameter.

    Returns:
        An experiment config where sensors have noise parameters.
    """
    config = deepcopy(template)

    # Optionally, use the provided run name. Otherwise, append "_noise" to the
    # existing run name (if one exists).
    if run_name:
        config["logging_config"].run_name = run_name
    else:
        template_name = config["logging_config"].run_name
        if template_name:
            config["logging_config"].run_name = f"{template_name}_noise"

    # Add sensor noise. Modifies `config` in-place.
    add_sensor_noise(config)

    return config


def make_randrot_variant(
    template: Dict[str, Any],
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a copy of an experiment config that uses the 5 predefined random rotations.

    Args:
        template: Experiment config to copy.
        run_name: Name of the new experiment or `None`. By default, this
          function will add "_randrot" to the template config's run name (if possible),
          but the run name can be specified directly via this parameter.

    Returns:
        An experiment config that uses the 5 predefined random rotations.
    """

    config = deepcopy(template)

    # Optionally, use the provided run name. Otherwise, append "_randrot" to the
    # existing run name (if one exists).
    if run_name:
        config["logging_config"].run_name = run_name
    else:
        template_name = config["logging_config"].run_name
        if template_name:
            config["logging_config"].run_name = f"{template_name}_randrot"

    # Set up with 5 "random" rotations, and update the number of epochs.
    config["eval_dataloader_args"].object_init_sampler = PredefinedObjectInitializer(
        rotations=RANDOM_ROTATIONS_5
    )
    config["experiment_args"].n_eval_epochs = len(RANDOM_ROTATIONS_5)

    return config


def make_randrot_noise_variant(
    template: Dict[str, Any],
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a copy of an experiment config w/ sensor noise and  5 random rotations.

    Args:
        template: Experiment config to copy.
        run_name: Name of the new experiment or `None`. By default, this
          function will add "_randrot_noise" to the template config's run name (if
          possible), but the run name can be specified directly via this parameter.

    Returns:
        An experiment config that uses the 5 predefined random rotations and has
        added sensor noise.
    """

    config = make_randrot_variant(template)
    config = make_noise_variant(config, run_name=run_name)
    return config


"""
Logging
--------------------------------------------------------------------------------
"""


class SelectiveEvidenceHandler(DetailedJSONHandler):
    """Detailed JSON Logger that saves limited LM and SM data for evidence logging.

    This handler stores the following subset of LM data:
     - `current_mlh`
     - `evidences`
     - `lm_processed_steps`
     - `possible_locations`
     - `possible_rotations`
     - `possible_matches`
     - `symmetry_evidence`
     - `symmetric_locations`
     - `symmetric_rotations`

    However, if `selective_handler_args["last_evidences"]` is `True`, then only final
    evidences, locations, and rotations are saved. This means `evidences`,
    `possible_locations`, and `possible_rotations` are replaced with `evidences_ls`,
    `possible_locations_ls`, and `possible_rotations_ls` respectively.

    By default, all sensor module data is saved but only for steps where an LM
    has processed data which greatly reduces storage requirements. Furthermore,
    sensor module data can be omitted entirely via the `selective_handler_args`
    argument. For example, if the supplied argument for `selective_handler_args`
    is like:
    ```
    selector_handler_args = {"exclude": ["SM_0", "SM_1"]}
    ```
    then all sensor module data for `SM_0` and `SM_1` will be omitted.

    NOTE: `selective_handler_args` can only be supplied if this class is used in
    conjunction with `SelectiveEvidenceLoggingConfig`.

    """

    def __init__(self, selective_handler_args: Optional[Mapping] = None):
        super().__init__()
        self.handler_args = (
            deepcopy(selective_handler_args) if selective_handler_args else {}
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
        # Initialize buffer data with limited LM and SM data.
        episode_total, buffer_data = self.init_buffer_data(
            data, episode, mode, **kwargs
        )

        # Finally, write the data.
        self.save(episode_total, buffer_data, output_dir)

    def save(self, episode_total: int, buffer_data: Mapping, output_dir: str) -> None:
        """Save data to a JSON file.

        Args:
            episode_total (int): Cumulative episode number (not within epoch).
            buffer_data (Mapping): Data to save.
            output_dir (str): Directory to save the data to.
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

        Populates the `buffer_data` dict with LM and SM data, though either may be
        modified or removed based on `self.handler_args`.

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
            basic = data["BASIC"]["train_stats"][episode]
        elif mode == "eval":
            episode_total = kwargs["eval_episodes_to_total"][episode]
            basic = data["BASIC"]["eval_stats"][episode]
        detailed = data["DETAILED"][episode_total]

        buffer_data = deepcopy(basic)

        # Add LM data.
        lm_ids = [key for key in detailed if key.startswith("LM")]
        lm_attrs = (
            "current_mlh",
            "evidences",
            "lm_processed_steps",
            "possible_locations",
            "possible_rotations",
            "possible_matches",
            "symmetry_evidence",
            "symmetric_locations",
            "symmetric_rotations",
            "goal_states",
            "goal_state_achieved",
        )
        for lm_id in lm_ids:
            lm_dict_in = detailed[lm_id]
            lm_dict_out = {}
            for name in lm_attrs:
                if name in lm_dict_in:
                    lm_dict_out[name] = lm_dict_in[name]
            buffer_data[lm_id] = lm_dict_out

        # Add SM data, but only where LMs have processed data.
        sm_ids = [key for key in detailed if key.startswith("SM")]
        matching_steps = self.find_matching_steps(detailed)
        for sm_id in sm_ids:
            sm_dict = dict()
            for name in [
                "raw_observations",
                "processed_observations",
                "sm_properties",
            ]:
                if name in detailed[sm_id]:
                    lst = [detailed[sm_id][name][step] for step in matching_steps]
                    sm_dict[name] = lst
            buffer_data[sm_id] = sm_dict

        # Handle last evidences.
        last_evidences = self.handler_args.get("last_evidences", False)
        if last_evidences:
            self.take_last_evidences(buffer_data)

        # Handle excludes.
        exclude = self.handler_args.get("exclude", [])
        for key in exclude:
            buffer_data.pop(key, None)

        # Return cumulative episode number and buffer data.
        return episode_total, buffer_data

    def find_matching_steps(self, detailed: Mapping) -> np.ndarray:
        """Find steps where any LM has processed data.

        Args:
            detailed (Mapping): Detailed stats.

        Returns:
            np.ndarray: Array of indices indicating which steps were matching steps.
        """
        lm_ids = [key for key in detailed if key.startswith("LM")]
        if len(lm_ids) == 1:
            is_matching_step = np.array(detailed[lm_ids[0]]["lm_processed_steps"])
        else:
            n_monty_steps = len(detailed[lm_ids[0]]["lm_processed_steps"])
            is_matching_step = np.zeros(n_monty_steps, dtype=bool)
            for step in range(n_monty_steps):
                processed = [
                    detailed[key]["lm_processed_steps"][step] for key in lm_ids
                ]
                is_matching_step[step] = any(processed)

        return np.atleast_1d(np.argwhere(is_matching_step).squeeze())

    def take_last_evidences(self, buffer_data: Mapping) -> Mapping:
        """Remove all but final values for evidence-related items.

        The following keys are removed:
            - `LM_*/evidences`
            - `LM_*/possible_locations`
            - `LM_*/possible_rotations`

        And the following keys are added:
            - `LM_*/evidences_ls`
            - `LM_*/possible_locations_ls`
            - `LM_*/possible_rotations_ls`

        Args:
            buffer_data (Mapping): The buffer data.

        Returns:
        """
        possible_keys = (
            "evidences",
            "possible_locations",
            "possible_rotations",
        )
        lm_ids = [key for key in buffer_data if key.startswith("LM")]
        for lm_id in lm_ids:
            lm_dict = buffer_data[lm_id]
            for key in possible_keys:
                if key in lm_dict:
                    val = lm_dict[key]
                    val = None if val is None else val[-1]
                    lm_dict[f"{key}_ls"] = val
                    lm_dict.pop(key)


@dataclass
class SelectiveEvidenceLoggingConfig(EvalEvidenceLMLoggingConfig):
    """Logging config best used with `SelectiveEvidenceHandler`.

    Other than using a `SelectiveEvidenceHandler` by default, this config also
    has the `selective_handler_args` attribute which can be supplied to the
    `SelectiveEvidenceHandler`'s `__init__` method.
    """

    output_dir: str = str(DMC_RESULTS_DIR)
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            SelectiveEvidenceHandler,
        ]
    )
    wandb_group: str = "dmc"
    wandb_handlers: list = field(default_factory=list)
    monty_log_level: str = "DETAILED"
    selective_handler_args: dict = field(default_factory=dict)


@dataclass
class DMCEvalLoggingConfig(ParallelEvidenceLMLoggingConfig):
    """Basic logging config with DMC-specific output directory and wandb group.

    This config also drops the reproduce episode handler which is included
    as a default handler in `ParallelEvidenceLMLoggingConfig`.
    """

    output_dir: str = str(DMC_RESULTS_DIR)
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
        ]
    )
    wandb_group: str = "dmc"
