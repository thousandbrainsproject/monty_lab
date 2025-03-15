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
Config Functions for Creating/Modifying Evaluation Experiments Configs
----------------------------------------------------------------------
Note: See `pretraining_experiments.py` for functions that return/modify configs
suitable for pretraining experiments.

"""


def get_dist_lm_config(
    sensor_module_id: str = "patch",
    color: bool = True,
) -> Mapping:
    """Create a learning module config for a distant agent.

    This function returns a learning module config that uses default settings for
    DMC distant agent evaluation experiments. It is identical to the surface agent LM
    config (see `get_surf_lm_config`) except for the goal state generator's
    `desired_object_distance` argument (which is set to 0.03 meters, rather than
    0.025 meters as in the surface agent LM config).

    NOTE: Separate functions for getting distant vs surface agent LM configs were
    created for development purposes, but they may be merged in the future given
    that their respective defaults have not diverged.

    Args:
        sensor_module_id (str): The sensor module this LM receives input from.
          Defaults to "patch" which is appropriate for single-LM experiments.
        color (bool): Whether to include color (HSV) features. Defaults to True. If
          False, then color-related parameters are removed.

    Returns:
        dict: A dictionry with two items:
            - "learning_module_class": The EvidenceGraphLM class.
            - "learning_module_args": A dictionary of arguments for the EvidenceGraphLM
              class.
    """
    out = dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            # Specify graph matching thresholds and tolerances.
            max_match_distance=0.01,  # 1 cm
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
                desired_object_distance=0.03,  # Further than surface agent.
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
) -> Mapping:
    """Create a learning module config for a surface agent.

    This function returns a learning module config that uses default settings for
    DMC surface agent evaluation experiments. It is identical to the distant agent LM
    config (see `get_dist_lm_config`) except for the goal state generator's
    `desired_object_distance` argument (which is set to 0.025 meters, rather than
    0.03 meters as in the distant agent LM config).

    NOTE: Separate functions for getting distant vs surface agent LM configs were
    created for development purposes, but they may be merged in the future given
    that their respective defaults have not diverged.

    Args:
        sensor_module_id (str): The sensor module this LM receives input from.
          Defaults to "patch" which is appropriate for single-LM experiments.
        color (bool): Whether to include color (HSV) features. Defaults to True. If
          False, then color-related parameters are removed.

    Returns:
        dict: A dictionary with two items:
            - "learning_module_class": The EvidenceGraphLM class.
            - "learning_module_args": A dictionary of arguments for the EvidenceGraphLM
              class.
    """
    out = dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            # Specify graph matching thresholds and tolerances.
            max_match_distance=0.01,  # 1 cm
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
                desired_object_distance=0.025,  # Closer than distant agent.
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
) -> Mapping:
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
) -> Mapping:
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


def get_view_finder_config() -> Mapping:
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
    """Get default motor system config for distant agent evaluation experiments.

    Returns:
        MotorSystemConfigInformedGoalStateDriven: Motor system configuration for
            distant agents that uses goal states to drive actions.
    """
    return MotorSystemConfigInformedGoalStateDriven()


def get_surf_motor_config() -> MotorSystemConfigCurInformedSurfaceGoalStateDriven:
    """Get default motor system config for surface agent evaluation experiments.

    Returns:
        MotorSystemConfigCurInformedSurfaceGoalStateDriven: Motor system configuration
            for surface agents that uses curvature-informed goal states to drive
            actions.
    """
    return MotorSystemConfigCurInformedSurfaceGoalStateDriven()


"""
Functions for generating variations of existing configs.
--------------------------------------------------------
"""


def add_sensor_noise(
    config: Mapping,
    color: bool = True,
    pose_vectors: float = 2.0,
    hsv: float = 0.1,
    principal_curvatures_log: float = 0.1,
    pose_fully_defined: float = 0.01,
    location: float = 0.002,
) -> None:
    """Add sensor noise to an experiment config. Modifies the config in-place.

    Applies noise parameters to all sensor modules except the view finder. The
    `color` parameter controls whether to add 'hsv' noise. Set this to `False` for
    touch agent experiments or when a a monty model uses a pretrained model
    constructed with a touch agent (i.e., for multimodal experiments).

    Args:
        config (dict): Experiment config to add sensor noise to.
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


def make_noise_variant(
    template: Mapping,
    color: bool = True,
    run_name: Optional[str] = None,
) -> Mapping:
    """Create a copy of an experiment config with added sensor noise.

    Args:
        template (Mapping): Experiment config to copy.
        color (bool): Whether to add noise to color features. Defaults to True.
        run_name (str, optional): Name of the new experiment. By default, this
          function will add "_noise" to the template config's run name (if possible),
          but the run name can be specified directly via this parameter.

    Returns:
        Mapping: Copy of `template` with added sensor noise.
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
    add_sensor_noise(config, color=color)

    return config


def make_randrot_variant(
    template: Mapping,
    run_name: Optional[str] = None,
) -> Mapping:
    """Create a copy of an experiment config that uses the 5 predefined random rotations.

    Args:
        template (Mapping): Experiment config to copy.
        run_name (str, optional): Name of the new experiment. By default, this
          function will add "_randrot" to the template config's run name (if possible),
          but the run name can be specified directly via this parameter.

    Returns:
        Mapping: Copy of `template` that uses the 5 predefined random rotations.
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
    template: Mapping,
    color: bool = True,
    run_name: Optional[str] = None,
) -> Mapping:
    """Create a copy of an experiment config w/ sensor noise and  5 random rotations.

    Args:
        template (Mapping): Experiment config to copy.
        color (bool): Whether to add noise to color features. Defaults to True.
        run_name (str, optional): Name of the new experiment. By default, this
          function will add "_randrot_noise" to the template config's run name (if
          possible), but the run name can be specified directly via this parameter.

    Returns:
        Mapping: Copy of `template` that uses the 5 predefined random rotations and has
        added sensor noise.
    """

    config = make_randrot_variant(template)
    config = make_noise_variant(config, color=color, run_name=run_name)
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

    However, if `selective_handler_args["last_evidence"]` is `True`, then only final
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

    NOTE: Though not otherwise used (to my knowledge), Monty does have a mechanism for
    passing arguments to a handler's `__init__` function. If the parent `LoggingConfig`
    has an attribute whose name matches a parameter in a handler's `__init__`, then
    that attribute will be supplied to the handler's `__init__`. To do so, use the
    `SelectiveEvidenceLoggingConfig` class and set the `selective_handler_args`
    attribute.

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
        )
        for lm_id in lm_ids:
            lm_dict = detailed[lm_id]
            lm_dict_out = {}
            for name in lm_attrs:
                lm_dict_out[name] = lm_dict.get(name, None)
            buffer_data[lm_id] = lm_dict_out

        # Optionally, only store the final evidences, locations, and rotations.
        last_evidence = self.handler_args.get("last_evidence", False)
        last_lm_attrs = (
            "evidences",
            "possible_locations",
            "possible_rotations",
        )
        if last_evidence:
            for lm_id in lm_ids:
                lm_dict = buffer_data[lm_id]
                for name in last_lm_attrs:
                    val = lm_dict.get(name)
                    if val is None:
                        lm_dict[f"{name}_ls"] = None
                    else:
                        lm_dict[f"{name}_ls"] = val[-1]
                    lm_dict.pop(name, None)

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

        # Return cumulative episode number and buffer data.
        return episode_total, buffer_data

    def find_lm_processed_steps(self, detailed: Mapping) -> np.ndarray:
        """Find steps where any LM has processed data.

        Args:
            detailed (Mapping): Detailed stats.

        Returns:
            np.ndarray: Array of indices indicating which steps were matching steps.
        """
        lm_ids = [key for key in detailed if key.startswith("LM")]
        if len(lm_ids) == 1:
            bool_array = np.array(detailed[lm_ids[0]]["lm_processed_steps"])
        else:
            n_monty_steps = len(detailed[lm_ids[0]]["lm_processed_steps"])
            bool_array = np.zeros(n_monty_steps, dtype=bool)
            for step in range(n_monty_steps):
                processed = [
                    detailed[key]["lm_processed_steps"][step] for key in lm_ids
                ]
                bool_array[step] = any(processed)

        return np.atleast_1d(np.argwhere(bool_array).squeeze())


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
            ReproduceEpisodeHandler,
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