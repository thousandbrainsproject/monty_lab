# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Configs for visualizations (not core experiments)

This file contains configs defined solely for making visualizations that go into
paper figures. The configs defined are:

- `fig3_evidence_run`: A one-episode experiment used to collect evidence data
   at every step of the episode. The output is read and plotted by `scripts/fig3.py`.
- `fig4_visualize_8lm_patches`: An one-episode, one-step experiment that is used to
  collect one set of observations for the 8-LM model. The output is read and plotted
  by `scripts/fig4.py` to show how the sensors patches fall on the object.

All experiments save their results to subdirectories of `DMC_ROOT` / `visualizations`.

"""
import copy
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Tuple

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
    EvalEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
    MontyHandler,
    ReproduceEpisodeHandler,
)
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.utils.logging_utils import maybe_rename_existing_file

from .common import (
    DMC_PRETRAIN_DIR,
    DMC_ROOT_DIR,
)
from .fig3_robust_sensorimotor_inference import dist_agent_1lm
from .fig4_rapid_inference_with_voting import (
    dist_agent_1lm_randrot_noise,
    dist_agent_8lm_half_lms_match,
)
from .fig9_structured_object_representations import (
    EvidenceLoggingMontyObjectRecognitionExperiment,
)

# Main output directory for visualization experiment results.
VISUALIZATION_RESULTS_DIR = os.path.join(DMC_ROOT_DIR, "visualizations")



class SelectiveHandler(MontyHandler):
    """Detailed Logger that only saves evidence data and limited sensor data.

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
    """

    def __init__(self, selective_handler_args: Optional[Mapping] = None):
        super().__init__()
        self.handler_args = (
            dict(selective_handler_args) if selective_handler_args else {}
        )
        self.report_count = 0

    @classmethod
    def log_level(cls):
        return "DETAILED"

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

class SelectiveHandlerLastEvidence(SelectiveHandler):
    """Detailed Logger that only saves evidence data and limited sensor data."""

    def report_episode(
        self,
        data: Mapping,
        output_dir: str,
        episode: int,
        mode: str = "train",
        **kwargs,
    ):
        # Initialize buffer data, using only certain LM data and only sensor data
        # for steps where an LM has processed data.
        episode_total, buffer_data = self.init_buffer_data(
            data, episode, mode, **kwargs
        )

        lm_ids = [key for key in buffer_data.keys() if key.startswith("LM")]
        for lm_id in lm_ids:
            lm_dict = buffer_data[lm_id]
            lm_dict["evidences_ls"] = lm_dict["evidences"][-1]
            lm_dict["possible_locations_ls"] = lm_dict["possible_locations"][-1]
            lm_dict["possible_rotations_ls"] = lm_dict["possible_rotations"]
            lm_dict.pop("evidences")
            lm_dict.pop("possible_locations")
            lm_dict.pop("possible_rotations")

        # Save data.
        self.save(episode_total, buffer_data, output_dir)


@dataclass
class SelectiveLoggingConfig(EvalEvidenceLMLoggingConfig):
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            SelectiveHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_handlers: list = field(default_factory=list)
    monty_log_level: str = "DETAILED"
    selective_handler_args: dict = field(default_factory=dict)


"""
Figure 3
"""


fig3_evidence_run = deepcopy(dist_agent_1lm)
fig3_evidence_run.update(
    dict(
        # experiment_class=EvidenceLoggingMontyObjectRecognitionExperiment,
        experiment_args=EvalExperimentArgs(
            model_name_or_path=str(
                DMC_PRETRAIN_DIR / "dist_agent_1lm_10distinctobj/pretrained"
            ),
            n_eval_epochs=1,
            max_total_steps=100,
            max_eval_steps=100,
        ),
        logging_config=SelectiveLoggingConfig(
            output_dir=str(VISUALIZATION_RESULTS_DIR),
            run_name="fig3_evidence_run",
        ),
        eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
            object_names=["mug"],
            object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
        ),
    )
)
fig3_evidence_run["monty_config"].monty_args.min_eval_steps = 40
# ----

fig3_symmetry_run = deepcopy(dist_agent_1lm_randrot_noise)
fig3_symmetry_run.update(
    dict(
        # experiment_class=EvidenceLoggingMontyObjectRecognitionExperiment,
        logging_config=SelectiveLoggingConfig(
            output_dir=str(VISUALIZATION_RESULTS_DIR),
            run_name="fig3_symmetry_run",
            monty_handlers=[
                BasicCSVStatsHandler,
                SelectiveHandlerLastEvidence,
                ReproduceEpisodeHandler,
            ],
            selective_handler_args=dict(exclude=["SM_0", "SM_1"]),
        ),
    )
)


"""
Figure 4
"""


fig4_visualize_8lm_patches = deepcopy(dist_agent_8lm_half_lms_match)
fig4_visualize_8lm_patches.update(
    dict(
        experiment_args=EvalExperimentArgs(
            model_name_or_path=str(DMC_PRETRAIN_DIR / "dist_agent_8lm/pretrained"),
            n_eval_epochs=1,
            max_total_steps=1,
            max_eval_steps=1,
        ),
        logging_config=DetailedEvidenceLMLoggingConfig(
            output_dir=str(VISUALIZATION_RESULTS_DIR),
            run_name="fig4_visualize_8lm_patches",
            monty_log_level="DETAILED",
            monty_handlers=[BasicCSVStatsHandler, DetailedJSONHandler],
        ),
        eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
            object_names=["mug"],
            object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
        ),
    )
)
# Need to reduce num_exploratory_steps to 1 to ensure we only take one step. ??
# fig4_visualize_8lm_patches["monty_config"].monty_args.num_exploratory_steps = 1

# Set view-finder resolution to 256 x 256 for a denser "background" image. The remaining
# patches use the standard resolution for this model (64 x 64).
resolutions = [[64, 64]] * 9
resolutions[-1] = [256, 256]
dataset_args = fig4_visualize_8lm_patches["dataset_args"]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = resolutions
dataset_args.__post_init__()


CONFIGS = {
    "fig3_evidence_run": fig3_evidence_run,
    "fig3_symmetry_run": fig3_symmetry_run,
    "fig4_visualize_8lm_patches": fig4_visualize_8lm_patches,
}
