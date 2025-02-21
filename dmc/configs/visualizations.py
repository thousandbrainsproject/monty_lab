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
from typing import List, Mapping, Optional

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

# @dataclass
# class EvalEvidenceLMLoggingConfig(LoggingConfig):
#     output_dir: str = os.path.expanduser(
#         os.path.join(monty_logs_dir, "projects/evidence_eval_runs/logs")
#     )
#     monty_handlers: List = field(
#         default_factory=lambda: [
#             BasicCSVStatsHandler,
#             ReproduceEpisodeHandler,
#         ]
#     )
#     wandb_handlers: List = field(
#         default_factory=lambda: [
#             BasicWandbTableStatsHandler,
#             BasicWandbChartStatsHandler,
#             # DetailedWandbMarkedObsHandler,
#         ]
#     )
#     wandb_group: str = "evidence_eval_runs"
#     monty_log_level: str = "BASIC"

import fnmatch

try:
    import h5py

    has_h5py = True
except ImportError:
    has_h5py = False


class EvidenceJSONHandler(DetailedJSONHandler):
    """_summary_

    - LM_*: dict
      -

    Filters:
      - On a dict:
        - None: don't filter.
        - str: key to use or glob pattern of keys (e.g., "SM_*").
        - list[str]: list of keys (or glob patterns).

      - On a list:
        - None: don't filter.
        - int: index to use, such as 0 or -1 for the first and last observations.
        - str: a condition, such as '$lm_processed' (for SM items).

    Examples:
    ```
    # Store first RGBA observation for SM_1 (view_finder).
    filt = ["SM_1", "raw_observations", 0, "rgba"]

    # For all SMs, store RGBA and depth for steps where an LM has processed data.
    filt = ["SM_*", "raw_observations", "lm_processed", ["rgba", "depth"]]

    # For all LMs, store the final evidence counts for all objects.
    filt = ["LM_*", "evidences", -1, None] # None means all keys.

    # For LM_0, store hypotheses counts and locations for just the primary target.
    filt = ["LM_*", "possible*, None, "$primary_target] # None means all keys.
    ```
    Supported special names (i.e., those that start with '$') are:
      - '$lm_processed': condition is true when any LM has processed data.
      - '$target': condition is true when the key matches the target of the episode.

    Args:
        DetailedJSONHandler (_type_): _description_

    Returns:
        _type_: _description_
    """

    use_h5py: bool = True

    # Filters
    include: Optional[List[List]] = None
    exclude: Optional[List[List]] = None

    def __init__(self, detailed_handler_args: Optional[Mapping] = None):
        super().__init__()
        self.handler_args = dict(detailed_handler_args) if detailed_handler_args else {}

        # Decide whether to use h5py.
        if has_h5py:
            self.handler_args.get("use_h5py", self.use_h5py)
        else:
            self.use_h5py = False

        # Handle filters.
        self.include = list(self.include) if self.include else []
        self.include.extend(self.handler_args.get("include", []))
        self.exclude = list(self.exclude) if self.exclude else []
        self.exclude.extend(self.handler_args.get("exclude", []))

    def init(self, data: Mapping, output_dir: str) -> None:
        # Add include filters from handler args.

        self.json_path = os.path.join(output_dir, "detailed_run_stats.json")
        self.h5_path = os.path.join(output_dir, "detailed_run_stats.h5")
        maybe_rename_existing_file(self.json_path, ".json", self.report_count)
        maybe_rename_existing_file(self.h5_path, ".h5", self.report_count)

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
            episode (int): Episode number within the epoch. (right?)
            mode (str): Either "train" or "eval".
            **kwargs: Additional keyword arguments.

        Changed name to report episode since we are currently running with
        reporting and flushing exactly once per episode.
        """
        output_data = dict()
        if mode == "train":
            total = kwargs["train_episodes_to_total"][episode]
            basic = data["BASIC"]["train_stats"][episode]

        elif mode == "eval":
            total = kwargs["eval_episodes_to_total"][episode]
            basic = data["BASIC"]["eval_stats"][episode]

        detailed = data["DETAILED"][total]

        # Initialize special variables.
        target = basic["target"]["primary_target_object"]
        lm_processed_steps = self.find_lm_processed_steps(detailed)

        sm_ids = [key for key in detailed if key.startswith("SM")]
        lm_ids = [key for key in detailed if key.startswith("LM")]

        lm_processed_steps = self.find_lm_processed_steps(detailed)
        n_monty_steps = len(detailed[sm_ids[0]]["raw_observations"])

        # Filter data.
        json_data = {}
        json_data[total] = copy.deepcopy(basic)

        output_data[total] = copy.deepcopy(stats)
        output_data[total].update(data["DETAILED"][total])

        self.init(data, output_dir)
        with open(self.json_path, "a") as f:
            json.dump({total: output_data[total]}, f, cls=BufferEncoder)
            f.write(os.linesep)

        print("Stats appended to " + self.json_path)
        self.report_count += 1

    def dump_with_h5py(
        self,
        data: Mapping,
        output_dir: str,
        episode: int,
        mode: str = "train",
        **kwargs,
    ) -> None:
        pass

    def dump_with_json(
        self,
        basic: Mapping,
        detailed: Mapping,
        output_dir: str,
        episode_total: int,
        **kwargs,
    ) -> None:
        """

        Args:
            basic (Mapping): _description_
            detailed (Mapping): _description_
            output_dir (str): _description_
            episode_total (int): _description_
        """
        if not self.include and not self.exclude:
            output_data = dict()
            output_data[episode_total] = copy.deepcopy(basic)
            output_data[episode_total].update(detailed)
            return output_data

        # Filter data.
        buf = {}
        if self.include:
            for include_filter in self.include:
                cur_data = detailed

                for key in include_filter:
                    if isinstance(key, str):
                        if fnmatch.fnmatch(key, include_filter):
                            buf[key] = detailed[key]

    def find_lm_processed_steps(self, detailed: Mapping) -> np.ndarray:
        """Find steps where any LM has processed data.

        Args:
            detailed (Mapping): Data from a single episode.

        Returns:
            np.ndarray: Bool area indicating which steps had an lm processing.
        """
        lm_ids = [key for key in detailed if key.startswith("LM")]
        if len(lm_ids) == 1:
            return np.array(detailed[lm_ids[0]]["lm_processed_steps"])

        n_monty_steps = len(detailed[lm_ids[0]]["lm_processed_steps"])
        lm_processed_steps = np.zeros(n_monty_steps, dtype=bool)
        for step in range(n_monty_steps):
            processed = [detailed[key]["lm_processed_steps"][step] for key in lm_ids]
            lm_processed_steps[step] = any(processed)
        return lm_processed_steps


@dataclass
class CustomEvidenceLMLoggingConfig(EvalEvidenceLMLoggingConfig):
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            EvidenceJSONHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_handlers: list = field(default_factory=list)
    monty_log_level: str = "DETAILED"
    detailed_handler_args: dict = field(default_factory=dict)


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
        logging_config=DetailedEvidenceLMLoggingConfig(
            output_dir=str(VISUALIZATION_RESULTS_DIR),
            run_name="fig3_evidence_run",
            wandb_handlers=[],
            # detailed_handler_args=dict(test_arg="test"),
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
        experiment_class=EvidenceLoggingMontyObjectRecognitionExperiment,
        logging_config=DetailedEvidenceLMLoggingConfig(
            output_dir=str(VISUALIZATION_RESULTS_DIR),
            run_name="fig3_symmetry_run",
            wandb_group="dmc",
            monty_log_level="SELECTIVE",
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
