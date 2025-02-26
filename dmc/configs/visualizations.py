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
import os
from copy import deepcopy
from typing import Mapping

from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
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

from .common import (
    DMC_PRETRAIN_DIR,
    DMC_ROOT_DIR,
    SelectiveEvidenceHandler,
    SelectiveEvidenceLoggingConfig,
)
from .fig3_robust_sensorimotor_inference import dist_agent_1lm
from .fig4_rapid_inference_with_voting import (
    dist_agent_1lm_randrot_noise,
    dist_agent_8lm_half_lms_match,
)

# Main output directory for visualization experiment results.
VISUALIZATION_RESULTS_DIR = os.path.join(DMC_ROOT_DIR, "visualizations")


class SelectiveEvidenceHandlerSymmetryRun(SelectiveEvidenceHandler):
    """Logging handler that only saves final evidence data no sensor data."""

    def report_episode(
        self,
        data: Mapping,
        output_dir: str,
        episode: int,
        mode: str = "train",
        **kwargs,
    ) -> None:
        episode_total, buffer_data = self.init_buffer_data(
            data, episode, mode, **kwargs
        )

        # Only store some data for the last step and for the mlh object.
        lm_ids = [key for key in buffer_data.keys() if key.startswith("LM")]
        for lm_id in lm_ids:
            mlh_object = buffer_data[lm_id]["current_mlh"][-1]["graph_id"]
            lm_dict = buffer_data[lm_id]
            evidences = lm_dict["evidences"]
            possible_locations = lm_dict["possible_locations"]
            possible_rotations = lm_dict["possible_rotations"]
            lm_dict["evidences_ls"] = {mlh_object: evidences[-1][mlh_object]}
            lm_dict["possible_locations_ls"] = {
                mlh_object: possible_locations[-1][mlh_object]
            }
            lm_dict["possible_rotations_ls"] = {
                mlh_object: possible_rotations[-1][mlh_object]
            }
            lm_dict.pop("evidences")
            lm_dict.pop("possible_locations")
            lm_dict.pop("possible_rotations")

        # Remove sensor module data.
        sm_ids = [key for key in buffer_data.keys() if key.startswith("SM")]
        for sm_id in sm_ids:
            buffer_data.pop(sm_id)

        # Save data.
        self.save(episode_total, buffer_data, output_dir)


"""
Figure 3
"""


fig3_evidence_run = deepcopy(dist_agent_1lm)
fig3_evidence_run.update(
    dict(
        experiment_args=EvalExperimentArgs(
            model_name_or_path=str(
                DMC_PRETRAIN_DIR / "dist_agent_1lm_10distinctobj/pretrained"
            ),
            n_eval_epochs=1,
            max_total_steps=100,
            max_eval_steps=100,
        ),
        logging_config=SelectiveEvidenceLoggingConfig(
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
# Experiment for visualizing symmetric poses

fig3_symmetry_run = deepcopy(dist_agent_1lm_randrot_noise)
fig3_symmetry_run.update(
    dict(
        logging_config=SelectiveEvidenceLoggingConfig(
            output_dir=str(VISUALIZATION_RESULTS_DIR),
            run_name="fig3_symmetry_run",
            monty_handlers=[
                BasicCSVStatsHandler,
                SelectiveEvidenceHandlerSymmetryRun,
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
