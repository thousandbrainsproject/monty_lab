# Copyright 2025 Thousand Brains Project
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

- `fig3_evidence_run`: A one-episode experiment used to collect evidence
   and sensor data for every step. The output is read and plotted by functions in
    `scripts/fig3.py`.
- `fig4_symmetry_run`: Runs `dist_agent_1lm_randrot_noise` with storage of
   evidence and symmetry including symmetry data for the MLH object only, and only
   for the terminal step of each episode. The output is read and plotted by
   functions in `scripts/fig4.py`.
- `fig5_visualize_8lm_patches`: An one-episode, one-step experiment that is used to
  collect one set of observations for the 8-LM model. The output is read and plotted
  by functions in `scripts/fig5.py` to show how the sensors patches fall on the object.

All experiments save their results to subdirectories of `DMC_ROOT` / `visualizations`.

"""
from copy import deepcopy
from typing import Mapping

from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    ReproduceEpisodeHandler,
)

from .common import (
    DMC_PRETRAIN_DIR,
    DMC_ROOT_DIR,
    SelectiveEvidenceHandler,
    SelectiveEvidenceLoggingConfig,
)
from .fig3_robust_sensorimotor_inference import dist_agent_1lm
from .fig5_rapid_inference_with_voting import (
    dist_agent_1lm_randrot_noise,
    dist_agent_8lm_half_lms_match,
)
from .fig6_rapid_inference_with_model_based_policies import surf_agent_1lm

# Main output directory for visualization experiment results.
VISUALIZATION_RESULTS_DIR = DMC_ROOT_DIR / "visualizations"


class MLHEvidenceHandler(SelectiveEvidenceHandler):
    """Logging handler that only saves terminal evidence data for the MLH object.

    A lean logger handler for the symmetry experiment (which are full-length runs,
    and so we need to be very selective about which data to log).

    """

    def report_episode(
        self,
        data: Mapping,
        output_dir: str,
        episode: int,
        mode: str = "train",
        **kwargs,
    ) -> None:
        """Store only final evidence data and no sensor data."""

        # Initialize output data.
        self.handler_args["last_evidence"] = True  # Required for this handler.
        episode_total, buffer_data = self.init_buffer_data(
            data, episode, mode, **kwargs
        )

        # Only store evidence data for the MLH object.
        lm_ids = [key for key in buffer_data.keys() if key.startswith("LM")]
        for lm_id in lm_ids:
            mlh_object = buffer_data[lm_id]["current_mlh"][-1]["graph_id"]
            lm_dict = buffer_data[lm_id]
            evidences_ls = lm_dict["evidences_ls"]
            possible_locations_ls = lm_dict["possible_locations_ls"]
            possible_rotations_ls = lm_dict["possible_rotations_ls"]
            lm_dict["evidences_ls"] = {mlh_object: evidences_ls[mlh_object]}
            lm_dict["possible_locations_ls"] = {
                mlh_object: possible_locations_ls[mlh_object]
            }
            lm_dict["possible_rotations_ls"] = {
                mlh_object: possible_rotations_ls[mlh_object]
            }

        # Store data.
        self.save(episode_total, buffer_data, output_dir)


"""
Figure 3
-------------------------------------------------------------------------------
"""

# `fig3_evidence_run`: Experiment for collecting detailed evidence values and sensor
# data for one episode only. Used in `scripts/fig3.py` to generate evidence graphs
# and visualize the path taken by the sensor/agent.
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

"""
Figure 4
-------------------------------------------------------------------------------
"""

# `fig4_symmetry_run`: Experiment for collecting data on symmetric rotations, the
# results of which are used in `scripts/fig4.py` to investigate rotation error
# metrics when Monty has detected symmetry.
fig4_symmetry_run = deepcopy(dist_agent_1lm_randrot_noise)
fig4_symmetry_run.update(
    dict(
        logging_config=SelectiveEvidenceLoggingConfig(
            output_dir=str(VISUALIZATION_RESULTS_DIR),
            run_name="fig4_symmetry_run",
            monty_handlers=[
                BasicCSVStatsHandler,
                MLHEvidenceHandler,
            ],
            selective_handler_args=dict(exclude=["SM_0", "SM_1"], last_evidence=True),
        ),
    )
)


"""
Figure 5
-------------------------------------------------------------------------------
"""

# `fig5_visualize_8lm_patches`: An experiment that runs one eval step with
# the 8-LM model so we can collect enough sensor data to visualize the arrangement
# of the sensors patches on the object. Used in `scripts/fig5.py`. Run in serial.
fig5_visualize_8lm_patches = deepcopy(dist_agent_8lm_half_lms_match)
fig5_visualize_8lm_patches.update(
    dict(
        experiment_args=EvalExperimentArgs(
            model_name_or_path=str(DMC_PRETRAIN_DIR / "dist_agent_8lm/pretrained"),
            n_eval_epochs=1,
            max_total_steps=1,
            max_eval_steps=1,
        ),
        # Exclude LM data to save space.
        logging_config=SelectiveEvidenceLoggingConfig(
            output_dir=str(VISUALIZATION_RESULTS_DIR),
            run_name="fig5_visualize_8lm_patches",
            selective_handler_args=dict(exclude=[f"LM_{i}" for i in range(8)]),
        ),
        eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
            object_names=["mug"],
            object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
        ),
    )
)
# Set view-finder resolution to 256 x 256 for a denser "background" image. The remaining
# patches use the standard resolution for this model (64 x 64).
resolutions = [[64, 64]] * 9
resolutions[-1] = [256, 256]
dataset_args = fig5_visualize_8lm_patches["dataset_args"]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = resolutions
dataset_args.__post_init__()

"""
Figure 6
-------------------------------------------------------------------------------
"""
fig6_curvature_guided_policy = deepcopy(surf_agent_1lm)
fig6_curvature_guided_policy["experiment_args"].n_eval_epochs = 1
fig6_curvature_guided_policy["logging_config"] = SelectiveEvidenceLoggingConfig(
    output_dir=str(VISUALIZATION_RESULTS_DIR),
    run_name="fig6_curvature_guided_policy",
    selective_handler_args=dict(exclude=["LM_0"]),
)
fig6_curvature_guided_policy[
    "monty_config"
].motor_system_config.motor_system_args.use_goal_state_driven_actions = False
fig6_curvature_guided_policy["eval_dataloader_args"] = (
    EnvironmentDataloaderPerObjectArgs(
        object_names=["mug"],
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
    )
)

class TestPointHandler(SelectiveEvidenceHandler):
    """Logging handler that only saves terminal evidence data for the MLH object.

    A lean logger handler for the symmetry experiment (which are full-length runs,
    and so we need to be very selective about which data to log).

    """

    def report_episode(
        self,
        data: Mapping,
        output_dir: str,
        episode: int,
        mode: str = "train",
        **kwargs,
    ) -> None:
        """Store only final evidence data and no sensor data."""

        # Initialize output data.
        self.handler_args["last_evidence"] = True  # Required for this handler.
        episode_total, buffer_data = self.init_buffer_data(
            data, episode, mode, **kwargs
        )

        # Only store evidence data for the MLH object.
        lm_ids = [key for key in buffer_data.keys() if key.startswith("LM")]
        for lm_id in lm_ids:
            mlh_object = buffer_data[lm_id]["current_mlh"][-1]["graph_id"]
            lm_dict = buffer_data[lm_id]
            evidences_ls = lm_dict["evidences_ls"]
            possible_locations_ls = lm_dict["possible_locations_ls"]
            possible_rotations_ls = lm_dict["possible_rotations_ls"]
            lm_dict["evidences_ls"] = {mlh_object: evidences_ls[mlh_object]}
            lm_dict["possible_locations_ls"] = {
                mlh_object: possible_locations_ls[mlh_object]
            }
            lm_dict["possible_rotations_ls"] = {
                mlh_object: possible_rotations_ls[mlh_object]
            }

        # Store data.
        self.save(episode_total, buffer_data, output_dir)


fig6_surf_test_point = deepcopy(surf_agent_1lm)
fig6_surf_test_point["experiment_args"].n_eval_epochs = 1
fig6_surf_test_point["logging_config"] = SelectiveEvidenceLoggingConfig(
    output_dir=str(VISUALIZATION_RESULTS_DIR),
    run_name="fig6_surf_test_point",
    # selective_handler_args=dict(),
)
fig6_surf_test_point["eval_dataloader_args"] = EnvironmentDataloaderPerObjectArgs(
    object_names=["spoon"],
    object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
)

CONFIGS = {
    "fig3_evidence_run": fig3_evidence_run,
    "fig4_symmetry_run": fig4_symmetry_run,
    "fig5_visualize_8lm_patches": fig5_visualize_8lm_patches,
    "fig6_curvature_guided_policy": fig6_curvature_guided_policy,
    "fig6_surf_test_point": fig6_surf_test_point,
}
