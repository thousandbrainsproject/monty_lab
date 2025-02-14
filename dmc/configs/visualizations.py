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
)

from .common import (
    DMC_PRETRAIN_DIR,
    DMC_ROOT_DIR,
)
from .fig3_robust_sensorimotor_inference import dist_agent_1lm
from .fig4_rapid_inference_with_voting import dist_agent_8lm_half_lms_match
from .fig9_structured_object_representations import (
    EvidenceLoggingMontyObjectRecognitionExperiment,
)

# Main output directory for visualization experiment results.
VISUALIZATION_RESULTS_DIR = os.path.join(DMC_ROOT_DIR, "visualizations")


"""
Figure 3
"""


fig3_evidence_run = deepcopy(dist_agent_1lm)
fig3_evidence_run.update(
    dict(
        experiment_class=EvidenceLoggingMontyObjectRecognitionExperiment,
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
            wandb_group="dmc",
            monty_log_level="SELECTIVE",
        ),
        eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
            object_names=["mug"],
            object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
        ),
    )
)
# We want to be able to run for at least 40 consecutive steps (without jumps) so we
# can plot a nice smooth trajectory over the object model.
fig3_evidence_run["monty_config"].monty_args.min_eval_steps = 41
fig3_evidence_run[
    "monty_config"
].motor_system_config.motor_system_args.use_goal_state_driven_actions = False


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
    "fig4_visualize_8lm_patches": fig4_visualize_8lm_patches,
}
