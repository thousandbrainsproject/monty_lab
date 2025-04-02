# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Figure 7: Rapid Learning

Consists of one pretraining experiment and 6 evaluation experiments:
- pretrain_dist_agent_1lm_checkpoints
- dist_agent_1lm_randrot_nohyp_1rot_trained
- dist_agent_1lm_randrot_nohyp_2rot_trained
- dist_agent_1lm_randrot_nohyp_4rot_trained
- dist_agent_1lm_randrot_nohyp_8rot_trained
- dist_agent_1lm_randrot_nohyp_16rot_trained
- dist_agent_1lm_randrot_nohyp_32rot_trained

This means performance is evaluated with:
- 77 objects
- 5 random rotations
- NO sensor noise*
- NO hypothesis-testing*
- No voting
- Varying numbers of rotations trained on (evaluations use different baseline models)

"""

import time
from copy import deepcopy
from pathlib import Path

import torch
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)

from .common import DMC_PRETRAIN_DIR, make_randrot_variant
from .fig3_robust_sensorimotor_inference import dist_agent_1lm
from .pretraining_experiments import DMCPretrainLoggingConfig, pretrain_dist_agent_1lm

"""
Pretraining Configs
--------------------------------------------------------------------------------
"""

TRAIN_ROTATIONS_32 = [
    # cube faces
    [0, 0, 0],
    [0, 90, 0],
    [0, 180, 0],
    [0, 270, 0],
    [90, 0, 0],
    [90, 180, 0],
    # cube corners
    [35, 45, 0],
    [325, 45, 0],
    [35, 315, 0],
    [325, 315, 0],
    [35, 135, 0],
    [325, 135, 0],
    [35, 225, 0],
    [325, 225, 0],
    # random rotations (numpy.random.randint)
    [305, 143, 316],
    [63, 302, 307],
    [286, 207, 136],
    [164, 2, 181],
    [276, 68, 121],
    [114, 88, 272],
    [152, 206, 301],
    [242, 226, 282],
    [235, 321, 32],
    [33, 243, 166],
    [65, 298, 9],
    [185, 14, 224],
    [259, 249, 53],
    [113, 8, 73],
    [20, 158, 74],
    [289, 327, 94],
    [148, 181, 282],
    [240, 143, 10],
]


class PretrainingExperimentWithCheckpointing(
    MontySupervisedObjectPretrainingExperiment
):
    """Supervised pretraining class that saves the model after every epoch.

    NOTE: Experiments using this class cannot be run in parallel.
    """

    def post_epoch(self):
        """Store a model checkpoint."""
        super().post_epoch()

        # Check which epooch?
        if self.train_epochs == 1:
            self.t_last_checkpoint = time.time()
        else:
            t_per_epoch = time.time() - self.t_last_checkpoint
            mins, secs = divmod(t_per_epoch, 60)
            print(f"Time per epoch: {mins:.2f} minutes, {secs:.2f} seconds")
            self.t_last_checkpoint = time.time()

        # Save the model.
        checkpoints_dir = Path(self.output_dir) / "checkpoints"
        output_dir = checkpoints_dir / f"{self.train_epochs}"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pt"
        model_state_dict = self.model.state_dict()
        torch.save(model_state_dict, model_path)


pretrain_dist_agent_1lm_checkpoints = deepcopy(pretrain_dist_agent_1lm)
pretrain_dist_agent_1lm_checkpoints.update(
    dict(
        experiment_class=PretrainingExperimentWithCheckpointing,
        experiment_args=ExperimentArgs(
            n_train_epochs=len(TRAIN_ROTATIONS_32),
            do_eval=False,
        ),
        logging_config=DMCPretrainLoggingConfig(run_name="dist_agent_1lm_checkpoints"),
        train_dataloader_class=ED.InformedEnvironmentDataLoader,
        train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
            object_names=SHUFFLED_YCB_OBJECTS,
            object_init_sampler=PredefinedObjectInitializer(
                rotations=TRAIN_ROTATIONS_32
            ),
        ),
    )
)

"""
Evaluation Configs
--------------------------------------------------------------------------------
"""


def make_partially_trained_eval_config(n_rot: int) -> dict:
    """Make an eval config that loads a pretrained model checkpoint.

    The returned config specifies a 1-LM distant agent that evaluates on
      - All 77 YCB objects
      - 5 (predetermined) random rotations
      - no sensor noise
      - no hypothesis-driven actions
    and loads a model pretrained after `n_rot` observations per object.

    Args:
        n_rot (int): Number of rotations trained on. This controls which model
          checkpoint is loaded, so if we want to evaluate using a model that has only
          been trained on, say, 4 rotations, we should pass `n_rot=4`.
    Returns:
        dict: Config for a partially trained model.
    """
    # Use base distant agent model with 5 random rotations
    config = make_randrot_variant(dist_agent_1lm)

    # Change model loading path
    config["experiment_args"].model_name_or_path = str(
        DMC_PRETRAIN_DIR
        / f"dist_agent_1lm_checkpoints/pretrained/checkpoints/{n_rot}/model.pt"
    )

    # Rename the experiment
    config[
        "logging_config"
    ].run_name = f"dist_agent_1lm_randrot_nohyp_{n_rot}rot_trained"

    # Disable hypothesis-driven actions
    config[
        "monty_config"
    ].motor_system_config.motor_system_args.use_goal_state_driven_actions = False

    return config


dist_agent_1lm_randrot_nohyp_1rot_trained = make_partially_trained_eval_config(1)
dist_agent_1lm_randrot_nohyp_2rot_trained = make_partially_trained_eval_config(2)
dist_agent_1lm_randrot_nohyp_4rot_trained = make_partially_trained_eval_config(4)
dist_agent_1lm_randrot_nohyp_8rot_trained = make_partially_trained_eval_config(8)
dist_agent_1lm_randrot_nohyp_16rot_trained = make_partially_trained_eval_config(16)
dist_agent_1lm_randrot_nohyp_32rot_trained = make_partially_trained_eval_config(32)

CONFIGS = {
    "pretrain_dist_agent_1lm_checkpoints": pretrain_dist_agent_1lm_checkpoints,
    "dist_agent_1lm_randrot_nohyp_1rot_trained": dist_agent_1lm_randrot_nohyp_1rot_trained,
    "dist_agent_1lm_randrot_nohyp_2rot_trained": dist_agent_1lm_randrot_nohyp_2rot_trained,
    "dist_agent_1lm_randrot_nohyp_4rot_trained": dist_agent_1lm_randrot_nohyp_4rot_trained,
    "dist_agent_1lm_randrot_nohyp_8rot_trained": dist_agent_1lm_randrot_nohyp_8rot_trained,
    "dist_agent_1lm_randrot_nohyp_16rot_trained": dist_agent_1lm_randrot_nohyp_16rot_trained,
    "dist_agent_1lm_randrot_nohyp_32rot_trained": dist_agent_1lm_randrot_nohyp_32rot_trained,
}
