import copy
import time
from pathlib import Path

import torch
from tbp.monty.frameworks.config_utils.config_args import (
    PretrainLoggingConfig,
)
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

from .common import PRETRAIN_DIR
from .pretraining_experiments import pretrain_dist_agent_1lm

"""
Rapid Learning Config (for storing checkpoints)
--------------------------------------------------------------------------------
"""

TRAIN_ROTATIONS = [
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
    """Supervised pretraining class that saves the model after certain epochs.

    NOTE: I'm not sure this can be done in parallel runs. Post-parallel cleanup,
    deletes checkpoints, and I'm not sure the models are combined in a reasonable
    way during parallel mode mid-experiment anyways. It may be better just to
    define the handful of pretraining experiments we really want rather than
    checkpointing, but this is worth trying.
    """

    def post_epoch(self):
        """Save the model.
        TODO: Check how well this works in parallel runs. Could be problematic.
        TODO: Save checkpoints every...
        """
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
        # checkpoints_dir = Path(self.output_dir).parent / "checkpoints"
        output_dir = checkpoints_dir / f"{self.train_epochs}"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pt"
        model_state_dict = self.model.state_dict()
        torch.save(model_state_dict, model_path)


pretrain_dist_agent_1lm_checkpoints = copy.deepcopy(pretrain_dist_agent_1lm)
pretrain_dist_agent_1lm_checkpoints.update(
    dict(
        experiment_class=PretrainingExperimentWithCheckpointing,
        experiment_args=ExperimentArgs(
            n_train_epochs=len(TRAIN_ROTATIONS),
            do_eval=False,
        ),
        logging_config=PretrainLoggingConfig(
            output_dir=str(PRETRAIN_DIR),
            run_name="dist_agent_1lm_checkpoints",
        ),
        train_dataloader_class=ED.InformedEnvironmentDataLoader,
        train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
            object_names=SHUFFLED_YCB_OBJECTS,
            object_init_sampler=PredefinedObjectInitializer(rotations=TRAIN_ROTATIONS),
        ),
    )
)

CONFIGS = {
    "pretrain_dist_agent_1lm_checkpoints": pretrain_dist_agent_1lm_checkpoints,
}
