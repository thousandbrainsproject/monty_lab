import copy
from pathlib import Path

import torch

from dmc_configs.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
    PretrainLoggingConfig,
)

from .pretraining_experiments import pretrain_dist_agent_1lm

"""
Define pretraining experiment(s)
"""


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

        # Save the model.
        output_dir = Path(self.output_dir) / f"checkpoints/{self.train_epochs}"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pt"
        model_state_dict = self.model.state_dict()
        torch.save(model_state_dict, model_path)


"""
Rapid Learning Config (for storing checkpoints)
--------------------------------------------------------------------------------
"""

pretrain_dist_agent_1lm_checkpoints = copy.deepcopy(pretrain_dist_agent_1lm)
pretrain_dist_agent_1lm_checkpoints["experiment_class"] = (
    PretrainingExperimentWithCheckpointing
)
pretrain_dist_agent_1lm_checkpoints["logging_config"] = PretrainLoggingConfig(
    run_name="dist_agent_1lm_checkpoints"
)

# pretrain_dist_agent_1lm_checkpoints[
#     "train_dataloader_args"
# ].object_names = DISTINCT_OBJECTS

CONFIGS = {
    "pretrain_dist_agent_1lm_checkpoints": pretrain_dist_agent_1lm_checkpoints,
}
