# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Used to be in frameworks/experiments/experiment_classes/

from copy import deepcopy

import matplotlib.pyplot as plt
from tbp.monty.frameworks.experiments.experiment_classes.monty_experiment import (
    MontyExperiment,
)
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.htm_monty_model import MontyHTM
from tbp.monty.frameworks.utils.dataclass_utils import config_to_dict
from torch.utils.data import DataLoader, Subset

# turn interactive plotting off -- call plt.show() to open all figures
plt.ioff()


class HTMExperiment(MontyExperiment):
    """
    Intended for running grid cell experiments that do not involve an embodied
    environment simulation.
    """

    def setup_experiment(self, config):
        self.config = config
        config = deepcopy(config)
        config = config_to_dict(config)

        self.run_args = config["run_args"]
        self.do_train = self.run_args["do_train"]
        self.do_eval = self.run_args["do_eval"]

        # load dataset
        dataset_class = config["dataset_class"]
        dataset_args = config["dataset_args"]

        self.train_dataset = dataset_class(**dataset_args)

        dataset_args.update(train=False)
        self.eval_dataset = dataset_class(**dataset_args)

        self.init_dataloaders()
        self.init_model()

    def init_dataloaders(self):
        self.train_loader = DataLoader(
            Subset(
                self.train_dataset,
                indices=self.train_dataset.objects
            ),
            batch_size=1,
            shuffle=False
        )

        self.eval_loader = DataLoader(
            Subset(
                self.eval_dataset,
                indices=self.eval_dataset.objects
            ),
            batch_size=1,
            shuffle=False
        )

    def init_model(self):
        # TODO: generalize to multiple LMs and SMs

        model_config = self.config["monty_config"]

        assert issubclass(model_config["monty_class"], MontyHTM)

        learning_modules = {}
        for lm_id, lm_cfg in model_config["learning_module_configs"].items():
            lm_class = lm_cfg["learning_module_class"]
            lm_args = lm_cfg["learning_module_args"]
            assert issubclass(lm_class, LearningModule)
            learning_modules[lm_id] = lm_class(**lm_args)

        self.model = model_config["monty_class"](
            learning_modules=[learning_modules[i] for i in learning_modules]
        )

    def run_train_episode(self, curvatures, coordinates):
        for (curve, coord) in zip(curvatures, coordinates):
            self.model.step((curve, coord))

            if self.model.is_done():
                break

    def run_eval_episode(self, object_id, curvatures, coordinates):
        for i, (curve, coord) in enumerate(zip(curvatures, coordinates)):
            self.model.step((i, (object_id, curve, coord)))

            if self.model.is_done():
                break

    def train(self):
        self.model.set_step_type(step_type="exploratory_step")

        self.model.pre_epoch()

        # only one training epoch is executed for this experiment type.
        # execute all the episodes directly.
        for object_id, curvatures, coordinates in self.train_loader:
            self.model.pre_episode()

            self.run_train_episode(curvatures, coordinates.squeeze())

            self.model.post_episode(object_id=object_id)

        self.model.post_epoch()

    def evaluate(self):
        self.model.set_step_type(step_type="matching_step")

        self.model.pre_epoch()

        # only one evaluation epoch is executed for this experiment type.
        # execute all the episodes directly.
        for object_id, curvatures, coordinates in self.eval_loader:
            self.model.pre_episode()

            self.run_eval_episode(object_id, curvatures, coordinates.squeeze())

            self.model.post_episode(object_id=object_id)

        self.model.post_epoch()
        self.model.post_epoch()
