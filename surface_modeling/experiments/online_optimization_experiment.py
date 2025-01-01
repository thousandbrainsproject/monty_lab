# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import time
from copy import deepcopy

import numpy as np
import torch
import torchvision.transforms as T
from tbp.monty.frameworks.utils.transform_utils import find_transform_instance
from supervised_config_args import OnlineOptimizationExpArguments
from torch.utils.data import DataLoader
from transforms import RandomRigidBody, RandomRotate, RandomTranslation

# TODO: ICP might have an off-by-one index bug when finding best params
# TODO: metrics all over the dang place. It should be that the target IS the transform
# or inverse transform, and then we can comptue error using src, dst, tgt. and then we
# can have a consistent API for from_monty experiments and modelnet40, but gawsh I am
# sick of shuffling stuff around right now


class OnlineOptimizationExperiment:
    """Experiment where each iteration calls a model that does an energy minimization.

    In other words, each eval step is training a small model which is not
    stored. Therefore, the default implementation does not include a train method. The
    distinction here between model and optimizer is also blurry: an MCMC model is itself
    basically an optimizer. So let's get a list of assumptions going while I dev this:

        - No optimizer, no loss function, just model which has its own arguments
          that specify optimization algorithm, loss, etc.
        - No train dataset, loader, or train method
        - Stuff that computes error measures could be done via callbacks...?
    """

    DEFAULT_ARGS = OnlineOptimizationExpArguments

    def setup_experiment(self, config):

        self.config = deepcopy(config)
        self.compose = self.config["compose"]
        self.experiment_args = config["experiment_args"]
        self.init_dataloaders()
        self.init_model()
        self.init_eval_metrics()

    def init_dataloaders(self):

        dataset_class = self.config["dataset_class"]
        dataset_args = self.config["dataset_args"]
        self.dataset = dataset_class(**dataset_args)
        self.loader = DataLoader(self.dataset, **self.config["dataloader_args"])

    def init_model(self):

        model_class = self.config["model_class"]
        model_args = self.config["model_args"]
        self.model = model_class(**model_args)

    def init_eval_metrics(self):

        self.eval_metric_fn_dict = self.config["eval_metrics"]
        self.eval_metric_logs = dict()

    def evaluate(self):

        results = dict()
        for es_name, es_transform in self.config["eval_scenarios"].items():

            print(f".....Evaluating on scenario {es_name}.....")
            results[es_name] = self.evaluate_scenario(es_transform)

        save_dir = os.path.join(
            self.experiment_args.output_dir, self.experiment_args.run_name
        )
        print(f"Saving results to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(results, os.path.join(save_dir, "stats.pt"))

    def post_evaluate_scenario(self):
        """Convert a bunch of stuff to numpy arrays."""
        for key in self.eval_metric_fn_dict.keys():
            self.scenario_log[key] = np.array(self.scenario_log[key])

        self.scenario_log["time"] = np.array(self.scenario_log["time"])
        self.scenario_log["labels"] = np.array(self.scenario_log["labels"])
        print(f"Eval scenario took {self.scenario_log['time'].sum()}")
        print("Last few parameter estimates:")
        with np.printoptions(precision=4, suppress=True):
            for i in range(1, 6):
                print(f"{self.scenario_log['params'][-i]}")

    def pre_evaluate_scenario(self, es_transform=None):

        self.scenario_log = {key: [] for key in self.eval_metric_fn_dict.keys()}
        self.scenario_log["params"] = []
        self.scenario_log["transforms"] = []
        self.scenario_log["time"] = []
        self.scenario_log["labels"] = []
        self.scenario_step = 0
        self.base_transform = self.dataset.dst_transform

        self.es_transform = es_transform
        if self.compose:
            self.es_transform = T.Compose([self.base_transform, self.es_transform])

        self.scenario_log["transforms"] = self.es_transform
        self.dataset.src_transform = self.es_transform

    def evaluate_scenario(self, es_transform=None):

        self.pre_evaluate_scenario(es_transform=es_transform)

        for src, dst, label in self.loader:
            self.eval_step(src, dst, label)
            self.scenario_step += 1

        self.post_evaluate_scenario()

        return self.scenario_log

    def eval_step(self, src, dst, label):

        t1 = time.time()
        est = self.model(src, dst)
        t = time.time() - t1

        self.est = est
        self.best_params = self.model.best_params

        self.scenario_log["time"].append(t)
        self.scenario_log["params"].append(self.best_params)
        self.scenario_log["labels"].append(label)

    def compute_error(self, dst, out):

        for name, fn in self.eval_metric_fn_dict.items():
            metric = fn(dst, out)
            self.scenario_log[name].append(metric)

    def train(self):
        pass

    def train_epoch(self):
        pass


class OnlineOptimizationNoTransforms(OnlineOptimizationExperiment):
    """OnlineOptimizationExperiment that assumes data is already transformed.

    Just like OnlineOptimizationExperiment, except instead of applying many
    transforms to the dataset, assume the dataset contains transformed version of data
    points already. So, instead of looping over transforms / scenarios, just loop over
    the dataset directly.
    """

    def evaluate(self):

        # Skip loop over transforms. Still save as nested dict for consistency.
        results = dict()
        result = self.evaluate_scenario()
        results[0] = result

        save_dir = os.path.join(
            self.experiment_args.output_dir, self.experiment_args.run_name
        )
        print(f"Saving results to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(results, os.path.join(save_dir, "stats.pt"))

    def pre_evaluate_scenario(self, es_transform=None):

        self.scenario_log = {key: [] for key in self.eval_metric_fn_dict.keys()}
        self.scenario_log["params"] = []
        self.scenario_log["transforms"] = []
        self.scenario_log["time"] = []
        self.scenario_log["labels"] = []
        self.scenario_step = 0

    def compute_error(self, **kwargs):

        for name, fn in self.eval_metric_fn_dict.items():
            metric = fn(**kwargs)
            self.scenario_log[name].append(metric)

    def eval_step(self, src, dst, label):

        super().eval_step(src, dst, label)
        self.compute_error(
            src=src,
            dst=dst,
            est=self.est,
            # transform=self.transform_parameters,
            params=self.best_params,
            label=label
        )

    def post_evaluate_scenario(self):
        """Convert a bunch of stuff to numpy arrays."""
        # Unpack detailed monty targets
        labels = []
        euler_angles = []
        positions = []
        for sub_list in self.scenario_log["labels"]:

            labels.append(sub_list["object"])
            euler = np.array([i.numpy() for i in sub_list["euler_rotation"]])
            euler_angles.append(euler)
            pos = np.array([i.numpy() for i in sub_list["position"]])
            positions.append(pos)

        # Save target data as numpy arrays
        self.scenario_log["labels"] = np.array(labels)
        self.scenario_log["euler_angles"] = np.array(euler_angles)
        self.scenario_log["positions"] = np.array(positions)

        # Save metric data as numpy arrays
        self.scenario_log["time"] = np.array(self.scenario_log["time"])
        for key in self.eval_metric_fn_dict.keys():
            self.scenario_log[key] = np.array(self.scenario_log[key])

        print(f"Eval scenario took {self.scenario_log['time'].sum()}")


class OnlineRandomRigidBodyExperiment(OnlineOptimizationExperiment):

    ALLOWED_TRANSFORMS = (RandomRigidBody, RandomRotate, RandomTranslation)

    def pre_evaluate_scenario(self, es_transform):

        super().pre_evaluate_scenario(es_transform)
        tsfm = find_transform_instance(self.es_transform, self.ALLOWED_TRANSFORMS)
        self.transform_parameters = tsfm.params
        with np.printoptions(precision=4, suppress=True):
            print("Parameters for upcoming rigid body transform scenario: ")
            print(self.transform_parameters)

    def compute_error(self, **kwargs):

        for name, fn in self.eval_metric_fn_dict.items():
            metric = fn(**kwargs)
            self.scenario_log[name].append(metric)

    def eval_step(self, src, dst, label):

        super().eval_step(src, dst, label)
        self.compute_error(
            src=src,
            dst=dst,
            est=self.est,
            transform=self.transform_parameters,
            params=self.best_params,
            label=label
        )
