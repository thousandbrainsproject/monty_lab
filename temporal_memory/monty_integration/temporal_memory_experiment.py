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

import hashlib
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# turn interactive plotting off -- call plt.show() to open all figures
plt.ioff()


def overlap(a, b):
    return set(a).intersection(set(b))


def hash_for_order(curvature):
    return int(int(hashlib.md5(curvature.tobytes()).hexdigest(), 16)) / (2**128)


def hash_for_sdr(curvature):
    rng = np.random.default_rng(
        int(int(hashlib.md5(curvature.tobytes()).hexdigest(), 16))
    )
    return rng


def hash_slice_for_order(datum):
    return np.array(list(map(hash_for_order, datum)))


def hash_slice_for_sdr(datum):
    return np.array(list(map(hash_for_sdr, datum)))


class TemporalMemoryExperiment:
    """
    Intended for running Temporal Memory experiments that do not involve an embodied
    environment simulation.
    """

    def setup_experiment(self, config):
        self.config = deepcopy(config)

        self.run_args = config["run_args"]
        self.do_train = self.run_args["do_train"]
        self.do_eval = self.run_args["do_eval"]

        self.init_dataloaders()
        self.init_model()

    def init_dataloaders(self):
        # dataset class and args: assume the same class and args applies for training
        # and evaluation
        dataset_class = self.config["dataset_class"]
        dataset_args = deepcopy(self.config["dataset_args"])

        # evaluation dataset
        dataset_args.update(train=True)
        self.train_dataset = dataset_class(**dataset_args)

        # evaluation dataset
        dataset_args.update(train=False)
        self.eval_dataset = dataset_class(**dataset_args)

        self.train_loader = DataLoader(
            self.train_dataset, **self.config["train_dataloader_args"]
        )

        self.eval_loader = DataLoader(
            self.eval_dataset, **self.config["eval_dataloader_args"]
        )

    def init_model(self):
        model_class = self.config["model_class"]
        model_args = self.config["model_args"]

        self.model = model_class(**model_args)

    def get_object_id(self, object_id):
        basal_n = self.config["model_args"]["basal_n"]
        basal_w = self.config["model_args"]["basal_w"]

        id_sdr = np.random.default_rng(object_id).choice(
            basal_n, basal_w, replace=False
        )
        id_sdr.sort()

        return id_sdr

    def train(self):
        # using train() to be consistent with monty/frameworks/run.py > run(),
        # which calls train() and evaluate()
        self.train_epoch()

    def train_epoch(self):
        self.training_active_cells = []

        for object_id, coordinates, curvatures in self.train_loader:
            object_id = object_id.item()

            object_active_cells = []

            for coord, curve in zip(coordinates, curvatures):
                self.model.compute(
                    active_columns=curve.squeeze().numpy(),
                    basal_input=coord.squeeze().numpy(),
                    apical_input=self.get_object_id(object_id),
                    learn=True,
                )

                object_active_cells.append(set(self.model.get_winner_cells().tolist()))

            self.training_active_cells.append(object_active_cells)

    def evaluate(self):
        num_correct = 0

        objects = self.train_dataset.object_ids
        num_objects = len(objects)

        if not len(self.training_active_cells):
            raise Exception(
                "Experiment ran evaluate() before train(). Please run train() first."
            )

        for object_id, coordinates, curvatures in self.eval_loader:
            object_id = object_id.item()

            object_active_cells = []

            for coord, curve in zip(coordinates, curvatures):
                self.model.compute(
                    active_columns=curve.squeeze().numpy(),
                    basal_input=coord.squeeze().numpy(),
                    learn=False,
                )

                object_active_cells.append(
                    set(self.model.get_predicted_active_cells().tolist())
                )

            heatmap = np.zeros(
                (
                    num_objects,
                    len(object_active_cells),
                    len(self.training_active_cells[0]),
                ),
                dtype=np.int64,
            )

            for t in range(num_objects):
                for i in range(heatmap.shape[2]):
                    for j in range(heatmap.shape[1]):
                        num_overlap = len(
                            overlap(
                                self.training_active_cells[t][i], object_active_cells[j]
                            )
                        )

                        if num_overlap >= self.config["exp_args"]["overlap_threshold"]:
                            heatmap[t, j, i] = num_overlap

            prediction = objects[heatmap.reshape(num_objects, -1).sum(axis=1).argmax()]

            if prediction == object_id:
                num_correct += 1

            if self.config["exp_args"]["show_visualization"]:
                """
                plot heatmap to describe overlap between training observations and
                single evaluation observation
                """

                fig, axes = plt.subplots(
                    ncols=num_objects,
                    figsize=(10, 10),
                )

                if not isinstance(axes, np.ndarray):
                    axes = np.array([axes], dtype="O")

                for t in range(num_objects):
                    sns.heatmap(
                        heatmap[t, :],
                        ax=axes[t],
                        vmin=0,
                        vmax=heatmap.max(),
                        cbar=(t == (num_objects - 1)),
                    )

                    axes[t].set_xlabel("\nTrain {0}".format(objects[t]))
                    axes[t].set_title("{0}".format(int(heatmap[t, :].sum())))

                    if t == 0:
                        axes[t].set_ylabel("Evaluation Obj {0}\n".format(object_id))

                fig.suptitle("See: {0}.\nPredicted: {1}".format(object_id, prediction))

                fig.savefig(
                    os.path.join(
                        os.path.expanduser("~/tbp/results"), f"overlap-{object_id}.png"
                    )
                )

                self.train_dataset.show_object(object_id)

        eval_accuracy = num_correct / num_objects

        print("Evaluation Accuracy: {0}".format(eval_accuracy))

        if self.config["exp_args"]["show_visualization"]:
            plt.show()

        return eval_accuracy

    def pre_epoch(self):
        pass

    def post_epoch(self):
        pass

    def log(self):
        pass


class OnlineTemporalMemoryExperiment(TemporalMemoryExperiment):
    """Temporal memory experiment with a modified evaluate func
    that is meant to evaluate incoming sensations on the fly"""

    def __init__(self, config):
        super().__init__()
        curve_hash_radius = config["dataset_args"]["curve_hash_radius"]
        coord_hash_radius = config["dataset_args"]["coord_hash_radius"]
        self.add_curve = torch.arange(-curve_hash_radius, curve_hash_radius + 1)
        self.add_coord = torch.cartesian_prod(
            torch.arange(-coord_hash_radius, coord_hash_radius + 1),
            torch.arange(-coord_hash_radius, coord_hash_radius + 1),
            torch.arange(-coord_hash_radius, coord_hash_radius + 1),
        )
        self.curve_n = config["dataset_args"]["curve_n"]
        self.coord_n = config["dataset_args"]["coord_n"]
        self.curve_w = config["dataset_args"]["curve_w"]
        self.coord_w = config["dataset_args"]["coord_w"]

    def get_sdr(self, datum, add, n, w):
        datum_plus_neighbors = (
            torch.from_numpy(np.array([datum])).unsqueeze(1) + add
        ).numpy()

        hash_ordering_all = [hash_slice_for_order(datum_plus_neighbors[0])]
        hash_ordering_all = np.stack(hash_ordering_all)

        x_inds = (
            np.arange(datum_plus_neighbors.shape[0])
            .repeat(w)
            .reshape(datum_plus_neighbors.shape[0], w)
        )

        selected_curvatures_all = datum_plus_neighbors[
            x_inds, hash_ordering_all.argsort(axis=1)[:, -w:]
        ]

        sdr_slots = hash_slice_for_sdr(selected_curvatures_all[0])
        sdr_slots = [s.integers(0, n, size=1) for s in sdr_slots]
        sdr_slots = np.stack(sdr_slots).squeeze()
        return sdr_slots

    def evaluate(self, curve, coord):

        objects = self.train_dataset.object_ids
        num_objects = len(objects)

        if not len(self.training_active_cells):
            raise Exception(
                "Experiment ran evaluate() before train(). Please run train() first."
            )

        curve_sdr = self.get_sdr(curve, self.add_curve, n=self.curve_n, w=self.curve_w)
        coord_sdr = self.get_sdr(coord, self.add_coord, n=self.coord_n, w=self.coord_w)

        self.model.compute(active_columns=curve_sdr, basal_input=coord_sdr, learn=False)

        object_active_cells = [set(self.model.get_predicted_active_cells().tolist())]

        heatmap = np.zeros(
            (num_objects, len(object_active_cells), len(self.training_active_cells[0])),
            dtype=np.int64,
        )

        for t in range(num_objects):
            for i in range(heatmap.shape[2]):
                for j in range(heatmap.shape[1]):
                    num_overlap = len(
                        overlap(
                            self.training_active_cells[t][i], object_active_cells[j]
                        )
                    )

                    if num_overlap >= self.config["exp_args"]["overlap_threshold"]:
                        heatmap[t, j, i] = num_overlap

        overlap_per_object = heatmap.reshape(num_objects, -1).sum(axis=1)

        return overlap_per_object
