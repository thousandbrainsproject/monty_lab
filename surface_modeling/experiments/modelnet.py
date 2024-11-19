# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import os
from hashlib import blake2b

import torch
import trimesh
from torch.utils.data import Dataset
from torch_geometric.utils.convert import from_trimesh


def hash_dict(kwargs):
    return blake2b(str(kwargs).encode(), digest_size=20).hexdigest()


class ModelNet40(Dataset):
    """Dataset with objects taken from ModelNet40 loaded as a mesh.

    This can be used as a regular PyTorch Dataset.

    Attributes:
        root: Path to dataset
        num_classes: Number of classes to be used. Max is 40, the number of classes
            available in ModelNet40
        num_samples_train: Number of samples to be used for training. If specified it
            will use all samples available.
        num_samples_eval: Number of samples to be used for eval. If not specified
            will use all samples available.
        evaluate_on_training_set: Whether to use the train set for evaluation. If true,
            it will use the samples stored in the train folder instead of the test
            folder.
        face_count: Number of faces desired in the resulting mesh. If set to 0 or
            None, the simplify method is not applied
        as_graph: If true, returns as a torch_geometric graph. Otherwise returns
            the object as a trimesh mesh.
        transform: Transforms to be applied to each sample.
        target_transform: Transforms to be applied to the label. Label is a scalar
            representing an idx of the object class.
        train: Determines whether to use train or test datasets. If train is set to
            False but evaluate_on_training_set is True, it will use the data in
            the train set but the all other params defined for eval (such as
            num_samples_eval).
        download: Not implemented. For compatibility
    """
    def __init__(
        self,
        root,
        cache_root=None,
        do_cache=False,
        num_classes=40,
        num_samples_train=10_000,
        num_samples_eval=10_000,
        evaluate_on_training_set=False,
        face_count=0,
        as_graph=True,
        transform=None,
        target_transform=None,
        train=True,
        download=True,
        **kwargs
    ):
        if num_classes > 40:
            num_classes = 40
            logging.warn("Setting number of classes to 40, max allowed")

        self.id_to_object = {}
        self.data = []
        self.file_names = []
        self.labels = []

        def file_is_valid(f):
            return not f.startswith((".", "README"))

        mode = "train" if (train or evaluate_on_training_set) else "test"
        num_samples = num_samples_train if train else num_samples_eval
        classes = sorted(filter(file_is_valid, [f for f in os.listdir(root)]))

        # Hash kwargs to create unique identifier for dataset version
        kwargs = dict(
            num_classes=num_classes,
            num_samples=num_samples,
            mode=mode,
            face_count=face_count,
            as_graph=as_graph,
        )
        dataset_cache_id = hash_dict(kwargs)

        # Option to load from cache
        if cache_root is not None and do_cache:
            if dataset_cache_id in os.listdir(cache_root):
                cache_path = os.path.join(cache_root, dataset_cache_id)
                logging.warn(f"Loading from cache {cache_path}")
                self.data = torch.load(os.path.join(cache_path, "data.pt"))
                self.labels = torch.load(os.path.join(cache_path, "labels.pt"))
                self.file_names = torch.load(os.path.join(cache_path, "file_names.pt"))
                do_cache = False

        for id_class, class_name in enumerate(classes):
            if id_class >= num_classes:
                break
            self.id_to_object[id_class] = class_name

        # Only loads if not loaded from cache
        if not self.data:
            for id_class, class_name in self.id_to_object.items():
                class_path = os.path.join(root, class_name, mode)
                samples = sorted(filter(file_is_valid, os.listdir(class_path)))
                for obj_id, obj_file in enumerate(samples):
                    if obj_id >= num_samples:
                        break
                    self.data.append(trimesh.load(os.path.join(class_path, obj_file)))
                    self.labels.append(id_class)
                    self.file_names.append(obj_file)

            # Simplifies mesh
            if face_count:
                for mesh_id, mesh in enumerate(self.data):
                    self.data[mesh_id] = mesh.simplify_quadratic_decimation(
                        face_count=face_count
                    )

            # Converts to graph
            if as_graph:
                for mesh_id, mesh in enumerate(self.data):
                    self.data[mesh_id] = from_trimesh(mesh)

        self.object_to_id = {v: k for k, v in self.id_to_object.items()}
        self.transform = transform
        self.target_transform = target_transform

        # May save to allow loading later
        if do_cache:
            cache_path = os.path.join(cache_root, dataset_cache_id)
            logging.warn(f"Saving to cache {cache_path}")
            os.makedirs(cache_path, exist_ok=True)
            torch.save(self.data, os.path.join(cache_path, f"data.pt"))
            torch.save(self.labels, os.path.join(cache_path, f"labels.pt"))
            torch.save(self.file_names, os.path.join(cache_path, f"file_names.pt"))

    def get_obj_name(self, obj_id):
        return self.id_to_object[obj_id]

    def get_object_id(self, object_name):
        return self.object_to_id[object_name]

    def __getitem__(self, idx):
        mesh = self.data[idx]

        if self.transform:
            mesh = self.transform(mesh)

        label = self.labels[idx]
        if self.target_transform:
            label = self.target_transform(label)

        return mesh, label

    def __len__(self):
        return len(self.data)


class PreprocessedModelNet40(Dataset):
    """Like `ModelNet40`, but assumes data has been preprocessed for loading.

    More specifically, the assumptions are:

        - `root` is a path that contains two folders: "test" and "train", and
        - each folder contains files titled "objects_%.pt" which contain lists of
          PyTorch tensors representing object pointclouds and and "labels_%.pt" which
          contains lists of corresponding labels (where "%" is a placeholder for the
          `seed` argument).

    Directory structure:
                                      root
                                      /  \
                                     /    \
                                    /      \
                                train      test
                                  |          |
                                  |          |
                         {objects_%.pt}  {objects_%.pt}
                         {labels_%.pt}    {labels_%.pt}

    The following parameters are not used by this class and only included for
    compatibility: `num_classes`, `num_samples_train`, `num_samples_eval`, `face_count`,
    `as_graph`, and `download`.

    Attributes:
        root: Path to dataset.
        seed: The seed value.
        evaluate_on_training_set: Whether to use the train set for evaluation. If
            true, it will use the samples stored in the train folder instead of the
            test folder.
        transform: A function/transform that takes a PyTorch tensor as input and
            returns a transformed version.
        target_transform: A function/transform that takes in the target and
            transforms it.
        train: Determines whether to use train or test datasets. If train is set to
            False but evaluate_on_training_set is True, it will use the data in the
            train set but the all other params defined for eval (such as
            num_samples_eval).
    """
    def __init__(
        self,
        root,
        seed,
        num_classes=None,
        num_samples_train=None,
        num_samples_eval=None,
        evaluate_on_training_set=False,
        face_count=None,
        as_graph=None,
        transform=None,
        target_transform=None,
        train=True,
        download=None,
    ):
        mode = "train" if (train or evaluate_on_training_set) else "test"

        # Load preprocessed data
        data_path = os.path.join(root, mode, f"objects_{seed}.pt")
        self.data = torch.load(data_path)

        # Load preprocessed labels
        labels_path = os.path.join(root, mode, f"labels_{seed}.pt")
        self.labels = torch.load(labels_path)

        # Data and label transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj, label = self.data[idx], self.labels[idx]

        if self.transform:
            obj = self.transform(obj)

        if self.target_transform:
            label = self.target_transform(label)

        return obj, label
