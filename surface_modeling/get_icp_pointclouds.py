# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import pickle

import numpy as np
import torch
from poisson_modelnet_40 import get_transform
from pose_estimation_mixins import IterativeClosestPointStoreSources
from scipy.spatial.transform import Rotation

from experiments.modelnet import ModelNet40
from experiments.online_optimization_dataset import OnlineOptimizationExactCopyMixin
from experiments.transforms import RandomRotate

"""
Script for running ICP on two scenarios: a 45 degree rotation and a 180 degree rotation.
Save the point clouds at every step for visualization.
"""

###
# Initialize a bunch of stuff
###


class ModelNet40OnlineOptimizationExactCopy(OnlineOptimizationExactCopyMixin, ModelNet40):  # noqa E501
    pass


N_SAMPLES = 1024
dst_transform = get_transform(N_SAMPLES)
rot_transform = RandomRotate(axes=["x"], fix_rotation=True)
rotation_matrix = rot_transform.rotation_matrix
src_transform = rot_transform
dataset = ModelNet40OnlineOptimizationExactCopy(
    root=os.path.expanduser("~/tbp/datasets/ModelNet40/raw"),
    transform=None,  # raw torch geometric object
    train=True,
    num_samples_train=2,
    dst_transform=dst_transform,
    src_transform=rot_transform,
)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1)


###
# Hack into the dataset transforms and overwrite with simple 45 degree rotation
###

# Code for overwriting the transform
angles = [np.pi / 6.]  # aboot 45 degrees
rot_transform.angles = angles[0] if len(rot_transform.axes) == 1 else angles
rot_transform.rotation = Rotation.from_euler(rot_transform.axes, rot_transform.angles)
rot_transform.rotation_matrix = torch.from_numpy(rot_transform.rotation.as_matrix()).float()  # noqa E501
rot_transform.euler_angles = rot_transform.rotation.as_euler("xyz", degrees=True)
dataset.src_transform = rot_transform

# Make ICP model
n_steps = 20
icp = IterativeClosestPointStoreSources(n_steps=n_steps)

# Get src and dst datapoints and run ICP
for pcs in dataloader:
    src, dst, label = pcs
    icp_pc = icp(src, dst)
    break

# Save all src pointclouds in common format in single list
np_dst = dst.squeeze(dim=0).numpy()
np_sources = [src.squeeze(dim=0).numpy()]
for pc in icp.source_point_clouds:
    np_sources.append(pc[:3, :].T)


errors = [np.linalg.norm(pc - np_dst, axis=1).sum() for pc in np_sources]

###
# Repeat the experiment on a 180 degree rotation
###


# Rotate dst by 180, flip back to tensor with batch dim for consistent input to ICP
tsfm = Rotation.from_euler("x", np.pi)
bad_src = tsfm.apply(np_dst)
bad_torch_src = torch.tensor(bad_src).unsqueeze(0)

# run separate ICP model
icp2 = IterativeClosestPointStoreSources(n_steps=n_steps)
icp2(bad_torch_src, dst)

# separate list of source pointclouds
np_sources_bad = [bad_src]
for pc in icp2.source_point_clouds:
    np_sources_bad.append(pc[:3, :].T)


errors_bad = [np.linalg.norm(pc - np_dst, axis=1).sum() for pc in np_sources_bad]

results = dict(
    sources_good=np_sources,
    sources_bad=np_sources_bad,
    errors_good=errors,
    errors_bad=errors_bad,
    label=label,
    dst=np_dst
)

with open("icp_pointclouds.pkl", "wb") as f:
    pickle.dump(results, f)
