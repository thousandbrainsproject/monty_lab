# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import os

from tbp.monty.frameworks.config_utils.make_dataset_configs import ExperimentArgs
from transforms import (
    RandomRigidBody,
    RandomRotate,
    RandomTranslation,
)
from modelnet import ModelNet40
from online_optimization_dataset import (
    OnlineOptimizationDatasetMixin,
    OnlineOptimizationExactCopyMixin,
)
from metrics import (
    InverseMatrixDeviation,
    RigidBodyDisparity,
    TransformedPointCloudDistance,
)
from poisson_modelnet_40 import get_transform
from pose_estimation_models import (
    AlignVectors,
    IterativeClosestPoint,
    IterativeClosestPointScipyAlign,
    IterativeClosestPointScipyMinimize,
    PointCloudMetropolisHastings,
)

from experiments.online_optimization_experiment import OnlineRandomRigidBodyExperiment

#####
# Initial setup variables
#####

N_SAMPLES = 1024


class ModelNet40OnlineOptimizationExactCopy(OnlineOptimizationExactCopyMixin, ModelNet40):  # noqa E501
    pass


class ModelNet40OnlineOptimization(OnlineOptimizationDatasetMixin, ModelNet40):  # noqa E501
    pass


means = [2, 2, 2]
stdevs = [1, 1, 1]

rt_kwargs = dict(means=means, stdevs=stdevs, fix_translation=True)
rr_kwargs = dict(
    axes=["x"],
    means=means,
    stdevs=stdevs,
    fix_rotation=True,
    fix_translation=True)

rr_xy_kwargs = dict(
    axes=["x", "y"],
    means=means,
    stdevs=stdevs,
    fix_rotation=True,
    fix_translation=True)

rr_xyz_kwargs = dict(
    axes=["x", "y", "z"],
    means=means,
    stdevs=stdevs,
    fix_rotation=True,
    fix_translation=True)

mcmc_single_axis_rotation = dict(
    experiment_class=OnlineRandomRigidBodyExperiment,
    eval_metrics=dict(
        angle_disparity=RigidBodyDisparity(),
        inverse_matrix_deviation=InverseMatrixDeviation(),
        mean_pointwise_error=TransformedPointCloudDistance(),
    ),
    model_class=PointCloudMetropolisHastings,
    model_args=dict(
        n_steps=1_500,
        kappa=8,
        temp=0.1,
    ),
    dataset_class=ModelNet40OnlineOptimizationExactCopy,
    dataset_args=dict(
        root=os.path.expanduser("~/tbp/datasets/ModelNet40/raw"),
        transform=None,
        train=True,
        num_samples_train=5,
        dst_transform=get_transform(N_SAMPLES),
        src_transform=None,  # will be overloaded dynamically by eval scenarios
    ),
    dataloader_args=dict(batch_size=1, shuffle=True),
    experiment_args=ExperimentArgs(run_name="mcmc"),
    eval_scenarios={
        f"{i}": RandomRotate(axes=["x"], fix_rotation=True) for i in range(2)
    },
)


#####
# ICP experiments
#####

icp_x_rotation = copy.deepcopy(mcmc_single_axis_rotation)
icp_x_rotation.update(
    model_class=IterativeClosestPoint,
    model_args=dict(n_steps=20),
    experiment_args=ExperimentArgs(run_name="icp_x"),
    eval_scenarios={
        f"{i}": RandomRotate(axes=["x"], fix_rotation=True) for i in range(100)
    },
)

icp_x_rotation_1_step = copy.deepcopy(icp_x_rotation)
icp_x_rotation_1_step.update(
    model_args=dict(n_steps=1), experiment_args=ExperimentArgs(run_name="icp_x_1_step")
)

icp_x_rotation_scipy_minimize = copy.deepcopy(icp_x_rotation)
icp_x_rotation_scipy_minimize.update(
    model_class=IterativeClosestPointScipyMinimize,
    eval_scenarios={
        f"{i}": RandomRotate(axes=["x"], fix_rotation=True) for i in range(50)
    },
    experiment_args=ExperimentArgs(run_name="icp_scipy_x"),
)

# Using the scipy.align_vectors api for Kabsch algo
icp_x_rotation_align_vectors = copy.deepcopy(icp_x_rotation)
icp_x_rotation_align_vectors.update(
    model_class=IterativeClosestPointScipyAlign,
    experiment_args=ExperimentArgs(run_name="icp_align_vectors_x"),
)

icp_x_rotation_align_vectors_1_step = copy.deepcopy(icp_x_rotation_align_vectors)
icp_x_rotation_align_vectors_1_step.update(
    model_args=dict(n_steps=1),
    experiment_args=ExperimentArgs(run_name="icp_x_align_vectors_1_step"),
)

align_vectors_x_rotation = copy.deepcopy(icp_x_rotation)
align_vectors_x_rotation.update(
    model_class=AlignVectors, experiment_args=ExperimentArgs(run_name="align_vectors_x")
)

icp_xy_rotation = copy.deepcopy(icp_x_rotation)
icp_xy_rotation.update(
    experiment_args=ExperimentArgs(run_name="icp_xy"),
    eval_scenarios={
        f"{i}": RandomRotate(axes=["x", "y"], fix_rotation=True) for i in range(100)
    },
)

icp_xyz_rotation = copy.deepcopy(icp_x_rotation)
icp_xyz_rotation.update(
    experiment_args=ExperimentArgs(run_name="icp_xyz"),
    eval_scenarios={
        f"{i}": RandomRotate(axes=["x", "y", "z"], fix_rotation=True)
        for i in range(100)
    },
)

icp_rigid_body_x = copy.deepcopy(icp_x_rotation)
icp_rigid_body_x.update(
    experiment_args=ExperimentArgs(run_name="icp_rigid_body_xrot"),
    eval_scenarios={f"{i}": RandomRigidBody(**rr_kwargs) for i in range(100)},
)

icp_rigid_body_xy = copy.deepcopy(icp_x_rotation)
icp_rigid_body_xy.update(
    experiment_args=ExperimentArgs(run_name="icp_rigid_body_xyrot"),
    eval_scenarios={f"{i}": RandomRigidBody(**rr_xy_kwargs) for i in range(100)},
)

icp_rigid_body_xyz = copy.deepcopy(icp_x_rotation)
icp_rigid_body_xyz.update(
    experiment_args=ExperimentArgs(run_name="icp_rigid_body_xyzrot"),
    eval_scenarios={f"{i}": RandomRigidBody(**rr_xyz_kwargs) for i in range(100)},
)

#####
# Testing / Debugging experiments
#####

icp_rotation_test = copy.deepcopy(icp_x_rotation)
icp_rotation_test.update(
    experiment_args=ExperimentArgs(run_name="icp_debug"),
    eval_scenarios={
        f"{i}": RandomRotate(axes=["x", "y"], fix_rotation=True) for i in range(2)
    },
)

icp_translation_test = copy.deepcopy(
    icp_rotation_test)
icp_translation_test.update(
    experiment_args=ExperimentArgs(run_name="icp_debug_translation"),
    eval_scenarios={f"{i}": RandomTranslation(**rt_kwargs) for i in range(2)},
)

icp_rigid_body_test = copy.deepcopy(
    icp_rotation_test)
icp_rigid_body_test.update(
    experiment_args=ExperimentArgs(run_name="icp_debug_rigid_body"),
    eval_scenarios={f"{i}": RandomRigidBody(**rr_kwargs) for i in range(2)},
)


CONFIGS = dict(
    # MCMC, scipy minimize, scipy align_vectors
    mcmc_single_axis_rotation=mcmc_single_axis_rotation,
    icp_x_rotation_scipy_minimize=icp_x_rotation_scipy_minimize,
    icp_x_rotation_align_vectors=icp_x_rotation_align_vectors,
    align_vectors_x_rotation=align_vectors_x_rotation,
    icp_x_rotation_1_step=icp_x_rotation_1_step,
    icp_x_rotation_align_vectors_1_step=icp_x_rotation_align_vectors_1_step,
    # Debug / test runs
    icp_rotation_test=icp_rotation_test,
    icp_translation_test=icp_translation_test,
    icp_rigid_body_test=icp_rigid_body_test,
    # ICP: xy, xyz, rigid body
    icp_x_rotation=icp_x_rotation,
    icp_xy_rotation=icp_xy_rotation,
    icp_xyz_rotation=icp_xyz_rotation,
    icp_rigid_body_x=icp_rigid_body_x,
    icp_rigid_body_xy=icp_rigid_body_xy,
    icp_rigid_body_xyz=icp_rigid_body_xyz,
)
