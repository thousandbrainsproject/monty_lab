# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os

from online_optimization_dataset import OnlineOptimizationDatasetFromMonty
from pose_estimation_models import IterativeClosestPoint
from tbp.monty.frameworks.config_utils.make_dataset_configs import ExperimentArgs
from tbp.monty.frameworks.experiments import OnlineOptimizationNoTransforms

# from tbp.monty.frameworks.utils.metrics import DisparitySrcTgt
from utils import DisparitySrcTgt

dataset_path = os.path.expanduser(
    "~/tbp/tbp.monty/projects/monty_runs/run_datasets/pretrained_feature_pred_tests"
)  # noqa E501
icp_from_monty = dict(
    experiment_class=OnlineOptimizationNoTransforms,
    compose=False,
    model_class=IterativeClosestPoint,
    model_args=dict(n_steps=20),
    dataset_class=OnlineOptimizationDatasetFromMonty,
    dataset_args=dict(root=os.path.expanduser(dataset_path)),
    dataloader_args=dict(batch_size=1, shuffle=True),
    experiment_args=ExperimentArgs(run_name="icp_from_monty_01"),
    eval_metrics=dict(disparity=DisparitySrcTgt()),
)

# align_vectors_from_monty = copy.deepcopy(icp_from_monty)
# align_vectors_from_monty.update(
#     model_class=AlignVectors,
#     experiment_args=RunArgs(run_name="av_from_monty_01"),
# )

CONFIGS = dict(
    icp_from_monty=icp_from_monty,
    # align_vectors_from_monty=align_vectors_from_monty,
)
