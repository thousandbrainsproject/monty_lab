# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from copy import deepcopy

from tbp.monty.frameworks.config_utils.temporal_memory_config_args import (
    TemporalMemoryArgs,
)
from tbp.monty.frameworks.experiments import TemporalMemoryExperiment

base = deepcopy(TemporalMemoryArgs)
base.update(
    experiment_class=TemporalMemoryExperiment,
)

occlusion_cluster_by_coord_curve = deepcopy(base)
occlusion_cluster_by_coord_curve["dataset_args"].update(
    occluded=True,
    cluster_by_coord=True,
    cluster_by_curve=True
)

occlusion_cluster_by_coord = deepcopy(base)
occlusion_cluster_by_coord["dataset_args"].update(
    occluded=True,
    cluster_by_coord=True,
    cluster_by_curve=False
)

occlusion_cluster_by_curve = deepcopy(base)
occlusion_cluster_by_curve["dataset_args"].update(
    occluded=True,
    cluster_by_coord=False,
    cluster_by_curve=True,
)

no_occlusion_cluster_by_coord_curve = deepcopy(base)
no_occlusion_cluster_by_coord_curve["dataset_args"].update(
    occluded=False,
    cluster_by_coord=True,
    cluster_by_curve=True,
    test_data_size=50,
)

no_occlusion_cluster_by_coord = deepcopy(base)
no_occlusion_cluster_by_coord["dataset_args"].update(
    occluded=False,
    cluster_by_coord=True,
    cluster_by_curve=False,
    test_data_size=50,
)

no_occlusion_cluster_by_curve = deepcopy(base)
no_occlusion_cluster_by_curve["dataset_args"].update(
    occluded=False,
    cluster_by_coord=False,
    cluster_by_curve=True,
    test_data_size=50,
)


CONFIGS = dict(
    occlusion_cluster_by_coord_curve=occlusion_cluster_by_coord_curve,
    occlusion_cluster_by_coord=occlusion_cluster_by_coord,
    occlusion_cluster_by_curve=occlusion_cluster_by_curve,
    no_occlusion_cluster_by_coord_curve=no_occlusion_cluster_by_coord_curve,
    no_occlusion_cluster_by_coord=no_occlusion_cluster_by_coord,
    no_occlusion_cluster_by_curve=no_occlusion_cluster_by_curve,
)
