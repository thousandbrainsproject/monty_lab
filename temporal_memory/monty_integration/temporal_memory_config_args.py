# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Used to be in frameworks/config_utils/

from nupic.research.frameworks.columns import ApicalTiebreakPairMemoryWrapper
from tbp.monty.frameworks.config_utils.make_dataset_configs import RunArgs
from tbp.monty.frameworks.environments.ycb import YCBMeshSDRDataset10

TemporalMemoryArgs = dict(
    model_class=ApicalTiebreakPairMemoryWrapper,
    model_args=dict(
        proximal_n=1024,
        proximal_w=11,
        basal_n=2048,
        basal_w=30,
        apical_n=2048,
        apical_w=30,
        cells_per_column=5,
        activation_threshold=8,
        reduced_basal_threshold=8,
        initial_permanence=0.51,
        connected_permanence=0.5,
        matching_threshold=8,
        sample_size=30,
        permanence_increment=0.1,
        permanence_decrement=0.02,
        seed=42,
    ),
    run_args=RunArgs(),
    exp_args=dict(
        overlap_threshold=8,
        show_visualization=False,
    ),
    dataset_class=YCBMeshSDRDataset10,
    dataset_args=dict(
        root="~/tbp/tbp.monty/projects/finger_temporal_memory/tm_dataset",
        root_test="~/tbp/tbp.monty/projects/finger_temporal_memory/tm_dataset_test",
        curve_hash_radius=3,
        coord_hash_radius=2,
        num_clusters=100,
        cluster_by_coord=True,
        cluster_by_curve=True,
        test_data_size=100,
        occluded=True,
        deterministic_inference=True,
    ),
    train_dataloader_args=dict(batch_size=1, shuffle=False),
    eval_dataloader_args=dict(batch_size=1, shuffle=False),
)
