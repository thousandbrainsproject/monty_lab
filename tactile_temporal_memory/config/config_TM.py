# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import copy

from nupic.research.frameworks.columns import ApicalTiebreakPairMemoryWrapper
from tbp.monty.frameworks.config_utils.config_args import TouchAndViewForTMMontyConfig
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments.ycb import YCBMeshSDRDataset10
from tbp.monty.frameworks.experiments import TactileTemporalMemoryExperiment

from .config_explore import SHUFFLED_YCB_OBJECTS_II, explore_touch, rotation

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
    experiment_args=ExperimentArgs(),
    exp_args=dict(
        overlap_threshold=5,
        show_visualization=False,
    ),
    dataset_class=YCBMeshSDRDataset10,
    dataset_args=dict(
        root="~/tbp/tbp.monty/projects/tactile_temporal_memory/tm_dataset",
        root_test="~/tbp/tbp.monty/projects/tactile_temporal_memory/tm_dataset",
        curve_hash_radius=3,
        coord_hash_radius=2,
        curve_n=1024,
        coord_n=2048,
        curve_w=7,
        coord_w=18,
        num_clusters=100,
        cluster_by_coord=True,
        cluster_by_curve=True,
        test_data_size=100,
        occluded=True,
        deterministic_inference=True,
    ),
    train_dataloader_args=dict(batch_size=1, shuffle=False),
    eval_dataloader_args=dict(batch_size=1, shuffle=False),
    every_other=4,
)

tactile_temporal_memory = copy.deepcopy(explore_touch)
tactile_temporal_memory.update(
    monty_config=TouchAndViewForTMMontyConfig(),
    experiment_class=TactileTemporalMemoryExperiment,
    experiment_args=ExperimentArgs(
        n_eval_epochs=1,
        show_sensor_output=False,
        do_train=False,
        do_eval=True,
        max_train_steps=1600,
    ),
    temporal_memory_config=TemporalMemoryArgs,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, 10, object_list=SHUFFLED_YCB_OBJECTS_II
        ),
        object_init_sampler=PredefinedObjectInitializer(rotations=rotation),
    ),
)

tactile_temporal_memory_test = copy.deepcopy(tactile_temporal_memory)
tactile_temporal_memory_test.update(
    experiment_args=ExperimentArgs(
        n_eval_epochs=1,
        show_sensor_output=True,
        do_train=False,
        do_eval=True,
        max_train_steps=200,
    ),
)

CONFIGS = dict(
    tactile_TM=tactile_temporal_memory, tactile_TM_test=tactile_temporal_memory_test
)
