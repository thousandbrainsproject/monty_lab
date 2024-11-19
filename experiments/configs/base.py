# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy

from tbp.monty.frameworks.config_utils.config_args import (
    BaseMountMontyConfig,
    LoggingConfig,
    SingleCameraMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    DebugExperimentArgs,
    EnvironmentDataLoaderPerObjectEvalArgs,
    EnvironmentDataLoaderPerObjectTrainArgs,
    ExperimentArgs,
    SimpleMountHabitatDatasetArgs,
    SinglePTZHabitatDatasetArgs,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyExperiment

experiment_args = ExperimentArgs()
debug_experiment_args = DebugExperimentArgs()

single_camera_base = dict(
    experiment_class=MontyExperiment,
    experiment_args=experiment_args,
    logging_config=LoggingConfig(),
    monty_config=SingleCameraMontyConfig(),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SinglePTZHabitatDatasetArgs(),
    train_dataloader_class=ED.EnvironmentDataLoader,  # TODO: doesn't work anymore
    train_dataloader_args=dict(),
    eval_dataloader_class=ED.EnvironmentDataLoader,
    eval_dataloader_args=dict(),
)

bug_eye_base = dict(
    experiment_class=MontyExperiment,
    experiment_args=experiment_args,
    logging_config=LoggingConfig(),
    monty_config=BaseMountMontyConfig(),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SimpleMountHabitatDatasetArgs(),
    train_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(),
    eval_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(),
)

single_camera_multi_object = copy.deepcopy(single_camera_base)
single_camera_multi_object.update(
    train_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(),
    eval_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(),
)

CONFIGS = dict(
    single_camera_base=single_camera_base,
    bug_eye_base=bug_eye_base,
    single_camera_multi_object=single_camera_multi_object,
)
