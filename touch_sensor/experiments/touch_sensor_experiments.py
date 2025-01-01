# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from classes.dataloader import TouchEnvironmentDataLoader
from classes.experiment import MontyTouchSensorExperiment
from config.config_args import MontyRunArgs, TouchAndViewMontyConfig
from config.dataset_configs import (
    BaseLoggerArgs,
    DebugExperimentArgs,
    TouchViewFinderMountHabitatDatasetArgs,
    get_env_dataloader_per_object_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.loggers.exp_logger import BaseMontyLogger

camera_patch_multi_object_dev = dict(
    experiment_class=MontyTouchSensorExperiment,
    logger_args=BaseLoggerArgs(loggers=[BaseMontyLogger()]),
    experiment_args=DebugExperimentArgs(
        do_train=True, do_eval=False, max_train_steps=60
    ),
    monty_config=TouchAndViewMontyConfig(monty_args=MontyRunArgs()),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=TouchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=TouchEnvironmentDataLoader,
    # 1~bowl, 2~can, 3~pringles, 4~thimble, 5~spoon, 9~ball, 10~bucket, 11~strawberry
    train_dataloader_args=get_env_dataloader_per_object_by_idx(
        start=1, stop=12, list_of_indices=[2, 3, 5, 9, 11]
    ),
    eval_dataloader_class=TouchEnvironmentDataLoader,
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=0),
)

CONFIGS = dict(camera_patch_multi_object_dev=camera_patch_multi_object_dev)
