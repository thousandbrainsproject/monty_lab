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

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    MontyArgs,
    MontyFeatureGraphArgs,
    PatchAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    DebugExperimentArgs,
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PredefinedObjectInitializer,
    get_env_dataloader_per_object_by_idx,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.feature_location_matching import FeatureGraphLM

from .graph_experiments import debug_camera_patch_multi_object

monty_models_dir = os.getenv("MONTY_MODELS")

if monty_models_dir is None:
    monty_models_dir = "~/tbp/results/monty/pretrained_models/"
    print(f"MONTY_MODELS not set. Using default directory: {monty_models_dir}")

model_path = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_models_ycb_high_res_for_transfer")
)

rotations = [[0.0, r, 0.0] for r in np.linspace(0, 360, 9)[:-1]]
supervised_feat_pre_training_for_transfer = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(rotations),
    ),
    logging_config=LoggingConfig(output_dir=model_path),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                ),
            )
        ),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=rotations),
    ),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
)

pretrained_feature_pred_tests = copy.deepcopy(debug_camera_patch_multi_object)
pretrained_feature_pred_tests["logging_config"].python_log_level = "DEBUG"
pretrained_feature_pred_tests.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=False,
        n_train_epochs=1,
        n_eval_epochs=len(rotations),
        model_name_or_path=os.path.join(model_path, "pretrained"),
        max_eval_steps=10,  # TODO: run experiments varying this
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=rotations),
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(num_exploratory_steps=20),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "principal_curvatures": np.ones(2) * 5,
                        "point_normal": 1.57,  # angular difference
                    },
                ),
            )
        ),
    ),
)

CONFIGS = dict(
    supervised_feat_pre_training_for_transfer=supervised_feat_pre_training_for_transfer,
    pretrained_feature_pred_tests=pretrained_feature_pred_tests,
)
