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
import os

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    EvalEvidenceLMLoggingConfig,
    FiveLMMontyConfig,
    MontyFeatureGraphArgs,
    PatchAndViewMontyConfig,
    SurfaceAndViewMontyConfig,
    get_possible_3d_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    get_env_dataloader_per_object_by_idx,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments import (
    MontyGeneralizationExperiment,
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.evidence_matching import (
    EvidenceGraphLM,
    MontyForEvidenceGraphMatching,
)
from tbp.monty.simulators.habitat.configs import (
    FiveLMMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
)

# FOR SUPERVISED PRETRAINING
tested_degrees = np.linspace(0, 360, 5)[:-1]  # gives 32 combinations
test_rotations_all = get_possible_3d_rotations(tested_degrees)
test_rotations_8 = [
    [0, 0, 90],
    [0, 90, 0],
    [90, 0, 0],
    [0, 180, 0],
    [0, 270, 270],
    [90, 0, 180],
    [90, 180, 180],
    [90, 270, 90],
]
test_rotations_1 = [[90, 0, 180]]
test_rotations_1_new = [[45, 0, 180]]

monty_models_dir = os.getenv("MONTY_MODELS")

if monty_models_dir is None:
    monty_models_dir = "~/tbp/results/monty/pretrained_models/"
    print(f"MONTY_MODELS not set. Using default directory: {monty_models_dir}")

# v6 : Using TLS for point-normal estimation
# v7 : Updated for State class support + using new feature names like pose_vectors
# v8 : Using separate graph per input channel
fe_pretrain_dir = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_ycb_v8")
)

model_path = os.path.join(
    fe_pretrain_dir, "supervised_pre_training_base/pretrained/"
)
model_path_steps3 = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_stepsize3/pretrained/",
)
model_path_surf_agent = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_surf_agent/pretrained/",
)
model_path_all_objects = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_all_objects/pretrained/",
)
model_path_5lms_all_objects = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_5lms_all_objects/pretrained/",
)
model_path_mesh = os.path.join(
    fe_pretrain_dir, "supervised_pre_training_on_mesh/"
)
model_path_5lms = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_5lms/pretrained/",
)

test_rotations_partial = [
    [0, 0, 0],
    [0, 90, 0],
    [0, 180, 0],
]

default_tolerance_values = {
    "hsv": np.array([0.1, 1, 1]),  # only look at hue
    # principal_curvatures_log are mostly between -7 and 7.
    "principal_curvatures_log": np.ones(2),
}

default_tolerances = {"patch": default_tolerance_values}
# Everything is weighted 1, except for saturation and value which are not used.
default_feature_weights = {
    "patch": {
        "hsv": np.array([1, 0, 0]),
    }
}

base_config_elm = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_all),
    ),
    logging_config=EvalEvidenceLMLoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                    feature_weights=default_feature_weights,
                ),
            )
        ),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=4),
)

full_rotation_eval_elm = copy.deepcopy(base_config_elm)
full_rotation_eval_elm.update(
    # logging_config=EvalEvidenceLMLoggingConfig(
    #     run_name="full_rotation_eval_to1000_fi1"
    # ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all
        ),
    ),
)

full_rotation_eval_all_objects_elm = copy.deepcopy(full_rotation_eval_elm)
full_rotation_eval_all_objects_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_all),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 78, object_list=SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
)

all_objects_8_rotations_elm = copy.deepcopy(base_config_elm)
all_objects_8_rotations_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_8),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 78, object_list=SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_8),
    ),
)

all_objects_1_rotation_elm = copy.deepcopy(base_config_elm)
all_objects_1_rotation_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=1,
    ),
    logging_config=EvalEvidenceLMLoggingConfig(
        # monty_handlers=[BasicCSVStatsHandler, DetailedJSONHandler],
        # wandb_handlers=[],
        # monty_log_level="SELECTIVE",
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 78, object_list=SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_1),
    ),
)


partial_rotation_eval_base_elm = copy.deepcopy(full_rotation_eval_elm)
partial_rotation_eval_base_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
    ),
    # logging_config=EvalEvidenceLMLoggingConfig(run_name="weight_features_1"),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

partial_rotation_eval_base_surf_agent_elm = copy.deepcopy(
    partial_rotation_eval_base_elm
)
partial_rotation_eval_base_surf_agent_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
        show_sensor_output=False,
        max_eval_steps=4000,  # 4x larger than normal since we need 4 actions each step
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    feature_weights=dict(),
                ),
            )
        ),
    ),
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
)

full_rotation_eval_all_objects_surf_agent_elm = copy.deepcopy(
    partial_rotation_eval_base_surf_agent_elm
)
full_rotation_eval_all_objects_surf_agent_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_all),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 78, object_list=SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
)

partial_rotation_eval_base_surf_agent_model_elm = copy.deepcopy(
    partial_rotation_eval_base_surf_agent_elm
)
partial_rotation_eval_base_surf_agent_model_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_surf_agent,
        n_eval_epochs=len(test_rotations_partial),
        show_sensor_output=False,
    ),
)

partial_rotation_eval_on_mesh_elm = copy.deepcopy(
    partial_rotation_eval_base_elm
)
partial_rotation_eval_on_mesh_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_mesh,
        n_eval_epochs=len(test_rotations_partial),
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    feature_weights=dict(),
                ),
            )
        ),
    ),
)

default_5lm_config = dict(
    learning_module_0=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={"patch_0": default_tolerance_values},
            feature_weights={
                "patch_0": {
                    "hsv": np.array([1, 0, 0]),
                },
            },
        ),
    ),
    learning_module_1=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={"patch_1": default_tolerance_values},
            feature_weights={
                "patch_1": {
                    "hsv": np.array([1, 0, 0]),
                },
            },
        ),
    ),
    learning_module_2=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={"patch_2": default_tolerance_values},
            feature_weights={
                "patch_2": {
                    "hsv": np.array([1, 0, 0]),
                },
            },
        ),
    ),
    learning_module_3=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={"patch_3": default_tolerance_values},
            feature_weights={
                "patch_3": {
                    "hsv": np.array([1, 0, 0]),
                },
            },
        ),
    ),
    learning_module_4=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={"patch_4": default_tolerance_values},
            feature_weights={
                "patch_4": {
                    "hsv": np.array([1, 0, 0]),
                },
            },
        ),
    ),
)

partial_rotation_eval_5lms_elm = copy.deepcopy(partial_rotation_eval_base_elm)
partial_rotation_eval_5lms_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_5lms,
        n_eval_epochs=len(test_rotations_partial),
        min_lms_match=3,
    ),
    logging_config=EvalEvidenceLMLoggingConfig(
        # wandb_handlers=[],
        monty_log_level="BASIC",
        python_log_level="INFO",
    ),
    monty_config=FiveLMMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,  # has custom evidence voting method
        learning_module_configs=default_5lm_config,
    ),
    dataset_args=FiveLMMountHabitatDatasetArgs(),
)

all_objects_1_rot_5lms_elm = copy.deepcopy(partial_rotation_eval_5lms_elm)
all_objects_1_rot_5lms_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_5lms_all_objects,
        n_eval_epochs=len(test_rotations_1),
        min_lms_match=3,
        max_eval_steps=1000,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 78, object_list=SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_1),
    ),
)

one_to_all_connected_5lm_allobj1r_elm = copy.deepcopy(
    all_objects_1_rot_5lms_elm
)
one_to_all_connected_5lm_allobj1r_elm.update(
    monty_config=FiveLMMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,  # has custom evidence voting method
        learning_module_configs=default_5lm_config,
        # First patch sends voted to everyone and receives all votes but other
        # patches don't vote among each other.
        lm_to_lm_vote_matrix=[
            [1, 2, 3, 4],
            [0],
            [0],
            [0],
            [0],
        ],
    ),
)

evidence_generalization = copy.deepcopy(all_objects_1_rotation_elm)
evidence_generalization.update(
    experiment_class=MontyGeneralizationExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=1,
        max_eval_steps=100,
    ),
    logging_config=EvalEvidenceLMLoggingConfig(
        wandb_handlers=[],
        wandb_group="evidence_lm",
        run_name="evidence_generalization_maxS100",
    ),
)

CONFIGS = dict(
    full_rotation_eval_elm=full_rotation_eval_elm,
    full_rotation_eval_all_objects_elm=full_rotation_eval_all_objects_elm,
    all_objects_8_rotations_elm=all_objects_8_rotations_elm,
    all_objects_1_rotation_elm=all_objects_1_rotation_elm,
    partial_rotation_eval_base_elm=partial_rotation_eval_base_elm,
    partial_rotation_eval_base_surf_agent_elm=partial_rotation_eval_base_surf_agent_elm,
    full_rotation_eval_all_objects_surf_agent_elm=full_rotation_eval_all_objects_surf_agent_elm,  # noqa E501
    partial_rotation_eval_base_surf_agent_model_elm=partial_rotation_eval_base_surf_agent_model_elm,  # noqa E501
    partial_rotation_eval_on_mesh_elm=partial_rotation_eval_on_mesh_elm,
    partial_rotation_eval_5lms_elm=partial_rotation_eval_5lms_elm,
    all_objects_1_rot_5lms_elm=all_objects_1_rot_5lms_elm,
    one_to_all_connected_5lm_allobj1r_elm=one_to_all_connected_5lm_allobj1r_elm,
    evidence_generalization=evidence_generalization,
)
