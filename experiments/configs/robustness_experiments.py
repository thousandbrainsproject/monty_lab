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
    MotorSystemConfigInformedNoTransStepS6,
    PatchAndViewMontyConfig,
    SurfaceAndViewMontyConfig,
    get_possible_3d_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    FiveLMMountHabitatDatasetArgs,
    NoisyPatchViewFinderMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
    SurfaceViewFinderMountHabitatDatasetArgs,
    get_env_dataloader_per_object_by_idx,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching import (
    EvidenceGraphLM,
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.feature_location_matching import FeatureGraphLM
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatDistantPatchSM,
)

tested_degrees = np.linspace(0, 360, 5)[:-1]  # gives 32 combinations
test_rotations_all = get_possible_3d_rotations(tested_degrees)

monty_models_dir = os.getenv("MONTY_MODELS")

if monty_models_dir is None:
    monty_models_dir = "~/tbp/results/monty/pretrained_models/"
    print(f"MONTY_MODELS not set. Using default directory: {monty_models_dir}")

fe_pretrain_dir = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_ycb_v8")
)

model_path = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_base/pretrained/",
)

model_path_steps3 = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_stepsize3/pretrained/",
)

model_path_location_noise_01 = os.path.join(
    fe_pretrain_dir, "supervised_pre_training_location_noise01/pretrained/"
)

model_path_location_noise_005 = os.path.join(
    fe_pretrain_dir, "supervised_pre_training_location_noise005/pretrained/"
)

model_path_location_noise_001 = os.path.join(
    fe_pretrain_dir, "supervised_pre_training_location_noise001/pretrained/"
)

model_path_all_objects = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_all_objects/pretrained/",
)

# Model that has had additional surface-agent-based learning following distant-agent
# based scanning
model_path_all_objects_augmented = os.path.join(
    fe_pretrain_dir,
    "supervised_additional_training_surf_agent/pretrained/",
)

model_path_5lms_all_objects = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_5lms_all_objects/pretrained/",
)

test_rotations_partial = [
    [0, 0, 0],
    [0, 90, 0],
    [0, 180, 0],
]
test_rotations_1 = [[90, 0, 180]]

default_tolerance_values = {
    "hsv": np.array([0.1, 1, 1]),  # only look at hue
    "principal_curvatures_log": np.ones(2),
}

default_tolerances = {
    "patch": default_tolerance_values
}  # features where weight is not specified default weight to 1
# Everything is weighted 1, except for saturation and value which are not used.
default_feature_weights = {
    "patch": {
        "hsv": np.array([1, 0, 0]),
    }
}

default_evidence_lm_config = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        max_match_distance=0.01,
        tolerances=default_tolerances,
        feature_weights=default_feature_weights,
    ),
)

default_evidence_1lm_config = dict(learning_module_0=default_evidence_lm_config)

default_feature_matching_lm_config = dict(
    learning_module_0=dict(
        learning_module_class=FeatureGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances=default_tolerances,
        ),
    )
)

base_config_partial_elm = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalEvidenceLMLoggingConfig(wandb_group="evidence_robustness_runs"),
    monty_config=PatchAndViewMontyConfig(
        learning_module_configs=default_evidence_1lm_config,
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

base_config_all_rotations_elm = copy.deepcopy(base_config_partial_elm)
base_config_all_rotations_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_all),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
)

base_config_all_objects_elm = copy.deepcopy(base_config_partial_elm)
base_config_all_objects_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_all),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
)

base_config_all_objects_1rot_elm = copy.deepcopy(base_config_partial_elm)
base_config_all_objects_1rot_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_1),
        max_eval_steps=1000,
    ),
    logging_config=EvalEvidenceLMLoggingConfig(
        python_log_level="INFO",
        # monty_handlers=[BasicCSVStatsHandler, DetailedJSONHandler],
        # wandb_handlers=[],
        # monty_log_level="SELECTIVE",
        # wandb_group="evidence_robustness_runs",
        # run_name="base_config_all_objects_1rot_10perth_elm",
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        # object_names=get_object_names_by_idx(30, 33),
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_1),
    ),
)


base_config_all_objects_3rot_elm = copy.deepcopy(base_config_partial_elm)
base_config_all_objects_3rot_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
    ),
    logging_config=EvalEvidenceLMLoggingConfig(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# ----------- ROBUSTNESS TO NEW SAMPLING -----------

sampling_learns5_infs6_elm = copy.deepcopy(base_config_partial_elm)
sampling_learns5_infs6_elm.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=default_evidence_1lm_config,
        motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
    ),
)

sampling_learns3_infs5_elm = copy.deepcopy(base_config_partial_elm)
sampling_learns3_infs5_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_steps3,
        n_eval_epochs=len(test_rotations_partial),
    ),
)

sampling_learns3_infs5_all_rot_elm = copy.deepcopy(
    base_config_all_rotations_elm
)
sampling_learns3_infs5_all_rot_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_steps3,
        n_eval_epochs=len(test_rotations_all),
    ),
)

sampling_learns5_infs6_all_rot_elm = copy.deepcopy(
    base_config_all_rotations_elm
)
sampling_learns5_infs6_all_rot_elm.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=default_evidence_1lm_config,
        motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
    ),
)

sampling_learns5_infs6_all_objects_elm = copy.deepcopy(
    base_config_all_objects_elm
)
sampling_learns5_infs6_all_objects_elm.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=default_evidence_1lm_config,
        motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
    ),
)

sampling_learns5_infs6_all_objects_1rot_elm = copy.deepcopy(
    base_config_all_objects_1rot_elm
)
sampling_learns5_infs6_all_objects_1rot_elm.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=default_evidence_1lm_config,
        motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
    ),
)

# Have to pretrain a model with all objects and step size 3 first
# sampling_learns3_infs5_all_objects_elm = copy.deepcopy(
#     base_config_all_objects_elm
# )
# sampling_learns3_infs5_all_objects_elm.update(
#     experiment_args=EvalExperimentArgs(
#         model_name_or_path=model_path_steps3_all_objects,
#         n_eval_epochs=len(test_rotations_all),
#     ),
# )

# ----------- ROBUSTNESS TO NEW OBJECT LOCATION AND ORIENTATION -----------

test_rotations_plus1 = get_possible_3d_rotations(
    tested_degrees, displacement=1  # Displacement in degrees
)
new_rotation_1deg_all_rot_elm = copy.deepcopy(base_config_all_rotations_elm)
new_rotation_1deg_all_rot_elm.update(
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_plus1
        ),
    ),
)

test_rotations_plus10 = get_possible_3d_rotations(
    tested_degrees, displacement=10  # Displacement in degrees
)
new_rotation_10deg_all_rot_elm = copy.deepcopy(base_config_all_rotations_elm)
new_rotation_10deg_all_rot_elm.update(
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_plus10
        ),
    ),
)

random_rotations_elm = copy.deepcopy(base_config_all_rotations_elm)
random_rotations_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=3,  # number of random rotations to test for each object
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

random_rotations_allobj_elm = copy.deepcopy(base_config_all_rotations_elm)
random_rotations_allobj_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=1,
    ),
    logging_config=EvalEvidenceLMLoggingConfig(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

new_location_left01_all_rot_elm = copy.deepcopy(base_config_all_rotations_elm)
new_location_left01_all_rot_elm.update(
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            positions=[[-0.01, 1.5, 0.0]],
            rotations=test_rotations_all,
        ),
    ),
)
new_location_up01_all_rot_elm = copy.deepcopy(base_config_all_rotations_elm)
new_location_up01_all_rot_elm.update(
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            positions=[[0.0, 1.5 + 0.01, 0.0]],
            rotations=test_rotations_all,
        ),
    ),
)

# ----------- ROBUSTNESS TO NEW SENSOR -----------

# Standard surface-agent policy, some objects, on a model learned with distant-agent
surf_agent_on_dist_agent_model_elm = copy.deepcopy(base_config_partial_elm)
surf_agent_on_dist_agent_model_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=4000,  # 4x max_eval_steps for corrective surface-agent
        # movements
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
                            # can't use hue as feature here.
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
    ),
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
)
# === surface-agent policies with all YCB objects, all rotations ===
# Standard surface-agent policy
surf_agent_on_dist_agent_model_all_objects_elm = copy.deepcopy(
    surf_agent_on_dist_agent_model_elm
)
surf_agent_on_dist_agent_model_all_objects_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_all),
        max_eval_steps=1000,
        max_total_steps=4000,  # 4x max_eval_steps for corrective surface-agent
        # movements
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
)

# ----------- ROBUSTNESS TO NOISY SENSOR -----------

view_finder_config = dict(
    sensor_module_class=DetailedLoggingSM,
    sensor_module_args=dict(
        sensor_module_id="view_finder",
        save_raw_obs=True,
    ),
)
default_sensor_features = [
    "on_object",
    "hsv",
    "point_normal",
    "principal_curvatures_log",
    "curvature_directions",
    "pc1_is_pc2",
]

noisy_location_001_elm = copy.deepcopy(base_config_partial_elm)
noisy_location_001_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "location": 0.001,  # add gaussian noise with 0.001 std
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_location_005_elm = copy.deepcopy(base_config_partial_elm)
noisy_location_005_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "location": 0.005,  # add gaussian noise with 0.005 std
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_location_01_elm = copy.deepcopy(base_config_partial_elm)
noisy_location_01_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "location": 0.01,  # add gaussian noise with 0.01 std
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_location_01_train_and_inf_elm = copy.deepcopy(base_config_partial_elm)
noisy_location_01_train_and_inf_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_location_noise_01,
        n_eval_epochs=len(test_rotations_partial),
    ),
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "location": 0.01,  # add gaussian noise with 0.01 std
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_location_005_train_and_inf_elm = copy.deepcopy(base_config_partial_elm)
noisy_location_005_train_and_inf_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_location_noise_005,
        n_eval_epochs=len(test_rotations_partial),
    ),
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "location": 0.005,  # add gaussian noise with 0.005 std
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_location_001_train_and_inf_elm = copy.deepcopy(base_config_partial_elm)
noisy_location_001_train_and_inf_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_location_noise_001,
        n_eval_epochs=len(test_rotations_partial),
    ),
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "location": 0.001,  # add gaussian noise with 0.001 std
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_pose_1_elm = copy.deepcopy(base_config_partial_elm)
noisy_pose_1_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "features": {
                            # rotate by random degrees (0-360) along xyz with std=1
                            "point_normal": 1,
                            "curvature_directions": 1,
                        }
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_pose_5_elm = copy.deepcopy(base_config_partial_elm)
noisy_pose_5_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "features": {
                            # rotate by random degrees along xyz with std=5 degree
                            "point_normal": 5,
                            "curvature_directions": 5,
                        }
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_pose_10_elm = copy.deepcopy(base_config_partial_elm)
noisy_pose_10_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "features": {
                            # rotate by random degrees along xyz with std=10 degree
                            "point_normal": 10,
                            "curvature_directions": 10,
                        }
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_color_01_elm = copy.deepcopy(base_config_partial_elm)
noisy_color_01_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "features": {
                            "hsv": 0.1,  # add gaussian noise with 0.1 std
                        }
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_curvlog_01_elm = copy.deepcopy(base_config_partial_elm)
noisy_curvlog_01_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "features": {
                            "principal_curvatures_log": 0.1,
                        }
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_curvlog_1_elm = copy.deepcopy(base_config_partial_elm)
noisy_curvlog_1_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "features": {
                            "principal_curvatures_log": 1,
                        }
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_pc1ispc2_1_elm = copy.deepcopy(base_config_partial_elm)
noisy_pc1ispc2_1_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "features": {
                            # flip bool in 1% of cases
                            # if pc1_is_pc2 at the first step, more possible
                            # rotations around the yz plane are initialized.
                            # if pc1_is_pc2 at following steps, curvature
                            # directions are not taken into account for the
                            # pose error.
                            "pc1_is_pc2": 0.01,
                        }
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

noisy_pc1ispc2_10_elm = copy.deepcopy(base_config_partial_elm)
noisy_pc1ispc2_10_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "features": {
                            "pc1_is_pc2": 0.1,  # flip bool in 10% of cases
                        }
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

default_all_noise_params = {
    "features": {
        "hsv": 0.1,  # add gaussian noise with 0.1 std
        "principal_curvatures_log": 0.1,
        "pc1_is_pc2": 0.01,  # flip bool in 1% of cases
        "point_normal": 2,  # rotate by random degrees along xyz
        "curvature_directions": 2,
    },
    "location": 0.002,  # add gaussian noise with 0.001 std
}

default_all_noisy_sensor_module = dict(
    sensor_module_class=HabitatDistantPatchSM,
    sensor_module_args=dict(
        sensor_module_id="patch",
        features=default_sensor_features,
        save_raw_obs=True,
        noise_params=default_all_noise_params,
    ),
)

everything_noisy_elm = copy.deepcopy(base_config_partial_elm)
everything_noisy_elm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_sensor_module,
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

# Config for quickly testing different lm parameters
test_evidence_lm_config = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        max_match_distance=0.01,
        tolerances=default_tolerances,
        feature_weights={
            "patch": {
                # "point_normal": np.ones(3) *2,
                # "curvature_directions": np.ones(6) *2,
                "hsv": np.array([1, 0, 0]),
                "principal_curvatures_log": np.ones(2),
            }
        },
        x_percent_threshold=30,
    ),
)

test_evidence_1lm_config = dict(learning_module_0=test_evidence_lm_config)

everything_noisy_all_objects_1rot_elm = copy.deepcopy(
    base_config_all_objects_1rot_elm
)
everything_noisy_all_objects_1rot_elm.update(
    # logging_config=EvalEvidenceLMLoggingConfig(
    #     run_name="everything_noisy_all_objects_1rot_seed0_elm",
    # ),
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_sensor_module,
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_evidence_1lm_config,
    ),
)

everything_noisy_allobj3rot_elm = copy.deepcopy(everything_noisy_elm)
everything_noisy_allobj3rot_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=3,
        max_eval_steps=1000,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

noisy_raw_input_elm = copy.deepcopy(base_config_partial_elm)
noisy_raw_input_elm.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                    feature_weights=default_feature_weights,
                    # If we use informed hypotheses, noise in the pose features can
                    # ruin the whole rest of the matching procedure... If we set
                    # initial_possible_poses to uniform we get 100% performance. For
                    # now we keep it at informed though so we can see room for
                    # improvement and are not so vulnerable to testing novel
                    # orientations
                    # initial_possible_poses="uniform",
                    x_percent_threshold=20,
                ),
            )
        ),
    ),
    dataset_args=NoisyPatchViewFinderMountHabitatDatasetArgs(),
)

# ================= Feature Matching LM ==========================

sampling_learns3_infs5_all_rot_fm = copy.deepcopy(base_config_all_rotations_elm)
sampling_learns3_infs5_all_rot_fm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_steps3,
        n_eval_epochs=len(test_rotations_all),
    ),
    monty_config=PatchAndViewMontyConfig(
        learning_module_configs=default_feature_matching_lm_config,
    ),
)

sampling_learns5_infs6_all_rot_fm = copy.deepcopy(base_config_all_rotations_elm)
sampling_learns5_infs6_all_rot_fm.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=default_feature_matching_lm_config,
        motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
    ),
)

noisy_location_001_fm = copy.deepcopy(base_config_partial_elm)
noisy_location_001_fm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "location": 0.001,  # add gaussian noise with 0.001 std
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_feature_matching_lm_config,
    ),
)

noisy_location_005_fm = copy.deepcopy(base_config_partial_elm)
noisy_location_005_fm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "location": 0.005,  # add gaussian noise with 0.005 std
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_feature_matching_lm_config,
    ),
)

noisy_location_01_fm = copy.deepcopy(base_config_partial_elm)
noisy_location_01_fm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "location": 0.01,  # add gaussian noise with 0.01 std
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_feature_matching_lm_config,
    ),
)

everything_noisy_fm = copy.deepcopy(base_config_all_objects_1rot_elm)
everything_noisy_fm.update(
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_sensor_module,
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=default_feature_matching_lm_config,
    ),
)

everything_noisy_allobj3rot_fm = copy.deepcopy(everything_noisy_fm)
everything_noisy_allobj3rot_fm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=3,
        max_eval_steps=1000,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

no_noise_fm = copy.deepcopy(base_config_partial_elm)
no_noise_fm.update(
    monty_config=PatchAndViewMontyConfig(
        learning_module_configs=default_feature_matching_lm_config,
    ),
)

no_noise_allobj1rot_fm = copy.deepcopy(base_config_all_objects_1rot_elm)
no_noise_allobj1rot_fm.update(
    monty_config=PatchAndViewMontyConfig(
        learning_module_configs=default_feature_matching_lm_config,
    ),
)

no_noise_allobj3rot_fm = copy.deepcopy(base_config_all_objects_3rot_elm)
no_noise_allobj3rot_fm.update(
    monty_config=PatchAndViewMontyConfig(
        learning_module_configs=default_feature_matching_lm_config,
    ),
)

random_rotations_allobj_fm = copy.deepcopy(random_rotations_allobj_elm)
random_rotations_allobj_fm.update(
    logging_config=EvalEvidenceLMLoggingConfig(
        wandb_group="evidence_robustness_runs"
    ),
    monty_config=PatchAndViewMontyConfig(
        learning_module_configs=default_feature_matching_lm_config,
    ),
)

# ======== VOTING ============
# Here we use 5 sensor patches that all move together. They differ in their
# position (slightly offset), zoom value, and resolution as specified in
# FiveLMMountConfig of FiveLMMountHabitatDatasetArgs. Each patch is connected
# to one LM. The LMs then vote with each other.

lm0_config = copy.deepcopy(default_evidence_lm_config)
lm0_config["learning_module_args"]["tolerances"] = {
    "patch_0": default_tolerance_values
}
lm1_config = copy.deepcopy(default_evidence_lm_config)
lm1_config["learning_module_args"]["tolerances"] = {
    "patch_1": default_tolerance_values
}
lm2_config = copy.deepcopy(default_evidence_lm_config)
lm2_config["learning_module_args"]["tolerances"] = {
    "patch_2": default_tolerance_values
}
lm3_config = copy.deepcopy(default_evidence_lm_config)
lm3_config["learning_module_args"]["tolerances"] = {
    "patch_3": default_tolerance_values
}
lm4_config = copy.deepcopy(default_evidence_lm_config)
lm4_config["learning_module_args"]["tolerances"] = {
    "patch_4": default_tolerance_values
}

default_5lm_lmconfig = dict(
    learning_module_0=lm0_config,
    learning_module_1=lm1_config,
    learning_module_2=lm2_config,
    learning_module_3=lm3_config,
    learning_module_4=lm4_config,
)

base_config_allobj_1_rot_5lms_elm = copy.deepcopy(base_config_partial_elm)
base_config_allobj_1_rot_5lms_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_5lms_all_objects,
        n_eval_epochs=1,
        min_lms_match=3,
    ),
    monty_config=FiveLMMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,  # has custom evidence voting method
        learning_module_configs=default_5lm_lmconfig,
    ),
    dataset_args=FiveLMMountHabitatDatasetArgs(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_1),
    ),
)

# Robustness to new sampling with 5lms voting
all_objects_1_rot_5lms_l5i6_elm = copy.deepcopy(
    base_config_allobj_1_rot_5lms_elm
)
all_objects_1_rot_5lms_l5i6_elm.update(
    monty_config=FiveLMMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,  # has custom evidence voting method
        motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
        learning_module_configs=default_5lm_lmconfig,
    ),
)

custom_evidence_lm_config = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        max_match_distance=0.01,
        tolerances=default_tolerances,
        feature_weights=default_feature_weights,
        # Use 5% threshold for noisy config to converge faster. A higher threshold
        # actually works better (i.e. 15%) but takes significantly longer.
        x_percent_threshold=5,
    ),
)

custom_lm0_config = copy.deepcopy(custom_evidence_lm_config)
custom_lm0_config["learning_module_args"]["tolerances"] = {
    "patch_0": default_tolerance_values
}
custom_lm1_config = copy.deepcopy(custom_evidence_lm_config)
custom_lm1_config["learning_module_args"]["tolerances"] = {
    "patch_1": default_tolerance_values
}
custom_lm2_config = copy.deepcopy(custom_evidence_lm_config)
custom_lm2_config["learning_module_args"]["tolerances"] = {
    "patch_2": default_tolerance_values
}
custom_lm3_config = copy.deepcopy(custom_evidence_lm_config)
custom_lm3_config["learning_module_args"]["tolerances"] = {
    "patch_3": default_tolerance_values
}
custom_lm4_config = copy.deepcopy(custom_evidence_lm_config)
custom_lm4_config["learning_module_args"]["tolerances"] = {
    "patch_4": default_tolerance_values
}

custom_5lm_lmconfig = dict(
    learning_module_0=custom_lm0_config,
    learning_module_1=custom_lm1_config,
    learning_module_2=custom_lm2_config,
    learning_module_3=custom_lm3_config,
    learning_module_4=custom_lm4_config,
)

random_rotations_allobj_5lms_elm = copy.deepcopy(random_rotations_allobj_elm)
random_rotations_allobj_5lms_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_5lms_all_objects,
        n_eval_epochs=1,
        min_lms_match=3,
    ),
    monty_config=FiveLMMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,  # has custom evidence voting method
        learning_module_configs=custom_5lm_lmconfig,
    ),
    dataset_args=FiveLMMountHabitatDatasetArgs(),
)

allobj_1rot_5lms_noisy_elm = copy.deepcopy(base_config_allobj_1_rot_5lms_elm)
allobj_1rot_5lms_noisy_elm.update(
    # logging_config=EvalEvidenceLMLoggingConfig(
    #     run_name="allobj_1rot_5lms_noisy_color5_elm",
    # ),
    monty_config=FiveLMMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,  # has custom evidence voting method
        learning_module_configs=custom_5lm_lmconfig,
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_0",
                    features=default_sensor_features,
                    save_raw_obs=False,
                    noise_params=default_all_noise_params,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_1",
                    features=default_sensor_features,
                    save_raw_obs=False,
                    noise_params=default_all_noise_params,
                ),
            ),
            sensor_module_2=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_2",
                    features=default_sensor_features,
                    save_raw_obs=False,
                    noise_params=default_all_noise_params,
                ),
            ),
            sensor_module_3=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_3",
                    features=default_sensor_features,
                    save_raw_obs=False,
                    noise_params=default_all_noise_params,
                ),
            ),
            sensor_module_4=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_4",
                    features=default_sensor_features,
                    save_raw_obs=False,
                    noise_params=default_all_noise_params,
                ),
            ),
            sensor_module_5=view_finder_config,
        ),
    ),
)

all_objects_1rot_augmodels_elm = copy.deepcopy(base_config_all_objects_1rot_elm)
all_objects_1rot_augmodels_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented,
        n_eval_epochs=len(test_rotations_1),
        max_eval_steps=1000,
    ),
    # logging_config=EvalEvidenceLMLoggingConfig(
    #     python_log_level="INFO",
    #     run_name="all_objects_1rot_augmodels_30th_elm",
    # ),
    monty_config=PatchAndViewMontyConfig(
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                    feature_weights=default_feature_weights,
                    x_percent_threshold=20,
                ),
            )
        ),
    ),
)
everything_noisy_allobj_1rot_augmodels_elm = copy.deepcopy(
    everything_noisy_all_objects_1rot_elm
)
everything_noisy_allobj_1rot_augmodels_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented,
        n_eval_epochs=len(test_rotations_1),
        max_eval_steps=1000,
    ),
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_sensor_module,
            sensor_module_1=view_finder_config,
        ),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                    feature_weights=default_feature_weights,
                    x_percent_threshold=20,
                ),
            )
        ),
    ),
)

# testing new sampling on the augmented models (learned with distant-agent and
# surface-agent)
sampling56_allobj_1rot_augmodels_elm = copy.deepcopy(
    sampling_learns5_infs6_all_objects_1rot_elm
)
sampling56_allobj_1rot_augmodels_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented,
        n_eval_epochs=len(test_rotations_1),
        max_eval_steps=1000,
    ),
)

randomrot_allobj_1rot_augmodels_elm = copy.deepcopy(random_rotations_allobj_elm)
randomrot_allobj_1rot_augmodels_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented,
        n_eval_epochs=len(test_rotations_1),
        max_eval_steps=1000,
    ),
)

# Configs with # * are part of the current readme evaluation stats
CONFIGS = dict(
    base_config_partial_elm=base_config_partial_elm,
    base_config_all_rotations_elm=base_config_all_rotations_elm,
    base_config_all_objects_elm=base_config_all_objects_elm,
    base_config_all_objects_1rot_elm=base_config_all_objects_1rot_elm,  # *
    base_config_all_objects_3rot_elm=base_config_all_objects_3rot_elm,
    sampling_learns5_infs6_elm=sampling_learns5_infs6_elm,
    sampling_learns3_infs5_elm=sampling_learns3_infs5_elm,
    sampling_learns3_infs5_all_rot_elm=sampling_learns3_infs5_all_rot_elm,
    sampling_learns5_infs6_all_rot_elm=sampling_learns5_infs6_all_rot_elm,
    sampling_learns5_infs6_all_objects_elm=sampling_learns5_infs6_all_objects_elm,
    sampling_learns5_infs6_all_objects_1rot_elm=sampling_learns5_infs6_all_objects_1rot_elm,  # *  # noqa E501
    # sampling_learns3_infs5_all_objects_elm=sampling_learns3_infs5_all_objects_elm,
    new_rotation_1deg_all_rot_elm=new_rotation_1deg_all_rot_elm,
    new_rotation_10deg_all_rot_elm=new_rotation_10deg_all_rot_elm,
    random_rotations_elm=random_rotations_elm,
    random_rotations_allobj_elm=random_rotations_allobj_elm,  # *
    new_location_left01_all_rot_elm=new_location_left01_all_rot_elm,
    new_location_up01_all_rot_elm=new_location_up01_all_rot_elm,
    surf_agent_on_dist_agent_model_elm=surf_agent_on_dist_agent_model_elm,  # *
    surf_agent_on_dist_agent_model_all_objects_elm=surf_agent_on_dist_agent_model_all_objects_elm,  # noqa E501
    noisy_location_001_elm=noisy_location_001_elm,
    noisy_location_005_elm=noisy_location_005_elm,
    noisy_location_01_elm=noisy_location_01_elm,
    noisy_location_01_train_and_inf_elm=noisy_location_01_train_and_inf_elm,
    noisy_location_005_train_and_inf_elm=noisy_location_005_train_and_inf_elm,
    noisy_location_001_train_and_inf_elm=noisy_location_001_train_and_inf_elm,
    noisy_pose_1_elm=noisy_pose_1_elm,
    noisy_pose_5_elm=noisy_pose_5_elm,
    noisy_pose_10_elm=noisy_pose_10_elm,
    noisy_color_01_elm=noisy_color_01_elm,
    noisy_curvlog_01_elm=noisy_curvlog_01_elm,
    noisy_curvlog_1_elm=noisy_curvlog_1_elm,
    noisy_pc1ispc2_1_elm=noisy_pc1ispc2_1_elm,
    noisy_pc1ispc2_10_elm=noisy_pc1ispc2_10_elm,
    everything_noisy_elm=everything_noisy_elm,  # *
    everything_noisy_all_objects_1rot_elm=everything_noisy_all_objects_1rot_elm,  # *
    everything_noisy_allobj3rot_elm=everything_noisy_allobj3rot_elm,
    noisy_raw_input_elm=noisy_raw_input_elm,  # *
    # --------------------Augmented models----------------------
    all_objects_1rot_augmodels_elm=all_objects_1rot_augmodels_elm,
    everything_noisy_allobj_1rot_augmodels_elm=everything_noisy_allobj_1rot_augmodels_elm,  # noqa E501
    sampling56_allobj_1rot_augmodels_elm=sampling56_allobj_1rot_augmodels_elm,
    randomrot_allobj_1rot_augmodels_elm=randomrot_allobj_1rot_augmodels_elm,
    # ------------- Feature Matching (Old LM) ------------
    sampling_learns3_infs5_all_rot_fm=sampling_learns3_infs5_all_rot_fm,
    sampling_learns5_infs6_all_rot_fm=sampling_learns5_infs6_all_rot_fm,
    noisy_location_001_fm=noisy_location_001_fm,
    noisy_location_005_fm=noisy_location_005_fm,
    noisy_location_01_fm=noisy_location_01_fm,
    everything_noisy_fm=everything_noisy_fm,
    everything_noisy_allobj3rot_fm=everything_noisy_allobj3rot_fm,
    no_noise_fm=no_noise_fm,
    no_noise_allobj1rot_fm=no_noise_allobj1rot_fm,
    no_noise_allobj3rot_fm=no_noise_allobj3rot_fm,
    random_rotations_allobj_fm=random_rotations_allobj_fm,
    # ------------- Voting (5 LMs) -------------
    base_config_allobj_1_rot_5lms_elm=base_config_allobj_1_rot_5lms_elm,  # *
    random_rotations_allobj_5lms_elm=random_rotations_allobj_5lms_elm,  # *
    all_objects_1_rot_5lms_l5i6_elm=all_objects_1_rot_5lms_l5i6_elm,  # *
    allobj_1rot_5lms_noisy_elm=allobj_1rot_5lms_noisy_elm,  # *
)
