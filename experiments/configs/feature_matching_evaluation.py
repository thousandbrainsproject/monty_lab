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
    EvalLoggingConfig,
    MontyFeatureGraphArgs,
    MotorSystemConfigInformedNoTransCloser,
    MotorSystemConfigInformedNoTransFurtherAway,
    MotorSystemConfigInformedNoTransStepS3,
    MotorSystemConfigInformedNoTransStepS6,
    PatchAndViewFeatureChangeConfig,
    PatchAndViewMontyConfig,
    SurfaceAndViewMontyConfig,
    get_possible_3d_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PatchViewFinderLowResMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PredefinedObjectInitializer,
    SurfaceViewFinderMountHabitatDatasetArgs,
    get_env_dataloader_per_object_by_idx,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.feature_location_matching import FeatureGraphLM
from tbp.monty.frameworks.utils.logging_utils import get_reverse_rotation

# FOR SUPERVISED PRETRAINING
tested_degrees = np.linspace(0, 360, 5)[:-1]  # gives 32 combinations
test_rotations_all = get_possible_3d_rotations(tested_degrees)
test_rotations_y_axis = [[0.0, r, 0.0] for r in np.linspace(0, 360, 9)[:-1]]
test_rotations_shifted = [
    [0.0, (10.0 + r) % 360, 0.0] for r in np.linspace(0, 360, 9)[:-1]
]
# generated from [[x, y, z] for x, y, z in np.random.randint(0,360, size=(9,3))]
# First rotation has to be 0,0,0
random_rot = [
    [0, 0, 0],
    [59, 24, 268],
    [29, 240, 116],
    [296, 196, 273],
    [34, 186, 341],
    [160, 185, 244],
    [243, 214, 259],
    [210, 88, 296],
    [55, 175, 241],
]

monty_models_dir = os.getenv("MONTY_MODELS")

if monty_models_dir is None:
    monty_models_dir = "~/tbp/results/monty/pretrained_models/"
    print(f"MONTY_MODELS not set. Using default directory: {monty_models_dir}")

fe_pretrain_dir = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_ycb_v4")
)

# FOR TESTING
# NOTE: These will have to be retrained with the new graph format if we ever
# want to run those experiments again.
model_path = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_storeall/pretrained/",
)  # noqa E501
model_path_steps3 = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_stepsize3_storeall/pretrained/",
)  # noqa E501
model_path_feature_change = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_feature_change/pretrained/",
)
model_path_feature_change_s3 = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_feature_change_s3/pretrained/",
)
model_path_surf_agent = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_surf_agent/pretrained/",
)
model_path_all_objects = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_all_objects_storeall/pretrained/",
)
model_path_random_rot = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_random_rot_storeall/pretrained/",
)
model_path_mesh = os.path.join(
    fe_pretrain_dir, "supervised_pre_training_on_mesh/"
)
model_path_mesh_detailed = os.path.join(
    fe_pretrain_dir, "supervised_pre_training_on_mesh_detailed/"
)


test_rotations_partial = [
    [0, 0, 0],
    [0, 90, 0],
    [0, 180, 0],
]  # test_rotations_all[:3]
test_rotations_partial_shifted = [
    [0, 45, 0],
    [0, 135, 0],
    [0, 225, 0],
]  # test_rotations_shifted[:3]

default_tolerances = {
    "patch": {
        # "hsv": np.array([0.1, 1, 1]),  # only look at hue
        # "gaussian_curvature_sc": 16,  # in range [-64, 64]
        # "mean_curvature_sc": 8,  # in range [-16, 16]
        # "point_normal": 20,  # degree difference
        "hsv": np.array([0.1, 1, 1]),  # only look at hue
        "principal_curvatures_log": np.ones(2),
    }
}

alternative_tolerances = {
    "patch": {
        # "mean_curvature_sc": 4,  # in range [-16, 16]
        "principal_curvatures_log": np.ones(2) * 2,
        "hsv": np.array([0.1, 1, 1]),  # only look at hue
        "point_normal": 40,  # angular difference
    }
}

# default_tolerances = alternative_tolerances

base_config = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_all),
    ),
    logging_config=EvalLoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
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

full_rotation_eval = copy.deepcopy(base_config)
full_rotation_eval.update(
    # logging_config=EvalLoggingConfig(run_name="full_rotation_eval_5evidence"),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all
        ),
    ),
)

full_rotation_eval_all_objects = copy.deepcopy(full_rotation_eval)
full_rotation_eval_all_objects.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_all),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 78, object_list=SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
)


partial_rotation_eval_base = copy.deepcopy(full_rotation_eval)
partial_rotation_eval_base.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
    ),
    # logging_config=EvalLoggingConfig(run_name="partial_rotation_eval_base"),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

partial_rotation_eval_base_surf_agent = copy.deepcopy(partial_rotation_eval_base)
partial_rotation_eval_base_surf_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
        show_sensor_output=False,
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2) * 2,
                        }
                    },
                ),
            )
        ),
    ),
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
)

partial_rotation_eval_base_surf_agent_model = copy.deepcopy(
    partial_rotation_eval_base_surf_agent
)
partial_rotation_eval_base_surf_agent_model.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_surf_agent,
        n_eval_epochs=len(test_rotations_partial),
        show_sensor_output=False,
    ),
)


alternative_tolerances_default_setup = copy.deepcopy(partial_rotation_eval_base)
alternative_tolerances_default_setup.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=alternative_tolerances,
                ),
            )
        ),
    ),
)

partial_rotation_eval_all_obj_in_memory = copy.deepcopy(
    partial_rotation_eval_base
)
partial_rotation_eval_all_obj_in_memory.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_args=EvalLoggingConfig(),
)

partial_rotation_eval_all_obj_tested = copy.deepcopy(
    partial_rotation_eval_all_obj_in_memory
)
partial_rotation_eval_all_obj_tested.update(
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, 78, object_list=SHUFFLED_YCB_OBJECTS
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

partial_rotation_eval_on_mesh = copy.deepcopy(partial_rotation_eval_base)
partial_rotation_eval_on_mesh.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_mesh,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalLoggingConfig(
        # run_name="partial_rotation_eval_mesh_test_uniform"
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    # initial_possible_poses=list(
                    #     get_reverse_rotation(test_rotations_partial)
                    # ),
                    # path_similarity_threshold=0.05,
                ),
            )
        ),
    ),
)

partial_rotation_eval_on_mesh_high_tol = copy.deepcopy(
    partial_rotation_eval_base
)
partial_rotation_eval_on_mesh_high_tol.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_mesh,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalLoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2) * 2,
                            "point_normal": 90,
                        }
                    },
                ),
            )
        ),
    ),
)

partial_rotation_eval_on_mesh_detailed = copy.deepcopy(
    partial_rotation_eval_on_mesh
)
partial_rotation_eval_on_mesh_detailed.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_mesh_detailed,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=50,
    ),
    logging_config=EvalLoggingConfig(),
)

initial_possible_poses = get_reverse_rotation(
    [[0.0, (10.0 + r) % 360, 0.0] for r in np.linspace(0, 360, 9)[:-1]]
)

unseen_rotations = copy.deepcopy(partial_rotation_eval_base)
unseen_rotations.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,  # _rotshift10,
        n_eval_epochs=len(test_rotations_partial_shifted),
    ),
    logging_config=EvalLoggingConfig(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial_shifted
        ),
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                    initial_possible_poses=list(initial_possible_poses),
                ),
            )
        ),
    ),
)

uniform_tested_rotations = copy.deepcopy(partial_rotation_eval_base)
uniform_tested_rotations.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                    initial_possible_poses="uniform",
                ),
            )
        ),
    ),
)

untested_rotations = copy.deepcopy(partial_rotation_eval_base)
untested_rotations.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial_shifted),
    ),
    logging_config=EvalLoggingConfig(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial_shifted
        ),
    ),
)

random_same_rotations = copy.deepcopy(partial_rotation_eval_base)
random_same_rotations.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_random_rot,
        n_eval_epochs=len(random_rot),
    ),
    logging_config=EvalLoggingConfig(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=random_rot),
    ),
)

random_rot_new = [
    np.array([x, y, z], dtype=float)
    for x, y, z in np.random.randint(0, 360, size=(9, 3))
]
random_new_rotations = copy.deepcopy(partial_rotation_eval_base)
random_new_rotations.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_random_rot,
        n_eval_epochs=len(random_rot_new),
    ),
    logging_config=EvalLoggingConfig(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=random_rot_new),
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures_log": np.ones(2) * 2,
                            "point_normal": 180,  # ignore pose features
                            "curvature_directions": np.ones(2) * 180,
                        }
                    },
                    path_similarity_threshold=100,  # don't care about pose
                    pose_similarity_threshold=100,
                ),
            )
        ),
    ),
)

sampling_learns3_infs3 = copy.deepcopy(partial_rotation_eval_base)
sampling_learns3_infs3.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_steps3,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalLoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
)

sampling_learns3_infs5 = copy.deepcopy(partial_rotation_eval_base)
sampling_learns3_infs5.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_steps3,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalLoggingConfig(),
)

sampling_learns3_infs5_all_rot = copy.deepcopy(partial_rotation_eval_base)
sampling_learns3_infs5_all_rot.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_steps3,
        n_eval_epochs=len(test_rotations_all),
    ),
    logging_config=EvalLoggingConfig(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
)

# Don't run until pose is determined and ignore pose dependent features.
# Should be 100% object detection accuracy.
sampling_3_5_no_pose = copy.deepcopy(sampling_learns3_infs5)
sampling_3_5_no_pose.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            # without pose features we need to lower these to avoid
                            # time out
                            # Runs still take quite a bit longer though
                            "hsv": np.array([0.05, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                            "point_normal": 180,  # ignore pose features
                            "curvature_directions": np.ones(2) * 180,
                        }
                    },
                    path_similarity_threshold=100,  # don't care about pose
                    pose_similarity_threshold=100,
                ),
            )
        ),
    ),
)

sampling_3_5_no_pose_all_rot = copy.deepcopy(sampling_learns3_infs5_all_rot)
sampling_3_5_no_pose_all_rot.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.05, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                            "point_normal": 180,  # ignore pose features
                            "curvature_directions": np.ones(2) * 180,
                        }
                    },
                    path_similarity_threshold=100,  # don't care about pose
                    pose_similarity_threshold=100,
                ),
            )
        ),
    ),
)

sampling_3_5_no_curv_dir = copy.deepcopy(sampling_learns3_infs5)
sampling_3_5_no_curv_dir.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.05, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                            "curvature_directions": np.ones(2) * np.pi,
                        }
                    },
                ),
            )
        ),
    ),
)

sampling_learns3_infs6 = copy.deepcopy(partial_rotation_eval_base)
sampling_learns3_infs6.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_steps3,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalLoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
        motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
    ),
)

sampling_learns5_infs3 = copy.deepcopy(partial_rotation_eval_base)
sampling_learns5_infs3.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
)

sampling_learns5_infs6 = copy.deepcopy(partial_rotation_eval_base)
sampling_learns5_infs6.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
        motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
    ),
)


sampling_learnfc_infs3 = copy.deepcopy(partial_rotation_eval_base)
sampling_learnfc_infs3.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_feature_change,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalLoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
)

sampling_learnfc_infs5 = copy.deepcopy(partial_rotation_eval_base)
sampling_learnfc_infs5.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_feature_change,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalLoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
    ),
)

sampling_learnfc3_infs5 = copy.deepcopy(partial_rotation_eval_base)
sampling_learnfc3_infs5.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_feature_change_s3,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalLoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
    ),
)

sampling_learnfc_infs6 = copy.deepcopy(partial_rotation_eval_base)
sampling_learnfc_infs6.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_feature_change,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalLoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
        motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
    ),
)

sampling_learnfc_inffc = copy.deepcopy(partial_rotation_eval_base)
sampling_learnfc_inffc.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_feature_change,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=200,
    ),
    logging_config=EvalLoggingConfig(),
    monty_config=PatchAndViewFeatureChangeConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
    ),
)

sampling_learns5_inffc = copy.deepcopy(partial_rotation_eval_base)
sampling_learns5_inffc.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=200,
    ),
    logging_config=EvalLoggingConfig(),
    monty_config=PatchAndViewFeatureChangeConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
    ),
)

sampling_alt_tol = copy.deepcopy(sampling_learns5_infs6)
sampling_alt_tol.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=alternative_tolerances,
                ),
            )
        ),
        motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
    ),
)

positions_lr = [[0.0 + p, 1.5, 0.0] for p in np.linspace(-0.01, 0.01, 5)]
translation_left_right = copy.deepcopy(base_config)
translation_left_right.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(positions_lr),
    ),
    logging_config=EvalLoggingConfig(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            positions=positions_lr,
            rotations=[[0.0, 0.0, 0.0]],
        ),
    ),
)

positions_ud = [[0.0, 1.5 + p, 0.0] for p in np.linspace(-0.01, 0.01, 5)]
translation_up_down = copy.deepcopy(base_config)
translation_up_down.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(positions_ud),
    ),
    logging_config=EvalLoggingConfig(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            positions=positions_ud,
            rotations=[[0.0, 0.0, 0.0]],
        ),
    ),
)

translation_closer = copy.deepcopy(partial_rotation_eval_base)
translation_closer.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
        motor_system_config=MotorSystemConfigInformedNoTransCloser(),
    ),
)

translation_further_away = copy.deepcopy(partial_rotation_eval_base)
translation_further_away.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                ),
            )
        ),
        motor_system_config=MotorSystemConfigInformedNoTransFurtherAway(),
    ),
)

# Should not work yet
scales = [[s, s, s] for s in np.linspace(0.5, 2, 4)]
different_scales = copy.deepcopy(base_config)
different_scales.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(scales),
    ),
    logging_config=EvalLoggingConfig(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=[[0.0, 0.0, 0.0]],
            scales=scales,
        ),
    ),
)

max_match_dist05 = copy.deepcopy(partial_rotation_eval_base)
max_match_dist05.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                ),
            )
        ),
    ),
)

max_match_dist005 = copy.deepcopy(partial_rotation_eval_base)
max_match_dist005.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.005,
                    tolerances=default_tolerances,
                ),
            )
        ),
    ),
)

no_hue = copy.deepcopy(partial_rotation_eval_base)
no_hue.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "principal_curvatures_log": np.ones(2),
                    },
                ),
            )
        ),
    ),
)

scaled_curve_low_tolerance = copy.deepcopy(partial_rotation_eval_base)
scaled_curve_low_tolerance.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.1, 1, 1]),
                            "gaussian_curvature_sc": 4,
                            "mean_curvature_sc": 2,
                        }
                    },
                ),
            )
        ),
    ),
)

unscaled_curvature = copy.deepcopy(partial_rotation_eval_base)
unscaled_curvature.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.1, 1, 1]),
                            "gaussian_curvature": 256,
                            "mean_curvature": 64,
                        }
                    },
                ),
            )
        ),
    ),
)

principal_curvatures = copy.deepcopy(partial_rotation_eval_base)
principal_curvatures.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures": np.ones(2) * 10,
                        }
                    },
                ),
            )
        ),
    ),
)

scaled_pc_high_tol = copy.deepcopy(partial_rotation_eval_base)
scaled_pc_high_tol.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures_log": np.ones(2) * 3,
                        }
                    },
                ),
            )
        ),
    ),
)

point_norm_angle_05 = copy.deepcopy(partial_rotation_eval_base)
point_norm_angle_05.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                            "point_normal": 28.6,  # ~0.5 radians
                        }
                    },
                ),
            )
        ),
    ),
)

low_res_sensor_patch = copy.deepcopy(partial_rotation_eval_base)
low_res_sensor_patch.update(
    dataset_args=PatchViewFinderLowResMountHabitatDatasetArgs()
)

CONFIGS = dict(
    full_rotation_eval=full_rotation_eval,
    full_rotation_eval_all_objects=full_rotation_eval_all_objects,
    partial_rotation_eval_base=partial_rotation_eval_base,
    partial_rotation_eval_base_surf_agent=partial_rotation_eval_base_surf_agent,
    partial_rotation_eval_base_surf_agent_model=partial_rotation_eval_base_surf_agent_model,  # noqa: E501
    alternative_tolerances_default_setup=alternative_tolerances_default_setup,
    partial_rotation_eval_all_obj_in_memory=partial_rotation_eval_all_obj_in_memory,
    partial_rotation_eval_all_obj_tested=partial_rotation_eval_all_obj_tested,
    partial_rotation_eval_on_mesh=partial_rotation_eval_on_mesh,
    partial_rotation_eval_on_mesh_high_tol=partial_rotation_eval_on_mesh_high_tol,
    partial_rotation_eval_on_mesh_detailed=partial_rotation_eval_on_mesh_detailed,
    unseen_rotations=unseen_rotations,
    uniform_tested_rotations=uniform_tested_rotations,
    untested_rotations=untested_rotations,
    random_same_rotations=random_same_rotations,
    random_new_rotations=random_new_rotations,
    sampling_learns3_infs3=sampling_learns3_infs3,
    sampling_learns3_infs5=sampling_learns3_infs5,
    sampling_learns3_infs5_all_rot=sampling_learns3_infs5_all_rot,
    sampling_3_5_no_pose=sampling_3_5_no_pose,
    sampling_3_5_no_pose_all_rot=sampling_3_5_no_pose_all_rot,
    sampling_3_5_no_curv_dir=sampling_3_5_no_curv_dir,
    sampling_learns3_infs6=sampling_learns3_infs6,
    sampling_learns5_infs3=sampling_learns5_infs3,
    sampling_learns5_infs6=sampling_learns5_infs6,
    sampling_learnfc_infs3=sampling_learnfc_infs3,
    sampling_learnfc_infs5=sampling_learnfc_infs5,
    sampling_learnfc3_infs5=sampling_learnfc3_infs5,
    sampling_learnfc_infs6=sampling_learnfc_infs6,
    sampling_learnfc_inffc=sampling_learnfc_inffc,
    sampling_learns5_inffc=sampling_learns5_inffc,
    sampling_alt_tol=sampling_alt_tol,
    translation_left_right=translation_left_right,
    translation_up_down=translation_up_down,
    translation_closer=translation_closer,
    translation_further_away=translation_further_away,
    different_scales=different_scales,
    max_match_dist05=max_match_dist05,
    max_match_dist005=max_match_dist005,
    no_hue=no_hue,
    scaled_curve_low_tolerance=scaled_curve_low_tolerance,
    unscaled_curvature=unscaled_curvature,
    principal_curvatures=principal_curvatures,
    scaled_pc_high_tol=scaled_pc_high_tol,
    point_norm_angle_05=point_norm_angle_05,
    low_res_sensor_patch=low_res_sensor_patch,
)
