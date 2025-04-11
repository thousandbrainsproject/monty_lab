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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Union

import numpy as np
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.config_utils.config_args import (
    Dataclass,
    FiveLMMontyConfig,
    InformedPolicy,
    LoggingConfig,
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigSurface,
    PatchAndViewMontyConfig,
    TwoLMMontyConfig,
    TwoLMStackedMontyConfig,
    WandbLoggingConfig,
    get_possible_3d_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    DebugExperimentArgs,
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    OmniglotDataloaderArgs,
    OmniglotDatasetArgs,
    PredefinedObjectInitializer,
    WorldImageDataloaderArgs,
    WorldImageDatasetArgs,
    get_env_dataloader_per_object_by_idx,
    get_object_names_by_idx,
    get_omniglot_eval_dataloader,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_informed_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.evidence_matching import (
    EvidenceGraphLM,
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.feature_location_matching import FeatureGraphLM
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatDistantPatchSM,
)
from tbp.monty.frameworks.utils.logging_utils import get_reverse_rotation
from tbp.monty.simulators.habitat.configs import (
    FiveLMMountHabitatDatasetArgs,
    MultiLMMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PatchViewFinderShapenetMountHabitatDatasetArgs,
    TwoLMStackedDistantMountHabitatDatasetArgs,
    TwoLMStackedSurfaceMountHabitatDatasetArgs,
)

camera_patch_multi_object = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=ExperimentArgs(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=20)
    ),
    logging_config=LoggingConfig(),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=4),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=2, stop=6),
)

camera_patch_multi_object_same_objects = copy.deepcopy(camera_patch_multi_object)
camera_patch_multi_object_same_objects.update(
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=4),
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=4),
)

debug_camera_patch_multi_object = copy.deepcopy(camera_patch_multi_object)
debug_camera_patch_multi_object.update(
    experiment_args=DebugExperimentArgs(),
    logging_config=LoggingConfig(
        python_log_level="DEBUG",
        monty_handlers=[BasicCSVStatsHandler],
    ),
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=2),
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=2),
)

monty_models_dir = os.getenv("MONTY_MODELS")

if monty_models_dir is None:
    monty_models_dir = "~/tbp/results/monty/pretrained_models/"
    print(f"MONTY_MODELS not set. Using default directory: {monty_models_dir}")

pretrain_path = os.path.expanduser(os.path.join(monty_models_dir, "pretrained_ycb_v8"))

# When loading pretrained graphs, they need to have been learned with the same
# sensor module/transforms as the once that are used in the current experiment.
model_path = os.path.join(
    pretrain_path,
    "supervised_pre_training_base/pretrained/",
)
model_path_storeall = os.path.join(
    pretrain_path,
    "supervised_pre_training_storeall/pretrained/",
)
model_path_all_objects = os.path.join(
    pretrain_path,
    "supervised_pre_training_all_objects/pretrained/",
)
# Use for displacement testing to make sure we use same displacements as stored in model
model_path_one_view = (
    pretrain_path + "ycb_one_view/supervised_pre_training/pretrained/"
)
mesh_model_path = pretrain_path + "supervised_pre_training_on_mesh/"

model_path_omniglot = os.path.join(
    pretrain_path,
    "supervised_pre_training_on_omniglot/pretrained/",
)
model_path_omniglot_large = os.path.join(
    pretrain_path,
    "supervised_pre_training_on_omniglot_large/pretrained/",
)

multi_lm_model_path = pretrain_path + "/supervised_pre_training_2lms/pretrained/"

five_lm_10dist_obj = os.path.join(
    pretrain_path, "supervised_pre_training_5lms/pretrained/"
)

PPF_pred_tests = copy.deepcopy(debug_camera_patch_multi_object)
PPF_pred_tests.update(
    experiment_args=DebugExperimentArgs(
        n_train_epochs=2,
        # Comment in to test graph extension
        model_name_or_path=model_path_one_view,
    ),
    logging_config=LoggingConfig(
        python_log_level="DEBUG", monty_handlers=[BasicCSVStatsHandler]
    ),
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=2),
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=2),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=20),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="PPF",
                    tolerance=np.ones(4) * 0.001,
                    use_relative_len=True,
                ),
            )
        ),
    ),
)


displacement_pred_tests = copy.deepcopy(debug_camera_patch_multi_object)
displacement_pred_tests.update(
    experiment_args=DebugExperimentArgs(
        n_train_epochs=2,
        # Comment in to test graph extension
        model_name_or_path=model_path_one_view,
    ),
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=2),
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=2),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=20),
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
)

# tested_degrees = np.linspace(0, 360, 5)[:-1]  # gives 32 combinations
# test_rotations = get_possible_3d_rotations(tested_degrees)
# test_rotations = [[0, 0, 270]]
test_rotations = [[0.0, 0.0, 0.0], [0.0, 90.0, 0.0], [0.0, 180.0, 0.0]]

feature_pred_tests = copy.deepcopy(debug_camera_patch_multi_object)
feature_pred_tests.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=False,
        n_eval_epochs=len(test_rotations),
        model_name_or_path=model_path,
        max_eval_steps=200,
    ),
    logging_config=WandbLoggingConfig(
        wandb_handlers=[],
        run_name="one_LM",
        wandb_group="multi_lm_runs",
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(num_exploratory_steps=200),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.1, 1, 1]),  # only look at hue
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    # initial_possible_poses=test_rotations,
                ),
            )
        ),
    ),
)

tested_degrees = np.linspace(0, 360, 5)[:-1]  # gives 32 combinations
test_rotations = get_possible_3d_rotations(tested_degrees)

off_object_test = copy.deepcopy(feature_pred_tests)
off_object_test.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=False,
        n_eval_epochs=len(test_rotations),
        model_name_or_path=model_path,
        max_eval_steps=5,
    ),
    logging_config=LoggingConfig(
        python_log_level="DEBUG",
        monty_handlers=[
            BasicCSVStatsHandler,
        ],
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=[
            "power_drill",
            "banana",
            "skillet_lid",
            "spatula",
            "chain",
            "scissors",
        ],
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)

feature_pred_test_uniform_poses = copy.deepcopy(feature_pred_tests)
feature_pred_test_uniform_poses.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(num_exploratory_steps=200),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.005,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.1, 1, 1]),  # only look at hue
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    initial_possible_poses="uniform",
                ),
            )
        ),
    ),
)


train_rotations = [[0.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 60.0, 0.0]]
# has to be the reverse because we rotated the displacement and not the model
initial_possible_poses = get_reverse_rotation(train_rotations)

feature_pred_test_unsupervised = copy.deepcopy(feature_pred_tests)
feature_pred_test_unsupervised.update(
    experiment_args=DebugExperimentArgs(
        do_eval=False,
        do_train=True,
        n_train_epochs=len(train_rotations),
    ),
    logging_config=WandbLoggingConfig(),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 2),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(min_train_steps=5, num_exploratory_steps=400),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.005,
                    tolerances={
                        "patch": {
                            # "hsv": np.array([0.1, 1, 1]),  # only look at hue
                            # "gaussian_curvature_sc": 8,  # in range [-64, 64]
                            "mean_curvature_sc": 4,  # in range [-16, 16]
                            "pose_vectors": [10, 90, 90],  # degrees
                        }
                    },
                    initial_possible_poses=list(initial_possible_poses),
                ),
            )
        ),
    ),
)

default_evidence_lm_config = dict(
    learning_module_0=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            object_evidence_threshold=1,
            evidence_update_threshold="all",
            feature_evidence_increment=0.5,
            max_nneighbors=10,
            required_symmetry_evidence=5,
            max_match_distance=0.005,
            path_similarity_threshold=0.01,
            tolerances={
                "patch": {
                    # max angle error is pi/2 (1.57=90deg). This defines when
                    # positive evidence is added and when evidence is negative.
                    "pose_vectors": np.ones(3) * 45,
                    "hsv": np.array([0.1, 1, 1]),  # only look at hue
                    "principal_curvatures_log": np.ones(2),
                }
            },  # features where weight is not specified default weight to 1
            feature_weights={
                "hsv": np.array([1, 0, 0]),
            },
            use_multithreading=True,
            # initial_possible_poses=list([[0, 0, 0]]),
        ),
    )
)

evidence_lm_nomt_config = copy.deepcopy(default_evidence_lm_config)
evidence_lm_nomt_config["learning_module_0"]["learning_module_args"][
    "use_multithreading"
] = False

test_rotations = [[0.0, 0.0, 0.0], [0.0, 90.0, 0.0], [0.0, 180.0, 0.0]]
evidence_tests = copy.deepcopy(debug_camera_patch_multi_object)
evidence_tests.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=False,
        n_eval_epochs=len(test_rotations),
        model_name_or_path=model_path,
        max_eval_steps=50,
    ),
    logging_config=LoggingConfig(
        python_log_level="DEBUG",
        # monty_log_level="BASIC",
        # monty_handlers=[BasicCSVStatsHandler],
        wandb_handlers=[],
        # run_name="evidence_LM_debug",
        # wandb_group="evidence_lm",
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=default_evidence_lm_config,
        # motor_system_config=MotorSystemConfigInformedNoTransStepS6(),
    ),
)

# run debug experiment without multithreadin. Can be useful for debugging.
# Takes longer (with all objects in memory and 12 episodes this takes 82s compared to
# 32s with multithreading, on laptop without parallel episodes)
evidence_tests_nomt = copy.deepcopy(evidence_tests)
evidence_tests_nomt.update(
    logging_config=LoggingConfig(
        python_log_level="DEBUG",
        monty_handlers=[BasicCSVStatsHandler],
        monty_log_level="BASIC",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(
            num_exploratory_steps=10000, min_train_steps=20
        ),
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=evidence_lm_nomt_config,
    ),
)

evidence_noise_tests = copy.deepcopy(evidence_tests)
evidence_noise_tests.update(
    logging_config=WandbLoggingConfig(
        wandb_handlers=[],
        run_name="evidence_noise_tests",
        wandb_group="evidence_lm",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "hsv",
                        "principal_curvatures_log",
                    ],
                    save_raw_obs=True,
                    noise_params={
                        "features": {
                            "hsv": 0.1,  # add gaussian noise with 0.1 std
                            "principal_curvatures_log": 0.1,
                            "pose_fully_defined": 0.01,  # flip bool in 1% of cases
                            "pose_vectors": 2,  # rotate by random degrees along xyz
                        },
                        "location": 0.001,  # add gaussian noise with 0.001 std
                    },
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
        learning_module_configs=default_evidence_lm_config,
    ),
)

# Test hierarchy with voting. Here we use 3 lower level LMs (get sensory input)
# and 2 higher level LMs (get input from lower LMs). We therefor have 3 sensor
# modules which connect to LMs + a view finder which doesn't connect to an LM.
higher_level_lm_test = copy.deepcopy(evidence_tests_nomt)
higher_level_lm_test.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=True,
        n_train_epochs=1,
        n_eval_epochs=1,
        # TODO: Update path to model pretrained with hierarchy structure.
        # This currently doesn't run because the loaded graphs don't contain
        # features from the lower level LMs.
        # model_name_or_path=five_lm_10dist_obj,
        max_train_steps=200,
        max_eval_steps=200,
        min_lms_match=3,
    ),
    monty_config=FiveLMMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=2000),
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.001,
                    tolerances={
                        "patch_0": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    feature_weights={},
                    use_multithreading=False,
                ),
            ),
            learning_module_1=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.001,
                    tolerances={
                        "patch_1": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    feature_weights={},
                    use_multithreading=False,
                ),
            ),
            learning_module_2=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.001,
                    tolerances={
                        "patch_2": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    feature_weights={},
                    use_multithreading=False,
                ),
            ),
            learning_module_3=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.001,
                    tolerances={"learning_module_0": {"gaussian_curvature": 10}},
                    feature_weights={},
                    use_multithreading=False,
                ),
            ),
            learning_module_4=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.001,
                    tolerances={"learning_module_1": {"gaussian_curvature": 10}},
                    feature_weights={},
                    use_multithreading=False,
                ),
            ),
        ),
        sm_to_lm_matrix=[
            [0],  # LM0 gets input from SM0
            [1],  # LM1 gets input from SM1
            [2],  # LM2 gets input from SM2
            [],  # LM3 gets no sensory input
            [],  # LM4 gets no sensory input
        ],  # View finder not connected to lm.
        # First 3 LMs don't receive input from another LM
        # LM 3 receives input from LM 0, LM 4 from LM 1
        lm_to_lm_matrix=[
            [],  # LM0 gets no input from other LMs
            [],  # LM1 gets no input from other LMs
            [],  # LM2 gets no input from other LMs
            [0],  # LM3 gets input from LM0
            [1],
        ],  # LM4 gets input from LM1
        # All LMs on one 'level' connect to each other
        # lm_to_lm_vote_matrix=None
        lm_to_lm_vote_matrix=[
            [1, 2],  # LM0 votes with LM1 and LM2
            [0, 2],  # LM1 votes with LM0 and LM2
            [0, 1],  # LM2 votes with LM0 and LM1
            [4],  # LM3 votes with LM4
            [3],  # LM4 votes with LM3
        ],
    ),
    dataset_args=FiveLMMountHabitatDatasetArgs(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 2),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)

two_stacked_lms_config = dict(
    learning_module_0=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={
                "patch_0": {
                    "hsv": np.array([0.1, 1, 1]),
                    "principal_curvatures_log": np.ones(2),
                }
            },
            feature_weights={},
            use_multithreading=False,
            max_nneighbors=5,
            path_similarity_threshold=0.001,
            x_percent_threshold=30,
            required_symmetry_evidence=20,
            # generally would want this to be smaller than second LM
            # but setting same for now to get interesting results with YCB
            max_graph_size=0.3,
            num_model_voxels_per_dim=30,
            max_nodes_per_graph=500,
        ),
    ),
    learning_module_1=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={
                "patch_1": {
                    "hsv": np.array([0.1, 1, 1]),
                    "principal_curvatures_log": np.ones(2),
                },
                # object Id currently is an int representation of the strings
                # in the object label so we keep this tolerance high. This is
                # just until we have added a way to encode object ID with some
                # real similarity measure.
                "learning_module_0": {"object_id": 1},
            },
            feature_weights={"learning_module_0": {"object_id": 1}},
            use_multithreading=False,
            max_nneighbors=5,
            path_similarity_threshold=0.001,
            x_percent_threshold=30,
            required_symmetry_evidence=20,
            max_graph_size=0.3,
            num_model_voxels_per_dim=30,
            max_nodes_per_graph=500,
        ),
    ),
)

two_lm_distant_heterarchy = copy.deepcopy(evidence_tests)
two_lm_distant_heterarchy.update(
    experiment_args=DebugExperimentArgs(
        do_eval=False,
        do_train=True,
        n_train_epochs=2,  # len(test_rotations),
        n_eval_epochs=len(test_rotations),
        # model_name_or_path=five_lm_10dist_obj,
        max_train_steps=200,
        max_eval_steps=200,
        min_lms_match=2,
    ),
    logging_config=LoggingConfig(
        python_log_level="INFO",
        # Runs in 4min instead of 12min without detailed logging
        monty_handlers=[BasicCSVStatsHandler],
        monty_log_level="BASIC",
        # monty_handlers=[BasicCSVStatsHandler, DetailedJSONHandler],
        # monty_log_level="DETAILED",
    ),
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=10000, min_train_steps=3),
        learning_module_configs=two_stacked_lms_config,
    ),
    dataset_args=TwoLMStackedDistantMountHabitatDatasetArgs(),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 2),
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0.0, 0.0, 0.0]]),
    ),
)

two_lm_surface_heterarchy = copy.deepcopy(two_lm_distant_heterarchy)
two_lm_surface_heterarchy.update(
    dataset_args=TwoLMStackedSurfaceMountHabitatDatasetArgs(),
    monty_config=TwoLMStackedMontyConfig(
        # NOTE: When setting num_exploratory_steps smaller (like 100) we get an error
        # because the built graph only contains 4 nodes.
        # TODO: Two problems to look into:
        #   - Why do we collect so few observations? It says we are on the edge of an
        #     object and therefore making very small steps. Seems to not work very well
        #   - Fix being able to match when num_nodes < max_nneighbors -> kdtree search
        #     fills returned array with num_nodes + 1
        # Set to 1000 for debugging (will be much faster and still recognize objects).
        # Set to 10000 to learn good models of the objects.
        monty_args=MontyArgs(num_exploratory_steps=1000, min_train_steps=3),
        learning_module_configs=two_stacked_lms_config,
        motor_system_config=MotorSystemConfigSurface(),
    ),
)
two_lm_surface_heterarchy["monty_config"].sensor_module_configs["sensor_module_0"][
    "sensor_module_args"
]["surf_agent_sm"] = True

two_lm_surface_heterarchy_all_objects = copy.deepcopy(two_lm_surface_heterarchy)
two_lm_surface_heterarchy_all_objects.update(
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0.0, 0.0, 0.0]]),
    ),
)


@dataclass
class MotorSystemConfigFixed:
    """A motor system with a fixed action sequence that gets repeated."""

    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=InformedPolicy,
            policy_args=make_informed_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=ConstantSampler,
                rotation_degrees=5.0,
                file_name=Path(__file__).parent / "resources/fixed_test_actions.jsonl",
            ),
        )
    )


disp_pred_fixed_policy = copy.deepcopy(displacement_pred_tests)
disp_pred_fixed_policy.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=20),
        motor_system_config=MotorSystemConfigFixed(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=5,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                ),
            )
        ),
    )
)

ppf_pred_fixed_policy = copy.deepcopy(PPF_pred_tests)
ppf_pred_fixed_policy.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=20),
        motor_system_config=MotorSystemConfigFixed(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=5,
                    match_attribute="PPF",
                    tolerance=np.ones(4) * 0.0001,
                    use_relative_len=True,
                ),
            )
        ),
    )
)

feature_pred_fixed_policy = copy.deepcopy(feature_pred_tests)
feature_pred_fixed_policy.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(num_exploratory_steps=20),
        motor_system_config=MotorSystemConfigFixed(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.005,
                    tolerances={
                        "patch": {
                            "hsv": np.array([0.1, 1, 1]),  # only look at hue
                            "principal_curvatures_log": np.ones(2),
                            # rotation dependent features
                            "pose_vectors": [20, 1, 1],  # angular difference
                        }
                    },
                ),
            )
        ),
    )
)

evidence_fixed_policy = copy.deepcopy(evidence_tests)
evidence_fixed_policy.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        motor_system_config=MotorSystemConfigFixed(),
        learning_module_configs=default_evidence_lm_config,
    )
)

evidence_on_omniglot = copy.deepcopy(evidence_tests)
evidence_on_omniglot.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=False,
        n_eval_epochs=1,
        model_name_or_path=model_path_omniglot,
        max_eval_steps=200,
    ),
    logging_config=LoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    evidence_update_threshold="all",
                    object_evidence_threshold=0.8,
                    # xyz values are in larger range so need to increase mmd
                    max_match_distance=5,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2),
                            "pose_vectors": np.ones(3) * 45,
                        }
                    },
                    # Point normal always points up so is not usefull
                    feature_weights={
                        "patch": {
                            "pose_vectors": [0, 1, 0],
                        }
                    },
                    # We assume the letter is presented upright
                    initial_possible_poses=[[0, 0, 0]],
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "principal_curvatures_log",
                    ],
                    save_raw_obs=True,
                    # Need to set this lower since curvature is generally lower
                    pc1_is_pc2_threshold=1,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
        # motor_system_config=MotorSystemConfigInformedNoTransStepS1(),
    ),
    dataset_args=OmniglotDatasetArgs(),
    train_dataloader_class=ED.OmniglotDataLoader,
    train_dataloader_args=OmniglotDataloaderArgs(),
    eval_dataloader_class=ED.OmniglotDataLoader,
    # Using versions 1 means testing on same version of character as trained.
    # Version 2 is a new drawing of the previously seen characters. In this
    # small test setting these are 3 characters from 2 alphabets. (for alphabets
    # and characters we use the default of OmniglotDataloaderArgs)
    eval_dataloader_args=OmniglotDataloaderArgs(versions=[1, 1, 1, 1, 1, 1]),
    # eval_dataloader_args=OmniglotDataloaderArgs(versions=[2, 2, 2, 2, 2, 2]),
)

# Test 2 versions of all characters in the first alphabet. The first version is
# the same one as seen during training, the rest will be new.
evidence_on_omniglot_large = copy.deepcopy(evidence_on_omniglot)
evidence_on_omniglot_large.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=False,
        n_eval_epochs=1,
        model_name_or_path=model_path_omniglot_large,
        max_eval_steps=200,
    ),
    eval_dataloader_args=get_omniglot_eval_dataloader(
        start_at_version=0,
        alphabet_ids=[1],
        num_versions=2,
    ),
)

evidence_on_world_image = copy.deepcopy(evidence_tests_nomt)
evidence_on_world_image.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=True,
        n_train_epochs=1,
        n_eval_epochs=1,
        max_train_steps=1000,
        max_eval_steps=200,
        max_total_steps=4000,
    ),
    logging_config=LoggingConfig(
        python_log_level="INFO",
    ),
    dataset_args=WorldImageDatasetArgs(),
    train_dataloader_class=ED.SaccadeOnImageDataLoader,
    train_dataloader_args=WorldImageDataloaderArgs(scenes=[0], versions=[0]),
    eval_dataloader_class=ED.SaccadeOnImageDataLoader,
    eval_dataloader_args=WorldImageDataloaderArgs(scenes=[0, 0], versions=[0, 3]),
)

evidence_on_shapenet = copy.deepcopy(evidence_tests)
evidence_on_shapenet.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=False,
        n_eval_epochs=1,  # len(test_rotations),
        model_name_or_path=model_path,
        max_eval_steps=2,
    ),
    logging_config=LoggingConfig(),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderShapenetMountHabitatDatasetArgs(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=["a_gift_for_chineses"],
        object_init_sampler=PredefinedObjectInitializer(
            rotations=[[90, 20, 180]], scales=[[0.1, 0.1, 0.1]]
        ),
    ),
)

# ------------------------------------------------------------------------------------ #
# Multiple LMs, for voting

default_tolerances = {
    "hsv": np.array([0.1, 1, 1]),  # only look at hue
    "principal_curvatures_log": np.ones(2),
}

multi_lm_feature_pred_tests = copy.deepcopy(feature_pred_tests)
multi_lm_feature_pred_tests.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=False,
        n_eval_epochs=len(test_rotations),
        model_name_or_path=multi_lm_model_path,
        max_eval_steps=200,
    ),
    logging_config=WandbLoggingConfig(
        python_log_level="DEBUG", wandb_group="multi_lm_runs"
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
    monty_config=TwoLMMontyConfig(
        monty_args=MontyFeatureGraphArgs(num_exploratory_steps=200),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={"patch_0": default_tolerances},
                ),
            ),
            learning_module_1=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={"patch_1": default_tolerances},
                ),
            ),
        ),
    ),
    dataset_args=MultiLMMountHabitatDatasetArgs(),
)

test_rotations = [[0.0, 0.0, 0.0], [0.0, 90.0, 0.0], [0.0, 180.0, 0.0]]
five_lm_feature_matching = copy.deepcopy(multi_lm_feature_pred_tests)
five_lm_feature_matching.update(
    experiment_args=DebugExperimentArgs(
        do_eval=True,
        do_train=False,
        n_eval_epochs=len(test_rotations),
        model_name_or_path=five_lm_10dist_obj,
        max_eval_steps=200,
        min_lms_match=1,
    ),
    logging_config=WandbLoggingConfig(
        # wandb_handlers=[],
        run_name="five_LMs_loc_vote_min1done",
        wandb_group="multi_lm_runs",
    ),
    monty_config=FiveLMMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={"patch_0": default_tolerances},
                ),
            ),
            learning_module_1=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={"patch_1": default_tolerances},
                ),
            ),
            learning_module_2=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={"patch_2": default_tolerances},
                ),
            ),
            learning_module_3=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={"patch_3": default_tolerances},
                ),
            ),
            learning_module_4=dict(
                learning_module_class=FeatureGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances={"patch_4": default_tolerances},
                ),
            ),
        ),
    ),
    dataset_args=FiveLMMountHabitatDatasetArgs(),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)


CONFIGS = dict(
    camera_patch_multi_object=camera_patch_multi_object,
    camera_patch_multi_object_same_objects=camera_patch_multi_object_same_objects,
    debug_camera_patch_multi_object=debug_camera_patch_multi_object,
    PPF_pred_tests=PPF_pred_tests,
    displacement_pred_tests=displacement_pred_tests,
    feature_pred_tests=feature_pred_tests,
    off_object_test=off_object_test,
    feature_pred_test_uniform_poses=feature_pred_test_uniform_poses,
    feature_pred_test_unsupervised=feature_pred_test_unsupervised,
    evidence_tests=evidence_tests,
    evidence_tests_nomt=evidence_tests_nomt,
    evidence_noise_tests=evidence_noise_tests,
    disp_pred_fixed_policy=disp_pred_fixed_policy,
    ppf_pred_fixed_policy=ppf_pred_fixed_policy,
    feature_pred_fixed_policy=feature_pred_fixed_policy,
    evidence_fixed_policy=evidence_fixed_policy,
    evidence_on_omniglot=evidence_on_omniglot,
    evidence_on_omniglot_large=evidence_on_omniglot_large,
    evidence_on_world_image=evidence_on_world_image,
    evidence_on_shapenet=evidence_on_shapenet,
    multi_lm_feature_pred_tests=multi_lm_feature_pred_tests,
    five_lm_feature_matching=five_lm_feature_matching,
    higher_level_lm_test=higher_level_lm_test,
    two_lm_distant_heterarchy=two_lm_distant_heterarchy,
    two_lm_surface_heterarchy=two_lm_surface_heterarchy,
    two_lm_surface_heterarchy_all_objects=two_lm_surface_heterarchy_all_objects,
)
