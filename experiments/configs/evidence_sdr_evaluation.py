# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import os

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    EvalEvidenceLMLoggingConfig,
    FiveLMMontyConfig,
    LoggingConfig,
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    ParallelEvidenceLMLoggingConfig,
    PatchAndViewMontyConfig,
    SurfaceAndViewSOTAMontyConfig,
    TwoLMStackedMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    DebugExperimentArgs,
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    FiveLMMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
    SurfaceViewFinderMountHabitatDatasetArgs,
    TwoLMStackedDistantMountHabitatDatasetArgs,
    get_env_dataloader_per_object_by_idx,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.models.evidence_matching import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.evidence_sdr_matching import EvidenceSDRGraphLM
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
)

"""
This is a subset of the benchmark experiment and evidence evaluations
with modified configs for using the EvidenceSDRGraphLM which uses the
EvidenceSDRMixin to encode similarity scores into SDRs.

When using the EvidenceSDRGraphLM instead of EvidenceGraphLM, make sure
to include the sdr_args parameters in the learning module args.

NOTE: These configs are provided as an example for how to use `EvidenceSDRGraphLM`
and the performances may not match the benchmark experiments performances since there
are additional changes (beyond `EvidenceSDRGraphLM`), such as the chosen rotations, etc.
"""


# === Default and Reusable Configs === #
monty_models_dir = os.getenv("MONTY_MODELS")
monty_logs_dir = os.getenv("MONTY_LOGS")

if monty_models_dir is None:
    monty_models_dir = "~/tbp/results/monty/pretrained_models/"
    print(f"MONTY_MODELS not set. Using default directory: {monty_models_dir}")

if monty_logs_dir is None:
    monty_logs_dir = "~/tbp/results/monty/"
    print(f"MONTY_LOGS not set. Using default directory: {monty_logs_dir}")

fe_pretrain_dir = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_ycb_v8")
)

default_tolerance_values = {
    "hsv": np.array([0.1, 1, 1]),  # only look at hue
    "principal_curvatures_log": np.ones(2),
}
default_tolerances = {"patch": default_tolerance_values}


default_feature_weights = {
    "patch": {
        "hsv": np.array([1, 0, 0]),
    }
}


default_sensor_features_surf_agent = [
    "pose_vectors",
    "pose_fully_defined",
    "on_object",
    "object_coverage",
    "min_depth",
    "mean_depth",
    "hsv",
    "principal_curvatures",
    "principal_curvatures_log",
]


default_all_noise_params = {
    "features": {
        "pose_vectors": 2,  # rotate by random degrees along xyz
        "hsv": 0.1,  # add gaussian noise with 0.1 std
        "principal_curvatures_log": 0.1,
        "pose_fully_defined": 0.01,  # flip bool in 1% of cases
    },
    "location": 0.002,  # add gaussian noise with 0.002 std
}


default_all_noisy_surf_agent_sensor_module = dict(
    sensor_module_class=FeatureChangeSM,
    sensor_module_args=dict(
        sensor_module_id="patch",
        features=default_sensor_features_surf_agent,
        save_raw_obs=False,
        delta_thresholds={
            "on_object": 0,
            "distance": 0.01,
        },
        surf_agent_sm=True,
        noise_params=default_all_noise_params,
    ),
)


def get_evidence_surf_1lm_config(max_nneighbors):

    return dict(
        learning_module_class=EvidenceSDRGraphLM,
        learning_module_args=dict(
            # mmd of 0.015 get higher performance but slower run time
            max_match_distance=0.01,  # =1cm
            tolerances=default_tolerances,
            feature_weights=default_feature_weights,
            # smaller threshold reduces runtime but also performance
            x_percent_threshold=20,
            # Using a smaller max_nneighbors (5 instead of 10) makes runtime faster,
            # but reduces performance a bit
            max_nneighbors=max_nneighbors,
            # Use this to update all hypotheses at every step as previously
            # evidence_update_threshold="all",
            # Use this to update all hypotheses > x_percent_threshold (faster)
            evidence_update_threshold="x_percent_threshold",
            # use_multithreading=False,
            # NOTE: Currently not used when loading pretrained graphs.
            max_graph_size=0.3,  # 30cm
            num_model_voxels_per_dim=100,
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=dict(
                goal_tolerances=dict(
                    location=0.015,  # distance in meters
                ),  # Tolerance(s) when determining goal-state success
                elapsed_steps_factor=10,  # Factor that considers the number of elapsed
                # steps as a possible condition for initiating a hypothesis-driven goal
                # state; should be set to an integer reflecting a number of steps
                min_post_goal_success_steps=5,  # Number of necessary steps for a
                # hypothesis
                # goal-state to be considered
                x_percent_scale_factor=0.75,  # Scale x-percent threshold to decide
                # when we should focus on pose rather than determining object ID; should
                # be bounded between 0:1.0; "mod" for modifier
                desired_object_distance=0.025,  # Distance from the object to the
                # agent that is considered "close enough" to the object
            ),
            # additional SDR related configs
            sdr_args=dict(
                log_path=os.path.join(monty_logs_dir, "evidence_sdr/log/patch_0"),
                sdr_length=2048,  # Size of SDRs
                sdr_on_bits=41,  # Number of active bits in the SDRs
                sdr_lr=1e-2,  # Learning rate of the encoding algorithm
                n_sdr_epochs=1000,  # Number of training epochs per episode
                stability=0.0,  # Stability of the old objects
                sdr_log_flag=True,  # log the output of the module
            ),
        ),
    )


def get_lm(patch_name):

    def get_lm_args():
        return dict(
            max_match_distance=0.01,
            tolerances={patch_name: default_tolerance_values},
            feature_weights=default_feature_weights,
            sdr_args=dict(
                log_path=os.path.join(monty_logs_dir, "evidence_sdr/log/", patch_name),
                sdr_length=2048,  # Size of SDRs
                sdr_on_bits=41,  # Number of active bits in the SDRs
                sdr_lr=1e-2,  # Learning rate of the encoding algorithm
                n_sdr_epochs=1000,  # Number of training epochs per episode
                stability=0.0,  # Stability of the old objects
                sdr_log_flag=True,  # log the output of the module
            ),
        )

    return dict(
        learning_module_class=EvidenceSDRGraphLM,
        learning_module_args=get_lm_args(),
    )


# ==== 1 Learning Module configs ======= #
all_objects_1_rotation_eslm = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=os.path.join(
            fe_pretrain_dir,
            "supervised_pre_training_all_objects/pretrained/",
        ),
        n_eval_epochs=1,
    ),
    logging_config=EvalEvidenceLMLoggingConfig(wandb_handlers=[]),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceSDRGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.01,
                    tolerances=default_tolerances,
                    feature_weights=default_feature_weights,
                    sdr_args=dict(
                        log_path=os.path.join(monty_logs_dir, "evidence_sdr/1"),
                        sdr_length=2048,  # Size of SDRs
                        sdr_on_bits=41,  # Number of active bits in the SDRs
                        sdr_lr=1e-2,  # Learning rate of the encoding algorithm
                        n_sdr_epochs=1000,  # Number of training epochs per episode
                        stability=0.0,  # Stability of the old objects
                        sdr_log_flag=True,  # log the output of the module
                    ),
                ),
            )
        ),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77, object_list=SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=[[90, 0, 180]]),
    ),
)


# ==== 5 Learning Module configs ======= #
all_objects_1_rot_5lms_eslm = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=os.path.join(
            fe_pretrain_dir,
            "supervised_pre_training_5lms_all_objects/pretrained/",
        ),
        n_eval_epochs=1,
        min_lms_match=3,
        max_eval_steps=1000,
    ),
    logging_config=EvalEvidenceLMLoggingConfig(
        # wandb_handlers=[],
        monty_log_level="BASIC",
        python_log_level="INFO",
    ),
    monty_config=FiveLMMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        monty_class=MontyForEvidenceGraphMatching,  # has custom evidence voting method
        learning_module_configs={
            f"learning_module_{i}": get_lm(f"patch_{i}") for i in range(5)
        },
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=FiveLMMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77, object_list=SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=[[90, 0, 180]]),
    ),
)


# ==== benchmark surf agent ======= #
base_77obj_surf_agent_sdr = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=os.path.join(
            fe_pretrain_dir,
            "surf_agent_1lm_77obj/pretrained/",
        ),
        n_eval_epochs=1,
        max_total_steps=5000,
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="benchmark_experiments",
    ),
    monty_config=SurfaceAndViewSOTAMontyConfig(
        learning_module_configs=dict(learning_module_0=get_evidence_surf_1lm_config(5)),
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
        monty_args=MontyArgs(min_eval_steps=20),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=10),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0,
            77,
        ),
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
    ),
)


# ==== benchmark surf agent with Noise ======= #
randrot_noise_77obj_surf_agent_sdr = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=os.path.join(
            fe_pretrain_dir,
            "surf_agent_1lm_77obj/pretrained/",
        ),
        n_eval_epochs=1,
        max_total_steps=5000,
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        # wandb_group="benchmark_experiments",
        wandb_handlers=[],
    ),
    monty_config=SurfaceAndViewSOTAMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_surf_agent_sensor_module,
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        learning_module_configs=dict(
            learning_module_0=get_evidence_surf_1lm_config(10)
        ),
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
        monty_args=MontyArgs(min_eval_steps=20),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=10),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0,
            77,
        ),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)


# === Two-LM STACKED EXPERIMENT === #

# The stacked LM experiments run as expected without errors, but since this is not a
# multi-object environment, various aspects of a hierarchy are not tested here.
# For example, Higher-level graphs of lower-level objects are not created and used
# for matching scenes.
# TODO: More testing is needed for a complete integration of EvidenceSDR in Monty

two_stacked_lms_config = dict(
    learning_module_0=dict(
        learning_module_class=EvidenceSDRGraphLM,
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
            sdr_args=dict(
                log_path=os.path.join(monty_logs_dir, "evidence_sdr/0"),
                sdr_length=2048,  # Size of SDRs
                sdr_on_bits=41,  # Number of active bits in the SDRs
                sdr_lr=1e-2,  # Learning rate of the encoding algorithm
                n_sdr_epochs=1000,  # Number of training epochs per episode
                stability=0.0,  # Stability of the old objects
                sdr_log_flag=False,  # log the output of the module
            ),
        ),
    ),
    learning_module_1=dict(
        learning_module_class=EvidenceSDRGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={
                "patch_1": {
                    "hsv": np.array([0.1, 1, 1]),
                    "principal_curvatures_log": np.ones(2),
                },
                # In EvidenceSDRGraphLM, we have an SDR representation for each
                # object and similarity is measured in bit overlap between SDRs.
                # The tolerance here refers to bits of overlap above which two SDRs
                # are considered the same.
                "learning_module_0": {"object_id": 20},
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
            sdr_args=dict(
                log_path=os.path.join(monty_logs_dir, "evidence_sdr/1"),
                sdr_length=2048,  # Size of SDRs
                sdr_on_bits=41,  # Number of active bits in the SDRs
                sdr_lr=1e-2,  # Learning rate of the encoding algorithm
                n_sdr_epochs=1000,  # Number of training epochs per episode
                stability=0.0,  # Stability of the old objects
                sdr_log_flag=False,  # log the output of the module
            ),
        ),
    ),
)


two_lm_distant_heterarchy_sdr = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=DebugExperimentArgs(
        do_eval=False,
        do_train=True,
        n_train_epochs=2,
        n_eval_epochs=len([[0.0, 0.0, 0.0]]),
        max_train_steps=200,
        max_eval_steps=200,
        min_lms_match=2,
    ),
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=10000, min_train_steps=3),
        learning_module_configs=two_stacked_lms_config,
    ),
    logging_config=LoggingConfig(
        python_log_level="INFO",
        monty_handlers=[BasicCSVStatsHandler],
        monty_log_level="BASIC",
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=TwoLMStackedDistantMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 2),
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0.0, 0.0, 0.0]]),
    ),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0.0, 0.0, 0.0]]),
    ),
)


# === MAIN CONFIGS === #
CONFIGS = dict(
    all_objects_1_rotation_eslm=all_objects_1_rotation_eslm,
    all_objects_1_rot_5lms_eslm=all_objects_1_rot_5lms_eslm,
    base_77obj_surf_agent_sdr=base_77obj_surf_agent_sdr,
    randrot_noise_77obj_surf_agent_sdr=randrot_noise_77obj_surf_agent_sdr,
    two_lm_distant_heterarchy_sdr=two_lm_distant_heterarchy_sdr,
)
