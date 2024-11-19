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
    MontyFeatureGraphArgs,
    MotorSystemConfigCurvatureInformedSurface,
    MotorSystemConfigInformedNoTransStepS3,
    ParallelEvidenceLMLoggingConfig,
    PatchAndViewMontyConfig,
    SurfaceAndViewMontyConfig,
    get_possible_3d_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    NoisyPatchViewFinderMountHabitatDatasetArgs,
    NoisySurfaceViewFinderMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PredefinedObjectInitializer,
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
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
    HabitatDistantPatchSM,
)

"""
Experiments used to evaluate the performance of different (typically "finger"-based)
motor policies

Note that as the surface-agent SM and associated motor policies can make use of visual
features (e.g. hue), more specifically
    - the distant-agent remains fixed in space at a distance, performing saccade-like
    movements across the surface of the object, like an eye
        - in theory, this enables large movements, so long as the object is still within
        the sensor's field-of-view afterwards
    - the surface-agent moves along the surface of the object, always aiming to remain
    relatively close, and as such it is constrained to take smaller incremental
    movements, like a finger
        - the advantage over the distant-agent (at least currently) is this enables it
        to explore the surface of an object more efficiently
        - in the long-term, it is likely that a real-world
        sensor (such as a robotic digit) following such a policy would also have more
        accurate depth readings than a camera alone; to reflect this, the finger
        is currently clipped in its depth reading distance (i.e. can only "see" to
        a certain distance)
        - furthermore, the surface-constrained movements may also carry valuable
        information for e.g. estimating the principal curvatures of a surface
"""

tested_degrees = np.linspace(0, 360, 5)[:-1]  # gives 32 combinations when processed
# by get_possible_3d_rotations, 6 of which we will sub-sample

# Offset the rotations so that they will be different from those used at training,
# but easily intepretable (i.e. not random)
# Currently sub-sampling the first of these *6* rotations after realizing it would take
# too long to evaluate all 32, while keeping results comparable to ones that were
# terminated early; eventually should shift to more orthogonal rotations
rot_displacement = 45
test_rotations_partial = get_possible_3d_rotations(
    tested_degrees, displacement=rot_displacement
)[:6]

monty_models_dir = os.getenv("MONTY_MODELS")

if monty_models_dir is None:
    monty_models_dir = "~/tbp/results/monty/pretrained_models/"
    print(f"MONTY_MODELS not set. Using default directory: {monty_models_dir}")

fe_pretrain_dir = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_ycb_v4")
)

model_path = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_base/pretrained/",
)

model_path_all_objects = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_all_objects/pretrained/",
)

# Model that has had additional surface-agent-based learning following
# distant-agent-based scanning
model_path_all_objects_augmented_training = os.path.join(
    fe_pretrain_dir,
    "supervised_additional_training_surf_agent/pretrained/",
)

# # Offset from typical values used in pre-training
# test_rotations_partial = [
#     [45, 45, 45],
#     [45, 135, 45],
#     [45, 225, 45],
# ]

default_tolerances = {
    "patch": {
        "hsv": np.array([0.1, 1, 1]),  # only look at hue
        "principal_curvatures_log": np.ones(2),
    }
}  # features where weight is not specified default weight to 1
# Everything is weighted 1, except for saturation and value which are not used.
default_feature_weights = {
    "patch": {
        "hsv": np.array([1, 0, 0]),
    }
}

default_evidence_lm_config = dict(
    learning_module_0=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            evidence_update_threshold="all",
            feature_evidence_increment=0.5,
            past_weight=1,
            present_weight=1,
            max_nneighbors=3,
            required_symmetry_evidence=3,
            max_match_distance=0.01,
            path_similarity_threshold=0.01,
            tolerances=default_tolerances,
            feature_weights=default_feature_weights,
            x_percent_threshold=20,
        ),
    )
)

base_config_partial_elm = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
    ),
    logging_config=EvalEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=default_evidence_lm_config,
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

base_config_all_objects_elm = copy.deepcopy(base_config_partial_elm)
base_config_all_objects_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# === Distant-agent model with new sampled points ===
# Used to compare surface-agent and distant-agent policies; evaluated on *all* objects
dist_agentlearns5_infs3_all_objects_elm = copy.deepcopy(base_config_all_objects_elm)
dist_agentlearns5_infs3_all_objects_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=1250,  # x1.25 max_eval_steps due to occasional off-object steps
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# As above, but with FeatureChangeSM
dist_agentlearns5_infs3_all_objects_fcsm_elm = copy.deepcopy(
    base_config_all_objects_elm
)
dist_agentlearns5_infs3_all_objects_fcsm_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=2500,  # 2.5x max-eval steps (due to feature-change-SM)
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=False,
                    # agent
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # Sample new points
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# As above, with noise included
dist_agentlearns5_infs3_all_objects_fcsm_elm_noisy_locs = copy.deepcopy(
    base_config_all_objects_elm
)
dist_agentlearns5_infs3_all_objects_fcsm_elm_noisy_locs.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=2500,  # 2.5x max-eval steps (due to feature-change-SM)
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=False,
                    save_raw_obs=False,
                    noise_params={
                        "location": 0.0025,  # add gaussian noise with 0.0025 std
                        # Based on visualizations of 0.001 vs. 0.005 noise, this seems
                        # like it would be a reasonable amount
                    },
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # Sample new points
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)


# As above, also making use of the augmented_models (NB no noise)
augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm = copy.deepcopy(
    base_config_all_objects_elm
)
augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented_training,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=2500,  # 2.5x max-eval steps (due to feature-change-SM)
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=False,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # Sample new points
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)


# As above, also making use of the augmented_models, and with noise in locations
augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_noisy_locs = copy.deepcopy(
    base_config_all_objects_elm
)
augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_noisy_locs.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented_training,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=2500,  # 2.5x max-eval steps (due to feature-change-SM)
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=False,
                    save_raw_obs=False,
                    noise_params={
                        "location": 0.0025,  # add gaussian noise with 0.0025 std
                        # Based on visualizations of 0.001 vs. 0.005 noise, this seems
                        # like it would be a reasonable amount
                    },
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # Sample new points
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# === Surface-agent policies using a subset of YCB objects ===
# Standard surface-agent policy, some objects
base_surf_agent_sensor_on_dist_agent_model_elm = copy.deepcopy(base_config_partial_elm)
base_surf_agent_sensor_on_dist_agent_model_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=4000,  # x4 max_eval_steps due to finger-policy
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
    ),
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
)

# Principal-curvature-guided surface-agent policy, some objects
surf_agent_sensor_on_dist_agent_model_curv_policy_elm = copy.deepcopy(
    base_surf_agent_sensor_on_dist_agent_model_elm
)
surf_agent_sensor_on_dist_agent_model_curv_policy_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=4000,  # x4 max_eval_steps due to finger-policy
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
)

# PC-guided surface-agent policy, some objects, with feature-change SM
surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm = copy.deepcopy(
    base_surf_agent_sensor_on_dist_agent_model_elm
)
surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=10000,  # x4 max_eval_steps due to finger-policy, x2.5 for
        # feature-change SM
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=True,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
)


# === Surface-agent policies with all YCB objects and all rotations ===

# Standard surface-agent policy
surf_agent_sensor_on_dist_agent_model_all_objects_elm = copy.deepcopy(
    base_surf_agent_sensor_on_dist_agent_model_elm
)
surf_agent_sensor_on_dist_agent_model_all_objects_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=4000,
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# PC-informed surface-agent policy
surf_agent_sensor_on_dist_agent_model_curv_policy_elm_all_objects = copy.deepcopy(
    surf_agent_sensor_on_dist_agent_model_curv_policy_elm
)
surf_agent_sensor_on_dist_agent_model_curv_policy_elm_all_objects.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=4000,
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# Standard surface-agent policy with feature-change SM
surf_agent_sensor_on_dist_agent_model_fcsm_all_objects_elm = copy.deepcopy(
    surf_agent_sensor_on_dist_agent_model_all_objects_elm
)
surf_agent_sensor_on_dist_agent_model_fcsm_all_objects_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=10000,  # x4 max_eval_steps for surface-agent policy, x2.5 for
        # feature-change SM
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=True,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)


# Standard surface-agent policy with feature-change SM, and using the augmented learning
# models
augmented_surf_agent_model_fcsm_all_objects_elm = copy.deepcopy(
    surf_agent_sensor_on_dist_agent_model_all_objects_elm
)
augmented_surf_agent_model_fcsm_all_objects_elm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented_training,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=10000,  # x4 max_eval_steps for surface-agent policy, x2.5 for
        # feature-change SM
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=True,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)


# PC informed surface-agent policy with feature-change SM, all objects
surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm_all_objects = copy.deepcopy(
    base_surf_agent_sensor_on_dist_agent_model_elm
)
surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm_all_objects.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=10000,  # x4 max_eval_steps for surface-agent policy, x2.5 for
        # feature-change SM
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/8,
                        # "principal_curvatures_log": [1.5, 1.5],
                        "distance": 0.01,
                    },
                    surf_agent_sm=True,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# PC informed surface-agent policy with feature-change SM, all objects; noise_added
surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm_all_objects_noisy_locs = (
    copy.deepcopy(base_surf_agent_sensor_on_dist_agent_model_elm)
)
surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm_all_objects_noisy_locs.update(  # noqa E501
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=10000,  # x4 max_eval_steps for surface-agent policy, x2.5 for
        # feature-change SM
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/8,
                        # "principal_curvatures_log": [1.5, 1.5],
                        "distance": 0.01,
                    },
                    surf_agent_sm=True,
                    save_raw_obs=False,
                    noise_params={
                        "location": 0.0025,  # add gaussian noise with 0.0025 std
                        # Based on visualizations of 0.001 vs. 0.005 noise, this seems
                        # like it would be a reasonable amount
                    },
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)


# Using the pre-trained model with additional (augmented) learning that
# combines visual scanning followed by PC-guided exploration with the finger policy
augmented_surf_agent_model_curv_policy_fcsm_elm_all_objects = copy.deepcopy(
    base_surf_agent_sensor_on_dist_agent_model_elm
)
augmented_surf_agent_model_curv_policy_fcsm_elm_all_objects.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented_training,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=10000,  # x4 max_eval_steps for surface-agent policy, x2.5 for
        # feature-change SM
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/8,
                        # "principal_curvatures_log": [1.5, 1.5],
                        "distance": 0.01,
                        # Only using distance, because minimal/no tangeable benefit
                        # when including the others, and likely to *reduce* performance
                        # when we add noise
                    },
                    surf_agent_sm=True,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# As above, but usng the pre-trained model with additional noise to locations;
# this is in addition to "noise" from sampling novel locations, and evaluating
# at novel rotations
augmented_surf_agent_model_curv_policy_fcsm_elm_all_objects_noisy_locs = copy.deepcopy(
    base_surf_agent_sensor_on_dist_agent_model_elm
)
augmented_surf_agent_model_curv_policy_fcsm_elm_all_objects_noisy_locs.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented_training,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=10000,  # x4 max_eval_steps for surface-agent policy, x2.5 for
        # feature-change SM
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/8,
                        # "principal_curvatures_log": [1.5, 1.5],
                        "distance": 0.01,
                        # Only using distance, because minimal/no tangeable benefit
                        # when including the others, and likely to *reduce* performance
                        # when we add noise
                    },
                    surf_agent_sm=True,
                    save_raw_obs=False,
                    noise_params={
                        "location": 0.0025,  # add gaussian noise with 0.0025 std
                        # Based on visualizations of 0.001 vs. 0.005 noise, this seems
                        # like it would be a reasonable amount
                    },
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# === Experiments with even more noise (i.e. the "all-noise" condition) ===
# Note this noise is applied to the features (i.e. after processing)

default_all_noise_params = {
    "features": {
        "hsv": 0.1,  # add gaussian noise with 0.1 std
        "principal_curvatures_log": 0.1,
        "pc1_is_pc2": 0.01,  # flip bool in 1% of cases
        "point_normal": 2,  # rotate by random degrees along xyz
        "curvature_directions": 2,
    },
    "location": 0.002,  # add gaussian noise with 0.002 std
}

noisy_pose_params = {
    # rotate by random degrees along xyz with std=10 degree
    "point_normal": 10,
    "curvature_directions": 10,
}

# Standard-learning of models, with distant-agent for inference, and all noise applied
dist_agentlearns5_infs3_all_objects_fcsm_elm_all_noise = copy.deepcopy(
    base_config_all_objects_elm
)
dist_agentlearns5_infs3_all_objects_fcsm_elm_all_noise.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=2500,  # 2.5x max-eval steps (due to feature-change-SM)
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=False,
                    save_raw_obs=False,
                    noise_params=default_all_noise_params,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # Sample new points
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# Augmented-learning of models, with distant-agent at inference, and all-noise applied
augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_all_noise = copy.deepcopy(
    base_config_all_objects_elm
)
augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_all_noise.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented_training,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=2500,  # 2.5x max-eval steps (due to feature-change-SM)
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=False,
                    save_raw_obs=False,
                    noise_params=default_all_noise_params,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # Sample new points
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)


# Standard-learing of models, with surface-agent at inference, and all noise applied
surf_agent_model_curv_policy_fcsm_elm_all_objects_all_noise = copy.deepcopy(
    base_surf_agent_sensor_on_dist_agent_model_elm
)
surf_agent_model_curv_policy_fcsm_elm_all_objects_all_noise.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=10000,  # x4 max_eval_steps for surface-agent policy, x2.5 for
        # feature-change SM
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/8,
                        # "principal_curvatures_log": [1.5, 1.5],
                        "distance": 0.01,
                        # Only using distance, because minimal/no tangeable benefit
                        # when including the others, and likely to *reduce* performance
                        # when we add noise
                    },
                    surf_agent_sm=True,
                    save_raw_obs=False,
                    noise_params=default_all_noise_params,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
)

# === Experiments with even raw sensor noise (i.e. before features are extracted) ===

# Standard-learning of models, with distant-agent for inference, and sensor noise
# applied
dist_agentlearns5_infs3_all_objects_fcsm_elm_raw_noise = copy.deepcopy(
    base_config_all_objects_elm
)
dist_agentlearns5_infs3_all_objects_fcsm_elm_raw_noise.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=2500,  # 2.5x max-eval steps (due to feature-change-SM)
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=False,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # Sample new points
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
    dataset_args=NoisyPatchViewFinderMountHabitatDatasetArgs(),
)

# Augmented-learning of models, with distant-agent at inference, and sensor-noise
# applied
augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_raw_noise = copy.deepcopy(
    base_config_all_objects_elm
)
augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_raw_noise.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects_augmented_training,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=2500,  # 2.5x max-eval steps (due to feature-change-SM)
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/4,
                        # "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    surf_agent_sm=False,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # Sample new points
        motor_system_config=MotorSystemConfigInformedNoTransStepS3(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
    dataset_args=NoisyPatchViewFinderMountHabitatDatasetArgs(),
)


# Standard-learing of models, with surface-agent at inference, and sensor noise applied
surf_agent_model_curv_policy_fcsm_elm_all_objects_raw_noise = copy.deepcopy(
    base_surf_agent_sensor_on_dist_agent_model_elm
)
surf_agent_model_curv_policy_fcsm_elm_all_objects_raw_noise.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_all_objects,
        n_eval_epochs=len(test_rotations_partial),
        max_eval_steps=1000,
        max_total_steps=10000,  # x4 max_eval_steps for surface-agent policy, x2.5 for
        # feature-change SM
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="policy_runs",
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.05,
                    tolerances=default_tolerances,
                    feature_weights=dict(),
                    x_percent_threshold=20,
                ),
            )
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "point_normal",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        # "hsv": [0.2, 1, 1],
                        # "point_normal": np.pi/8,
                        # "principal_curvatures_log": [1.5, 1.5],
                        "distance": 0.01,
                        # Only using distance, because minimal/no tangeable benefit
                        # when including the others, and likely to *reduce* performance
                        # when we add noise
                    },
                    surf_agent_sm=True,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 77),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_partial
        ),
    ),
    dataset_args=NoisySurfaceViewFinderMountHabitatDatasetArgs(),
)

CONFIGS = dict(
    # Distant policies
    dist_agentlearns5_infs3_all_objects_elm=dist_agentlearns5_infs3_all_objects_elm,
    dist_agentlearns5_infs3_all_objects_fcsm_elm=dist_agentlearns5_infs3_all_objects_fcsm_elm,  # noqa E501
    augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm=augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm,  # noqa E501
    # Surface-agent policies
    base_surf_agent_sensor_on_dist_agent_model_elm=base_surf_agent_sensor_on_dist_agent_model_elm,  # noqa: E501
    surf_agent_sensor_on_dist_agent_model_curv_policy_elm=surf_agent_sensor_on_dist_agent_model_curv_policy_elm,  # noqa E501
    surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm=surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm,  # noqa E501
    surf_agent_sensor_on_dist_agent_model_all_objects_elm=surf_agent_sensor_on_dist_agent_model_all_objects_elm,  # noqa E501
    surf_agent_sensor_on_dist_agent_model_fcsm_all_objects_elm=surf_agent_sensor_on_dist_agent_model_fcsm_all_objects_elm,  # noqa E501
    augmented_surf_agent_model_fcsm_all_objects_elm=augmented_surf_agent_model_fcsm_all_objects_elm,  # noqa E501
    surf_agent_sensor_on_dist_agent_model_curv_policy_elm_all_objects=surf_agent_sensor_on_dist_agent_model_curv_policy_elm_all_objects,  # noqa E501
    surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm_all_objects=surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm_all_objects,  # noqa E501
    augmented_surf_agent_model_curv_policy_fcsm_elm_all_objects=augmented_surf_agent_model_curv_policy_fcsm_elm_all_objects,  # noqa E501
    # Sims with noise added to the location data; note there is already "noise" in the
    # sense of sampling new points, as well as novel rotations
    dist_agentlearns5_infs3_all_objects_fcsm_elm_noisy_locs=dist_agentlearns5_infs3_all_objects_fcsm_elm_noisy_locs,  # noqa E501
    surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm_all_objects_noisy_locs=surf_agent_sensor_on_dist_agent_model_curv_policy_fcsm_elm_all_objects_noisy_locs,  # noqa E501
    augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_noisy_locs=augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_noisy_locs,  # noqa E501
    augmented_surf_agent_model_curv_policy_fcsm_elm_all_objects_noisy_locs=augmented_surf_agent_model_curv_policy_fcsm_elm_all_objects_noisy_locs,  # noqa E501
    # Sims with even *more* noise
    dist_agentlearns5_infs3_all_objects_fcsm_elm_all_noise=dist_agentlearns5_infs3_all_objects_fcsm_elm_all_noise,  # noqa E501
    augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_all_noise=augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_all_noise,  # noqa E501
    surf_agent_model_curv_policy_fcsm_elm_all_objects_all_noise=surf_agent_model_curv_policy_fcsm_elm_all_objects_all_noise,  # noqa E501
    # Sims with raw sensor noise
    dist_agentlearns5_infs3_all_objects_fcsm_elm_raw_noise=dist_agentlearns5_infs3_all_objects_fcsm_elm_raw_noise,  # noqa E501
    augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_raw_noise=augmented_dist_agentlearns5_infs3_all_objects_fcsm_elm_raw_noise,  # noqa E501
    surf_agent_model_curv_policy_fcsm_elm_all_objects_raw_noise=surf_agent_model_curv_policy_fcsm_elm_all_objects_raw_noise,  # noqa E501
)
