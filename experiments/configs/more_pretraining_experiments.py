# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
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
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigCurvatureInformedSurface,
    MotorSystemConfigInformedNoTransStepS1,
    MotorSystemConfigNaiveScanSpiral,
    PatchAndViewFeatureChangeConfig,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    SurfaceAndViewMontyConfig,
    TwoLMMontyConfig,
    get_possible_3d_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    OmniglotDataloaderArgs,
    OmniglotDatasetArgs,
    PredefinedObjectInitializer,
    get_env_dataloader_per_object_by_idx,
    get_object_names_by_idx,
    get_omniglot_train_dataloader,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import (
    DISTINCT_OBJECTS,
    SHUFFLED_YCB_OBJECTS,
)
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatDistantPatchSM,
    HabitatSurfacePatchSM,
)
from tbp.monty.simulators.habitat.configs import (
    MultiLMMountHabitatDatasetArgs,
    NoisyPatchViewFinderMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
)

# --- Further Pretraining Experiments ---
# Current defaults are in tbp.monty.benchmarks.configs.pretraining_experiments

# FOR SUPERVISED PRETRAINING
train_degrees = np.linspace(0, 360, 5)[:-1]  # gives 32 combinations
train_rotations_all = get_possible_3d_rotations(train_degrees)

monty_models_dir = os.getenv("MONTY_MODELS")

fe_pretrain_dir = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_ycb_v8")
)

pre_surf_agent_visual_training_model_path = os.path.join(
    fe_pretrain_dir, "supervised_pre_training_all_objects/pretrained/"
)

supervised_pre_training_base = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(train_rotations_all),
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=fe_pretrain_dir,
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                    graph_delta_thresholds=dict(
                        patch=dict(
                            distance=0.001,
                            # Only first pose vector (point normal) is currently used
                            pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                            principal_curvatures_log=[1, 1],
                            hsv=[0.1, 1, 1],
                        )
                    ),
                ),
                # NOTE: Learning works with any LM type. For instance you can use
                # the following code to run learning with the EvidenceGraphLM:
                # learning_module_class=EvidenceGraphLM,
                # learning_module_args=dict(
                #     max_match_distance=0.01,
                #     tolerances={"patch": dict()},
                #     feature_weights=dict(),
                #     graph_delta_thresholds=dict(patch=dict(
                #         distance=0.001,
                #         pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                #         principal_curvatures_log=[1, 1],
                #         hsv=[0.1, 1, 1],
                #     )),
                # ),
                # NOTE: When learning with the EvidenceGraphLM or FeatureGraphLM, no
                # edges will be added to the learned graphs (also not needed for
                # matching) while learning with DisplacementGraphLM is a superset of
                # these, i.e. captures all necessary information to do inference with
                # any three of the LM types.
            )
        ),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=5)
        ),  # use spiral policy for more even object coverage during learning
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=DISTINCT_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,  # just placeholder
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
)

supervised_pre_training_storeall = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_storeall.update(
    logging_config=PretrainLoggingConfig(
        output_dir=fe_pretrain_dir,
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                    graph_delta_thresholds=dict(patch=dict(distance=0.00001)),
                ),
            )
        ),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=5)
        ),  # use spiral policy for more even object coverage during learning
    ),
)

supervised_pre_training_stepsize3 = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_stepsize3.update(
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
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=3)
        ),
    ),
)

supervised_pre_training_stepsize3_storeall = copy.deepcopy(
    supervised_pre_training_storeall
)
supervised_pre_training_stepsize3_storeall.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                    graph_delta_thresholds=dict(patch=dict(distance=0.00001)),
                ),
            )
        ),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=3)
        ),
    ),
)

supervised_pre_training_stepsize6 = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_stepsize6.update(
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
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=6)
        ),
    ),
)

supervised_pre_training_stepsize6_storeall = copy.deepcopy(
    supervised_pre_training_storeall
)
supervised_pre_training_stepsize6_storeall.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                    graph_delta_thresholds=dict(patch=dict(distance=0.00001)),
                ),
            )
        ),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=6)
        ),
    ),
)

supervised_pre_training_feature_change = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_feature_change.update(
    monty_config=PatchAndViewFeatureChangeConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000),
    ),
)

supervised_pre_training_feature_change_s3 = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_feature_change_s3.update(
    monty_config=PatchAndViewFeatureChangeConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=3)
        ),
    ),
)

supervised_pre_training_all_objects = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_all_objects.update(
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(SHUFFLED_YCB_OBJECTS), object_list=SHUFFLED_YCB_OBJECTS
        ),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)

supervised_pre_training_all_objects_storeall = copy.deepcopy(
    supervised_pre_training_storeall
)
supervised_pre_training_all_objects_storeall.update(
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(SHUFFLED_YCB_OBJECTS), object_list=SHUFFLED_YCB_OBJECTS
        ),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)

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
supervised_pre_training_random_rot = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_random_rot.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(random_rot),
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4, object_list=SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=random_rot),
    ),
)

supervised_pre_training_random_rot_storeall = copy.deepcopy(
    supervised_pre_training_storeall
)
supervised_pre_training_random_rot_storeall.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(random_rot),
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4, object_list=SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=random_rot),
    ),
)

view_finder_config = dict(
    sensor_module_class=DetailedLoggingSM,
    sensor_module_args=dict(
        sensor_module_id="view_finder",
        save_raw_obs=True,
    ),
)
default_sensor_features = [
    "pose_vectors",
    "pose_fully_defined",
    "on_object",
    "hsv",
    "principal_curvatures_log",
]

supervised_pre_training_location_noise002 = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_location_noise002.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
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
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=default_sensor_features,
                    save_raw_obs=True,
                    noise_params={
                        "location": 0.002,  # add gaussian noise with 0.002 std
                    },
                ),
            ),
            sensor_module_1=view_finder_config,
        ),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=5)
        ),  # use spiral policy for more even object coverage during learning
    ),
)

supervised_pre_training_location_noise005 = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_location_noise005.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
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
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=5)
        ),  # use spiral policy for more even object coverage during learning
    ),
)

supervised_pre_training_location_noise001 = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_location_noise001.update(
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
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
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=5)
        ),  # use spiral policy for more even object coverage during learning
    ),
)

supervised_pre_training_raw_depth_noise = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_raw_depth_noise.update(
    dataset_args=NoisyPatchViewFinderMountHabitatDatasetArgs(),
)

supervised_pre_training_surf_agent = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_surf_agent.update(
    experiment_args=ExperimentArgs(
        n_train_epochs=1,
        do_eval=False,
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(num_exploratory_steps=10000),
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
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 4),
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
    ),
)

# Perform additional surface-agent based training after having learned with visual
# saccades
# Makes use of principal-curvature policy guidance; note inclusion of visual
# features (HSV)
supervised_additional_training_surf_agent = copy.deepcopy(supervised_pre_training_base)
supervised_additional_training_surf_agent.update(
    experiment_args=ExperimentArgs(
        n_train_epochs=len(train_rotations_all),
        do_eval=False,
        # Note the additional passing of a previously trained model
        model_name_or_path=pre_surf_agent_visual_training_model_path,
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(num_exploratory_steps=500),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                    graph_delta_thresholds=dict(
                        patch=dict(
                            distance=0.01,
                            pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                            principal_curvatures_log=[1.0, 1.0],
                            hsv=[0.1, 1, 1],
                        )
                    ),
                ),
            ),
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSurfacePatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    save_raw_obs=True,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, len(SHUFFLED_YCB_OBJECTS)),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)

supervised_additional_training_surf_agent_10obj_01 = copy.deepcopy(
    supervised_additional_training_surf_agent
)
supervised_additional_training_surf_agent_10obj_01.update(
    experiment_args=ExperimentArgs(
        n_train_epochs=len(train_rotations_all),
        do_eval=False,
        model_name_or_path=os.path.join(
            fe_pretrain_dir,
            "supervised_pre_training_10distinctobj_005dist_base/pretrained/",
        ),
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=fe_pretrain_dir,
        run_name="supervised_additional_training_10distinctobj_01on005",
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=DISTINCT_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)

supervised_pre_training_on_omniglot = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_on_omniglot.update(
    experiment_args=ExperimentArgs(
        n_train_epochs=1,
        do_eval=False,
    ),
    monty_config=PatchAndViewMontyConfig(
        # Take 1 step at a time, following the drawing path of the letter
        motor_system_config=MotorSystemConfigInformedNoTransStepS1(),
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
                    save_raw_obs=False,
                ),
            ),
        ),
    ),
    dataset_args=OmniglotDatasetArgs(),
    train_dataloader_class=ED.OmniglotDataLoader,
    # Train on the first version of each character (there are 20 drawings for each
    # character in each alphabet, here we see one of them).
    train_dataloader_args=OmniglotDataloaderArgs(versions=[1, 1, 1, 1, 1, 1]),
    eval_dataloader_class=ED.OmniglotDataLoader,
    eval_dataloader_args=OmniglotDataloaderArgs(),
)

# Not super large yet but learns all characters from the first alphabet (48).
# To learn more alphabets increase num_alphabets parameter.
supervised_pre_training_on_omniglot_large = copy.deepcopy(
    supervised_pre_training_on_omniglot
)
supervised_pre_training_on_omniglot_large.update(
    # Showing more than one version in supervised pretraining is difficult since
    # we don't have the ground truth location of the letters so we don't know
    # where to overlay them.
    train_dataloader_args=get_omniglot_train_dataloader(
        num_versions=1, alphabet_ids=[1]
    ),
)

supervised_pre_training_2lms = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_2lms.update(
    monty_config=TwoLMMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                ),
            ),
            learning_module_1=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=make_naive_scan_policy_config(step_size=5)
        ),
    ),
    dataset_args=MultiLMMountHabitatDatasetArgs(),
)

CONFIGS = dict(
    supervised_pre_training_base=supervised_pre_training_base,
    supervised_pre_training_storeall=supervised_pre_training_storeall,
    supervised_pre_training_stepsize3=supervised_pre_training_stepsize3,
    supervised_pre_training_stepsize3_storeall=supervised_pre_training_stepsize3_storeall,
    supervised_pre_training_stepsize6=supervised_pre_training_stepsize6,
    supervised_pre_training_stepsize6_storeall=supervised_pre_training_stepsize6_storeall,
    supervised_pre_training_feature_change=supervised_pre_training_feature_change,
    supervised_pre_training_feature_change_s3=supervised_pre_training_feature_change_s3,
    supervised_pre_training_all_objects=supervised_pre_training_all_objects,
    supervised_pre_training_all_objects_storeall=supervised_pre_training_all_objects_storeall,
    supervised_pre_training_random_rot=supervised_pre_training_random_rot,
    supervised_pre_training_random_rot_storeall=supervised_pre_training_random_rot_storeall,
    supervised_pre_training_location_noise005=supervised_pre_training_location_noise005,
    supervised_pre_training_location_noise002=supervised_pre_training_location_noise002,
    supervised_pre_training_location_noise001=supervised_pre_training_location_noise001,
    supervised_pre_training_raw_depth_noise=supervised_pre_training_raw_depth_noise,
    supervised_pre_training_surf_agent=supervised_pre_training_surf_agent,
    supervised_pre_training_on_omniglot=supervised_pre_training_on_omniglot,
    supervised_pre_training_on_omniglot_large=supervised_pre_training_on_omniglot_large,
    supervised_pre_training_2lms=supervised_pre_training_2lms,
    supervised_additional_training_surf_agent=supervised_additional_training_surf_agent,
    supervised_additional_training_surf_agent_10obj_01=supervised_additional_training_surf_agent_10obj_01,
)
