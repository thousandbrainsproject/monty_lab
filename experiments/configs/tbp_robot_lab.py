# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import copy
import os

from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    MotorSystemConfigInformedNoTransStepS20,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    SurfaceAndViewSOTAMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.models.sensor_modules import DetailedLoggingSM
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMontyWorldMountHabitatDatasetArgs,
    SurfaceViewFinderMontyWorldMountHabitatDatasetArgs,
)

from .config_utils import import_config_from_monty

pretrained_dir = import_config_from_monty("defaults.py", "pretrained_dir")

# A set of objects that can be obtained internationally and used to test Monty's
# performance on those physical objects.
TBP_ROBOT_LAB_OBJECTS = [
    "numenta_mug",
    "montys_brain",
    "montys_heart",
    "ramen_pack",
    "hot_sauce",
    "harissa_oil",
    "tomato_soup_can",
    "mustard_bottle",
    "tuna_fish_can",
    "potted_meat_can",
]


# ============== TRAINING CONFIGS ===============
train_rotations_all = get_cube_face_and_corner_views_rotations()

only_surf_agent_training_10obj = import_config_from_monty(
    "pretraining_experiments.py", "only_surf_agent_training_10obj"
)

tbp_robot_lab_dataset_args = SurfaceViewFinderMontyWorldMountHabitatDatasetArgs()
tbp_robot_lab_dataset_args.env_init_args["data_path"] = os.path.join(
    os.environ["MONTY_DATA"], "tbp_robot_lab"
)
only_surf_agent_training_tbp_robot_lab = copy.deepcopy(only_surf_agent_training_10obj)
only_surf_agent_training_tbp_robot_lab.update(
    logging_config=PretrainLoggingConfig(
        output_dir=pretrained_dir,
        run_name="surf_agent_1lm_tbp_robot_lab",
    ),
    dataset_args=tbp_robot_lab_dataset_args,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=TBP_ROBOT_LAB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)

# ============== INFERENCE CONFIGS ==============
min_eval_steps = import_config_from_monty("defaults.py", "min_eval_steps")

default_all_noisy_sensor_module = import_config_from_monty(
    "defaults.py", "default_all_noisy_sensor_module"
)
default_evidence_1lm_config = import_config_from_monty(
    "defaults.py", "default_evidence_1lm_config"
)

randrot_noise_sim_on_scan_monty_world = import_config_from_monty(
    "monty_world_habitat_experiments.py",
    "randrot_noise_sim_on_scan_monty_world",
)

default_all_noisy_surf_agent_sensor_module = import_config_from_monty(
    "ycb_experiments.py", "default_all_noisy_surf_agent_sensor_module"
)

model_path_tbp_robot_lab = os.path.join(
    pretrained_dir,
    "surf_agent_1lm_tbp_robot_lab/pretrained/",
)

tbp_robot_lab_dist_dataset_args = PatchViewFinderMontyWorldMountHabitatDatasetArgs()
tbp_robot_lab_dist_dataset_args.env_init_args["data_path"] = os.path.join(
    os.environ["MONTY_DATA"], "tbp_robot_lab"
)

randrot_noise_dist_sim_on_scan_tbp_robot_lab = copy.deepcopy(
    randrot_noise_sim_on_scan_monty_world
)
randrot_noise_dist_sim_on_scan_tbp_robot_lab.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_tbp_robot_lab,
        n_eval_epochs=10,
    ),
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_sensor_module,
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        learning_module_configs=default_evidence_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        # Not using the hypothesis-driven motor system here to compare to a fixed sensor
        # setup.
        motor_system_config=MotorSystemConfigInformedNoTransStepS20(),
    ),
    dataset_args=tbp_robot_lab_dist_dataset_args,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=TBP_ROBOT_LAB_OBJECTS),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

randrot_noise_surf_sim_on_scan_tbp_robot_lab = copy.deepcopy(
    randrot_noise_dist_sim_on_scan_tbp_robot_lab
)
randrot_noise_surf_sim_on_scan_tbp_robot_lab.update(
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
        learning_module_configs=default_evidence_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        # In our real-world experiments the sensor is now able to move around the object
        # so we also allow this here for the simlation comparison.
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
    ),
    dataset_args=tbp_robot_lab_dataset_args,
)

CONFIGS = dict(
    only_surf_agent_training_tbp_robot_lab=only_surf_agent_training_tbp_robot_lab,
    randrot_noise_dist_sim_on_scan_tbp_robot_lab=randrot_noise_dist_sim_on_scan_tbp_robot_lab,
    randrot_noise_surf_sim_on_scan_tbp_robot_lab=randrot_noise_surf_sim_on_scan_tbp_robot_lab,
)
