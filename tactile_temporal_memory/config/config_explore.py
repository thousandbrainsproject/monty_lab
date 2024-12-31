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

from tbp.monty.frameworks.config_utils.config_args import (
    EvalLoggingConfig,
    MontyFeatureGraphArgs,
    TouchAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    TouchViewFinderMountHabitatDatasetArgs,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import DataCollectionExperiment
from tbp.monty.frameworks.models.feature_location_matching import (
    TactileTemporalMemoryLM,
)
from tbp.monty.frameworks.models.monty_base import MontyBase

SHUFFLED_YCB_OBJECTS_II = [
    "dice",
    "sponge",
    "a_colored_wood_blocks",
    "lemon",
    "e_lego_duplo",
    "c_toy_airplane",
    "d_toy_airplane",
    "extra_large_clamp",
    "power_drill",
    "c_cups",
    "b_colored_wood_blocks",
    "flat_screwdriver",
    "b_cups",
    "a_marbles",
    "e_toy_airplane",
    "b_toy_airplane",
    "gelatin_box",
    "tomato_soup_can",
    "strawberry",
    "golf_ball",
    "orange",
    "plum",
    "mini_soccer_ball",
    "potted_meat_can",
    "g_lego_duplo",
    "sugar_box",
    "apple",
    "e_cups",
    "plate",
    "phillips_screwdriver",
    "i_cups",
    "padlock",
    "banana",
    "cracker_box",
    "h_cups",
    "large_clamp",
    "nine_hole_peg_test",
    "spatula",
    "d_cups",
    "b_marbles",
    "adjustable_wrench",
    "bleach_cleanser",
    "b_lego_duplo",
    "pear",
    "g_cups",
    "pitcher_base",
    "hammer",
    "scissors",
    "large_marker",
    "baseball",
    "bowl",
    "chain",
    "tennis_ball",
    "peach",
    "a_lego_duplo",
    "spoon",
    "j_cups",
    "d_lego_duplo",
    "pudding_box",
    "medium_clamp",
    "skillet_lid",
    "f_cups",
    "softball",
    "wood_block",
    "racquetball",
    "fork",
    "mug",
    "a_cups",
    "c_lego_duplo",
    "windex_bottle",
    "knife",
    "rubiks_cube",
    "f_lego_duplo",
    "mustard_bottle",
    "master_chef_can",
    "a_toy_airplane",
    "tuna_fish_can",
    "foam_brick",
]


rotation = [[0, 0, 0]]

base_config = dict(
    monty_config=TouchAndViewMontyConfig(
        monty_class=MontyBase,
        monty_args=MontyFeatureGraphArgs(),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=TactileTemporalMemoryLM,
                learning_module_args=dict(),
            )
        ),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=TouchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, 10, object_list=SHUFFLED_YCB_OBJECTS_II
        ),
        object_init_sampler=PredefinedObjectInitializer(rotations=rotation),
    ),
    logging_config=EvalLoggingConfig(wandb_handlers=[], monty_log_level="TEST"),
)


explore_touch = copy.deepcopy(base_config)
explore_touch.update(
    experiment_class=DataCollectionExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=1,
        show_sensor_output=False,
        do_train=True,
        do_eval=False,
        max_train_steps=10000,
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, 10, object_list=SHUFFLED_YCB_OBJECTS_II
        ),  # SHUFFLED_YCB_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=rotation),
    ),
)

explore_touch_test = copy.deepcopy(explore_touch)
explore_touch_test.update(
    experiment_args=ExperimentArgs(
        n_train_epochs=1,
        show_sensor_output=True,
        do_train=True,
        do_eval=False,
        max_train_steps=200,
    )
)


CONFIGS = dict(explore_touch=explore_touch, explore_touch_test=explore_touch_test)
