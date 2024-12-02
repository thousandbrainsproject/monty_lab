# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Adds additional configs for DMC."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from tbp.monty.frameworks.config_utils.config_args import (
    MontyConfig,
    MontyArgs,
    features,
    MotorSystemConfigInformedNoTrans,
)
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatDistantPatchSM,
)


@dataclass
class NineLMMontyConfig(MontyConfig):
    monty_class: Callable = MontyForGraphMatching
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict
    )
    monty_class: Callable = MontyForGraphMatching
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_1=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_2=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_3=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_4=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_5=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_6=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_7=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_8=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
        )
    )
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_0",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_1",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_2=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_2",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_3=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_3",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_4=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_4",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_5=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_5",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_6=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_6",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_7=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_7",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_8=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_8",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_9=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigInformedNoTrans
    )
    sm_to_agent_dict: Dict = field(
        default_factory=lambda: dict(
            patch_0="agent_id_0",
            patch_1="agent_id_0",
            patch_2="agent_id_0",
            patch_3="agent_id_0",
            patch_4="agent_id_0",
            patch_5="agent_id_0",
            patch_6="agent_id_0",
            patch_7="agent_id_0",
            patch_8="agent_id_0",
            view_finder="agent_id_0",
        )
    )
    sm_to_lm_matrix: List = field(
        default_factory=lambda: [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
        ],  # View finder (sm9) not connected to lm
    )
    lm_to_lm_matrix: Optional[List] = None
    # lm_to_lm_vote_matrix: Optional[List] = None
    # All LMs connect to each other
    lm_to_lm_vote_matrix: List = field(
        default_factory=lambda: [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 8],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ]
    )
    monty_args: Union[Dict, dataclass] = field(default_factory=MontyArgs)


@dataclass
class TenLMMontyConfig(MontyConfig):
    monty_class: Callable = MontyForGraphMatching
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict
    )
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_1=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_2=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_3=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_4=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_5=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_6=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_7=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_8=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_9=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
        )
    )
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_0",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_1",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_2=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_2",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_3=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_3",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_4=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_4",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_5=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_5",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_6=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_6",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_7=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_7",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_8=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_8",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_9=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_9",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_10=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigInformedNoTrans
    )
    sm_to_agent_dict: Dict = field(
        default_factory=lambda: dict(
            patch_0="agent_id_0",
            patch_1="agent_id_0",
            patch_2="agent_id_0",
            patch_3="agent_id_0",
            patch_4="agent_id_0",
            patch_5="agent_id_0",
            patch_6="agent_id_0",
            patch_7="agent_id_0",
            patch_8="agent_id_0",
            patch_9="agent_id_0",
            view_finder="agent_id_0",
        )
    )
    sm_to_lm_matrix: List = field(
        default_factory=lambda: [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9],
        ],  # View finder (sm10) not connected to lm
    )
    lm_to_lm_matrix: Optional[List] = None
    # lm_to_lm_vote_matrix: Optional[List] = None
    # All LMs connect to each other
    lm_to_lm_vote_matrix: List = field(
        default_factory=lambda: [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ]
    )
    monty_args: Union[Dict, dataclass] = field(default_factory=MontyArgs)
