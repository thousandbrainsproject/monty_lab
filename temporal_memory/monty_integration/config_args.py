# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Add this to frameworks/config_utils/config_args.py

import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigSurface,
    PatchAndViewMontyConfig,
)
from tbp.monty.frameworks.models.monty_base import LearningModuleBase, MontyBase
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatSurfacePatchSM,
)


class TactileTemporalMemoryLM(LearningModuleBase):
    """Learning module that combines tactile input with the temporal memory
    object recognition
    """

    def matching_step(self, observations):
        pass


# This is from frameworks/models/sensor_modules.py
class HabitatSurfacePatchForTemporalMemorySM(HabitatSurfacePatchSM):
    def __init__(
        self, sensor_module_id, features, tm_params_file_loc, save_raw_obs=False
    ):
        super().__init__(sensor_module_id, features, save_raw_obs)
        self.load_temporal_memory_params(tm_params_file_loc)

    def load_temporal_memory_params(self, tm_params_file_loc):
        tm_params_file = os.path.expanduser(tm_params_file_loc)
        with open(tm_params_file, "rb") as f:
            norm_params = pickle.load(f)
        (
            self.lower_bound_curv,
            self.upper_bound_curv,
            self.lower_bound_loc,
            self.upper_bound_loc,
            self.min_value,
            self.max_value,
        ) = (
            norm_params["lower_bound_curv"],
            norm_params["upper_bound_curv"],
            norm_params["lower_bound_loc"],
            norm_params["upper_bound_loc"],
            norm_params["min_value"],
            norm_params["max_value"],
        )

    def step(self, data):
        processed_ob = super().step(data)
        # update the processed observation with normalized curv and coord
        if processed_ob["features"]["on_object"]:
            gc = processed_ob["features"]["gaussian_curvature"]
            curv_for_tm = (gc - self.lower_bound_curv) * (
                self.max_value - self.min_value
            ) / (self.upper_bound_curv - self.lower_bound_curv) + self.min_value
            if curv_for_tm < self.min_value:
                curv_for_tm = self.min_value
            elif curv_for_tm > self.max_value:
                curv_for_tm = self.max_value
            processed_ob["features"]["curvature_for_TM"] = np.round(curv_for_tm).astype(
                np.int64
            )

            loc = processed_ob.location
            coord_for_tm = (loc - self.lower_bound_loc) * (
                self.max_value - self.min_value
            ) / (self.upper_bound_loc - self.lower_bound_loc) + self.min_value
            processed_ob["features"]["coords_for_TM"] = np.round(coord_for_tm).astype(
                np.int64
            )

            self.processed_obs[-1] = processed_ob
        return processed_ob


@dataclass
class SurfaceAndViewForTMMontyConfig(PatchAndViewMontyConfig):

    monty_class: Callable = MontyBase

    monty_args: Union[Dict, MontyArgs] = field(default_factory=MontyFeatureGraphArgs)

    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSurfacePatchForTemporalMemorySM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    # TODO: would be nicer to just use lm.tolerances.keys() here
                    # but not sure how to easily do this.
                    features=[
                        # morphological features (nescessarry)
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        # non-morphological features (optional)
                        "object_coverage",
                        "min_depth",
                        "mean_depth",
                        "gaussian_curvature",
                        "curvature_for_TM",
                        "coords_for_TM",
                    ],
                    tm_params_file_loc="~/tbp/tbp.monty/projects/tactile_temporal_memory/tm_dataset/norm_parameters.pkl",  # noqa: E501
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
        )
    )

    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            learning_module_0=dict(
                learning_module_class=TactileTemporalMemoryLM,
                learning_module_args=dict(),
            )
        )
    )

    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigSurface
    )
    sm_to_agent_dict: Dict = field(
        default_factory=lambda: dict(
            patch="agent_id_0",
            view_finder="agent_id_0",
        )
    )
    sm_to_lm_matrix: List = field(
        default_factory=lambda: [[0]],  # View finder (sm1) not connected to lm
    )
    lm_to_lm_matrix: Optional[List] = None
    lm_to_lm_vote_matrix: Optional[List] = None
    monty_args: Union[Dict, dataclass] = field(default_factory=MontyArgs)
