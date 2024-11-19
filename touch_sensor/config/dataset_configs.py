# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union

import numpy as np
from classes.agent import TouchAgent
from classes.transform import DepthTo3DLocationsTouch
from scipy.spatial.transform import Rotation
from tbp.monty.frameworks.config_utils.make_dataset_configs import scipy_to_numpy_quat
from tbp.monty.frameworks.environments.habitat import HabitatEnvironment
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.loggers.exp_logger import LoggingCallbackHandler
from tbp.monty.simulators.habitat import HabitatAgent, ObjectConfig

# ---------
# run / training / eval args
# ---------


@dataclass
class ExperimentArgs:
    do_train: bool = True
    do_eval: bool = True
    show_sensor_output: bool = False
    max_train_steps: int = 1000
    max_eval_steps: int = 1000
    n_train_epochs: int = 3
    n_eval_epochs: int = 3
    log_level: str = "INFO"
    output_dir: str = os.path.expanduser("~/tbp/tbp.monty/projects/monty_runs/logs/")
    run_name: str = ""
    model_name_or_path: str = ""
    save_stats: bool = True


@dataclass
class DebugExperimentArgs(ExperimentArgs):
    do_train: bool = True
    do_eval: bool = True
    show_sensor_output: bool = True
    max_train_steps: int = 50
    max_eval_steps: int = 50
    n_train_epochs: int = 1
    n_eval_epochs: int = 1
    log_level: str = "DEBUG"


@dataclass
class LoggerArgs:
    loggers: List
    logger_handler_class: Callable


@dataclass
class BaseLoggerArgs(LoggerArgs):
    loggers: List = field(default_factory=lambda: [])  # TODO fill in with defaults
    logger_handler_class: Callable = field(default=LoggingCallbackHandler)


@dataclass
class EnvInitArgs:
    """
    Args for :class:`HabitatEnvironment`
    """

    agents: List[HabitatAgent]
    objects: List[ObjectConfig] = field(
        default_factory=lambda: [ObjectConfig("coneSolid", position=(0.0, 1.5, -0.1))]
    )
    scene_id: Union[int, None] = field(default="None")
    seed: int = field(default=42)
    data_path: str = os.path.expanduser("~/tbp/data/habitat/objects/ycb")


@dataclass
class EnvInitArgsTouchViewMount(EnvInitArgs):
    agents: List[HabitatAgent] = field(
        default_factory=lambda: [TouchAgent(**TouchAndViewFinderMountConfig().__dict__)]
    )


@dataclass
class TouchViewFinderMountHabitatDatasetArgs:
    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsTouchViewMount().__dict__
    )

    def __post_init__(self):
        self.transform = DepthTo3DLocationsTouch(
            agent_id=self.env_init_args["agents"][0].agent_id,
            sensor_ids=self.env_init_args["agents"][0].sensor_ids,
            resolutions=self.env_init_args["agents"][0].resolutions,
            world_coord=True,
            zooms=self.env_init_args["agents"][0].zooms,
            get_all_points=True,
            clip_value=0.05,
        )


class DefaultObjectInitializer:
    def __call__(self):
        # TODO: does it work with xyz, 360?
        euler_rotation = np.random.uniform(0, 360, 3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        return dict(
            rotation=quat_rotation,
            euler_rotation=euler_rotation,
            position=(np.random.uniform(-0.5, 0.5), 0.0, 0.0),
            scale=[1.0, 1.0, 1.0],
        )

    def post_epoch(self):
        pass

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class PredefinedObjectInitializer(DefaultObjectInitializer):
    # TODO: validate length (n_poses==n_epochs). Maybe just warning?
    def __init__(self, positions=None, rotations=None, scales=None):

        self.positions = positions or [[0.0, 1.5, 0.0]]
        self.rotations = rotations or [[0.0, 0.0, 0.0], [0.0, 45.0, 0.0]]
        self.scales = scales or [[1.0, 1.0, 1.0]]
        self.current_epoch = 0

    def __call__(self):
        q = Rotation.from_euler(
            "xyz",
            self.rotations[self.current_epoch % len(self.rotations)],
            degrees=True,
        ).as_quat()
        quat_rotation = scipy_to_numpy_quat(q)
        return dict(
            rotation=quat_rotation,
            euler_rotation=self.rotations[self.current_epoch % len(self.rotations)],
            position=self.positions[self.current_epoch % len(self.positions)],
            scale=self.scales[self.current_epoch % len(self.scales)],
            semantic_id=1,
        )

    def post_epoch(self):
        self.current_epoch += 1


@dataclass
class EnvironmentDataloaderPerObjectArgs:

    object_names: List
    object_init_sampler: Callable


def get_object_names_by_idx(start, stop, list_of_indices=None):

    if isinstance(list_of_indices, list):
        if len(list_of_indices) > 0:
            return [SHUFFLED_YCB_OBJECTS[i] for i in list_of_indices]

    else:
        return SHUFFLED_YCB_OBJECTS[start:stop]


def get_env_dataloader_per_object_by_idx(start, stop, list_of_indices=None):

    return EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(start, stop, list_of_indices),
        object_init_sampler=PredefinedObjectInitializer(),
    )


@dataclass
class TouchAndViewFinderMountConfig:
    """
    Adaptation of Viviane's code that use the view finder to navigate so
    the object is in view before the real experiment happens + touch sensor
    """

    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: ["patch", "view_finder"]
    )
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.1])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[8, 8], [256, 256]]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.03]]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [True, True]
    )
    zooms: List[float] = field(default_factory=lambda: [6.0, 1.0])
