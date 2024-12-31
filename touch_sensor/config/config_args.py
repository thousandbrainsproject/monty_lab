# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Union

from classes.policy import TouchPolicy
from classes.sensor_module import HabitatTouchPatchSM
from tbp.monty.frameworks.environments.habitat import HabitatActionSpace
from tbp.monty.frameworks.models.monty_base import LearningModuleBase, MontyBase
from tbp.monty.frameworks.models.motor_policies import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import DetailedLoggingSM

# -- Table of contents --
# -----------------------
# Motor System Configurations
# Monty Configuration
# -----------------------

# -------------
# Motor System Data
# -------------

# Action Spaces


def agent_id_to_action_space(agent_id):
    return HabitatActionSpace(
        [
            f"{agent_id}.orient_horizontal",
            f"{agent_id}.orient_vertical",
            f"{agent_id}.move_forward",
            f"{agent_id}.move_tangentially",
        ]
    )


# Motor System Configurations
@dataclass
class InformedPolicyConfig:
    action_space: HabitatActionSpace
    action_to_dist: Dict
    action_to_params: Dict
    reverse_actions: List[str]
    file: Union[str, None] = None
    switch_frequency: float = 0.05
    min_perc_on_obj: float = 0.25
    good_view_percentage: float = 0.5
    desired_object_distance: float = 0.025
    alpha: float = 0.1


# Informed relative
def agent_id_to_base_relative_const_informed_policy_config(agent_id):
    action_space = agent_id_to_action_space(agent_id)
    return InformedPolicyConfig(
        action_space=action_space,
        reverse_actions=None,
        action_to_dist=asdict(ConstRelativeActionAmountDistributionConfig()),
        action_to_params=asdict(ConstRelativeActionAmountParamConfig()),
        switch_frequency=1,
    )


@dataclass
class MotorSystemConfigInformed:

    motor_system_class: Callable = field(default=TouchPolicy)
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: agent_id_to_base_relative_const_informed_policy_config(
            "agent_id_0"
        )
    )


# ---------
# Configs for relative action spaces
# ---------


@dataclass
class ConstRelativeActionAmountDistributionConfig:

    move_forward: str = "const"
    orient_horizontal: str = "const"
    orient_vertical: str = "const"
    move_tangentially: str = "const"


def make_const_relative_action_amount_param_config(rotation_step, translation_step):
    """
    Specify rotation and translation, and this will make the config for you
    """
    return ConstRelativeActionAmountParamConfig(
        move_forward=translation_step,
        orient_horizontal=rotation_step,
        orient_vertical=rotation_step,
        move_tangentially=translation_step,
    )


@dataclass
class ConstRelativeActionAmountParamConfig:
    move_forward: Dict = field(default_factory=lambda: dict(amount=0.001))
    orient_horizontal: Dict = field(default_factory=lambda: dict(amount=10.0))
    orient_vertical: Dict = field(default_factory=lambda: dict(amount=5.0))
    move_tangentially: Dict = field(default_factory=lambda: dict(amount=0.004))


# -------------
# Monty Configurations
# -------------


@dataclass
class MontyArgs:
    """
    This dataclass includes instantiated class instances (of LM for example) and is used
    to instantiate a MontyClass.
    """

    learning_modules: List[Callable]
    sensor_modules: List[Callable]
    motor_system: MotorSystem
    monty_args: Dict
    sm_to_agent_dict: Dict
    sm_to_lm_matrix: Iterable[Iterable] = field(default_factory=lambda: [[]])
    lm_to_lm_vote_matrix: Iterable[Iterable] = field(
        default_factory=lambda: [[]]
    )
    # TODO: step args end up here too

    def __post_init__(self):

        # If coupling not specified, set to default based on n learning modules
        if self.sm_to_lm_matrix == [[]]:
            self.lm_to_lm_vote_matrix = [
                [] for _ in range(len(self.learning_modules))
            ]

        if self.lm_to_lm_vote_matrix == [[]]:
            self.lm_to_lm_vote_matrix = [
                [] for _ in range(len(self.learning_modules))
            ]

        # Validate number of learning modules
        lm_error = 0
        lm_error += len(self.sm_to_lm_matrix) != len(self.learning_modules)
        lm_error += len(self.sm_to_lm_matrix) != len(self.lm_to_lm_vote_matrix)
        if lm_error > 0:
            raise ValueError(
                "The lengths of learning_modules, sm_to_lm_matrix, and "
                "lm_to_lm_vote_matrix must match exactly"
            )


@dataclass
class MontyRunArgs:
    min_eval_steps: int = field(default=3)
    min_train_steps: int = field(default=3)


@dataclass
class MontyGraphRunArgs(MontyRunArgs):
    num_exploratory_steps: int = field(default=1_000)
    min_eval_steps: int = field(default=3)
    min_train_steps: int = field(default=3)


@dataclass
class MontyFeatureGraphRunArgs(MontyRunArgs):
    num_exploratory_steps: int = field(default=1_000)
    min_eval_steps: int = field(default=0)
    min_train_steps: int = field(default=0)


@dataclass
class MontyConfig:
    """
    Use this config to specify a monty architecture in an experiment config. The
    monty_parser code will convert the configs for learning modules etc. into
    instances, and call MontyArgs to instantiate a Monty instance.
    """

    monty_class: Callable
    learning_module_configs: Dict
    sensor_module_configs: Dict
    motor_system_config: Dict
    sm_to_agent_dict: Dict
    sm_to_lm_matrix: Dict
    lm_to_lm_vote_matrix: Dict
    monty_args: Dict


@dataclass
class TouchAndViewMontyConfig(MontyConfig):

    monty_class: Callable = MontyBase
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            learning_module_0=dict(
                learning_module_class=LearningModuleBase,
                learning_module_args=dict(),
            )
        )
    )
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatTouchPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "depth",
                        "point_normal",
                        "principal_curvatures",
                        "curvature_directions",
                        "gaussian_curvature",
                        "mean_curvature",
                    ],
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since
                # it is not connected to a learning module
                # (just used to visualize the touch agent)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigInformed
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
    lm_to_lm_vote_matrix: List = field(default_factory=lambda: [[]])
    monty_args: Union[Dict, dataclass] = field(default_factory=lambda: MontyRunArgs())
