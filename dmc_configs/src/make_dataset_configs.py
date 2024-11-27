from dataclasses import dataclass, field
from typing import Dict, List, Union

from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvInitArgs,
    AgentConfig,
    MultiSensorAgent,
    MultiLMMountHabitatDatasetArgs,
    MultiLMMountConfig,
    EnvInitArgsFiveLMMount,
)


@dataclass
class EnvInitArgsMultiLMMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, MultiLMMountConfig().__dict__)
        ]
    )


class MultiLMMountConfig:
    # Modified from `PatchAndViewFinderMountConfig`
    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: ["patch_0", "patch_1", "view_finder"]
    )
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.2])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[64, 64], [64, 64], [64, 64]]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [True, True, True]
    )
    zooms: List[float] = field(default_factory=lambda: [10.0, 10.0, 1.0])


# - Two LMs (non-hierarchical). Basically just uses the base classes for everything,
# but they are given their own names here for clarity.


@dataclass
class TwoLMMountHabitatDatasetArgs(MultiLMMountHabitatDatasetArgs):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsTwoLMMount().__dict__
    )


@dataclass
class EnvInitArgsTwoLMMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, TwoLMMountConfig().__dict__)
        ]
    )


@dataclass
class TwoLMMountConfig(MultiLMMountConfig):
    """Two LMs, non-hierarchical.
    This class does nothing beyond `MultiLMMountConfig`, but I added it to be
    consistent with the convention of naming the classes after the number of LMs.
    """


# - Two LMs (hierarchical, distant agent)


@dataclass
class TwoLMStackedDistantMountHabitatDatasetArgs(MultiLMMountHabitatDatasetArgs):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsTwoLMDistantStackedMount().__dict__
    )


@dataclass
class EnvInitArgsTwoLMDistantStackedMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, TwoLMStackedDistantMountConfig().__dict__)
        ]
    )


@dataclass
class TwoLMStackedDistantMountConfig:
    # two sensor patches at the same location with different receptive field sizes
    # Used for basic test with heterarchy.
    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: ["patch_0", "patch_1", "view_finder"]
    )
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.2])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[64, 64], [64, 64], [64, 64]]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [True, True, True]
    )
    zooms: List[float] = field(default_factory=lambda: [10.0, 5.0, 1.0])


# - Two LMs (hierarchical, surface agent)


@dataclass
class TwoLMStackedSurfaceMountHabitatDatasetArgs(MultiLMMountHabitatDatasetArgs):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsTwoLMSurfaceStackedMount().__dict__
    )


@dataclass
class EnvInitArgsTwoLMSurfaceStackedMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, TwoLMStackedSurfaceMountConfig().__dict__)
        ]
    )


@dataclass
class TwoLMStackedSurfaceMountConfig(TwoLMStackedDistantMountConfig):
    action_space_type: str = "surface_agent"


# - Five LMs


@dataclass
class FiveLMMountHabitatDatasetArgs(MultiLMMountHabitatDatasetArgs):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsFiveLMMount().__dict__
    )


class FiveLMMountConfig:
    # Modified from `PatchAndViewFinderMountConfig`
    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: [
            "patch_0",
            "patch_1",
            "patch_2",
            "patch_3",
            "patch_4",
            "view_finder",
        ]
    )
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.2])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
        ]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, -0.01, 0.0],
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [True, True, True, True, True, True]
    )
    zooms: List[float] = field(
        default_factory=lambda: [10.0, 10.0, 10.0, 10.0, 10.0, 1.0]
    )


# - Nine LMs


@dataclass
class NineLMMountHabitatDatasetArgs(MultiLMMountHabitatDatasetArgs):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsNineLMMount().__dict__
    )


@dataclass
class EnvInitArgsNineLMMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, NineLMMountConfig().__dict__)
        ]
    )


@dataclass
class NineLMMountConfig:
    # Modified from `PatchAndViewFinderMountConfig`
    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: [
            "patch_0",
            "patch_1",
            "patch_2",
            "patch_3",
            "patch_4",
            "patch_5",
            "patch_6",
            "patch_7",
            "patch_8",
            "view_finder",
        ]
    )
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.2])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],  # view finder
        ]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],  # centered
            [0.0, 0.01, 0.0],  # 1 cm above
            [0.0, -0.01, 0.0],  # 1 cm below
            [0.01, 0.0, 0.0],  # 1 cm right
            [-0.01, 0.0, 0.0],  # 1 cm left
            [0.01, 0.01, 0.0],  # 1 cm right, 1 cm up
            [-0.01, 0.01, 0.0],  # 1 cm left, 1 cm up
            [0.01, -0.01, 0.0],  # 1 cm right, 1 cm down
            [-0.01, -0.01, 0.0],  # 1 cm left, 1 cm down
            [0.0, 0.0, 0.0],  # centered (view finder)
        ]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],  # view finder
        ]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,  # view finder
        ]
    )
    zooms: List[float] = field(
        default_factory=lambda: [
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            1.0,  # view finder
        ]
    )


# - Ten LMs


@dataclass
class TenLMMountHabitatDatasetArgs(MultiLMMountHabitatDatasetArgs):
    env_init_args: Dict = field(
        default_factory=lambda: EnvInitArgsTenLMMount().__dict__
    )


@dataclass
class EnvInitArgsTenLMMount(EnvInitArgs):
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, TenLMMountConfig().__dict__)
        ]
    )


@dataclass
class TenLMMountConfig:
    # Modified from `PatchAndViewFinderMountConfig`
    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: [
            "patch_0",
            "patch_1",
            "patch_2",
            "patch_3",
            "patch_4",
            "patch_5",
            "patch_6",
            "patch_7",
            "patch_8",
            "patch_9",
            "view_finder",
        ]
    )

    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.2])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],
            [64, 64],  # view finder
        ]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],  # centered
            [0.0, 0.01, 0.0],  # 1 cm above
            [0.0, -0.01, 0.0],  # 1 cm below
            [0.01, 0.0, 0.0],  # 1 cm right
            [-0.01, 0.0, 0.0],  # 1 cm left
            [0.01, 0.01, 0.0],  # 1 cm right, 1 cm up
            [-0.01, 0.01, 0.0],  # 1 cm left, 1 cm up
            [0.01, -0.01, 0.0],  # 1 cm right, 1 cm down
            [-0.01, -0.01, 0.0],  # 1 cm left, 1 cm down
            [0.0, 0.0, 0.0],  # centered (zoomed out)
            [0.0, 0.0, 0.0],  # centered (view finder)
        ]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,  # view finder
        ]
    )
    zooms: List[float] = field(
        default_factory=lambda: [
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            5.0,
            1.0,  # view finder
        ]
    )
