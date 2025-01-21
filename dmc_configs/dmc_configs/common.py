import copy
import os
from pathlib import Path

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigCurvatureInformedSurface,
    MotorSystemConfigNaiveScanSpiral,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    SurfaceAndViewMontyConfig,
    get_cube_face_and_corner_views_rotations,
    make_multi_lm_flat_dense_connectivity,
    make_multi_lm_monty_config,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PredefinedObjectInitializer,
    SurfaceViewFinderMountHabitatDatasetArgs,
    make_multi_sensor_habitat_dataset_args,
    make_sensor_positions_on_grid,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS, SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatDistantPatchSM,
    HabitatSurfacePatchSM,
)

__all__ = [
    "DMC_ROOT",
    "PRETRAIN_DIR",
]

# - Path setup
DMC_ROOT = Path("~/tbp/results/dmc").expanduser()
PRETRAIN_DIR = DMC_ROOT / "pretrained_models"

# monty_models_dir = os.getenv("MONTY_MODELS")
# if not monty_models_dir:
#     monty_models_dir = "~/tbp/results/monty/pretrained_models"
# PRETRAIN_DIR = Path(monty_models_dir).expanduser() / "pretrained_ycb_dmc"
