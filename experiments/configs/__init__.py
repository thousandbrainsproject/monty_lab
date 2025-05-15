# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from .base import CONFIGS as BASE
from .data_generators import CONFIGS as DATA_GENERATORS
from .evidence_evaluation import CONFIGS as EVIDENCEEVAL
from .evidence_sdr_evaluation import CONFIGS as EVIDENCESDREVAL
from .feature_matching_evaluation import CONFIGS as FEATUREEVAL
from .follow_ups import CONFIGS as FOLLOW_UPS
from .graph_experiments import CONFIGS as GRAPHS
from .policy_experiments import CONFIGS as POLICYEVAL
from .profiled_runs import CONFIGS as PROFILED_RUNS
from .robustness_experiments import CONFIGS as ROBUSTNESSEVAL
from .tbp_robot_lab import CONFIGS as TBP_ROBOT_LAB
from .view_finder_images import CONFIGS as VIEW_FINDER_IMAGES

CONFIGS = dict()
CONFIGS.update(BASE)
CONFIGS.update(PROFILED_RUNS)
CONFIGS.update(GRAPHS)
CONFIGS.update(DATA_GENERATORS)
CONFIGS.update(FEATUREEVAL)
CONFIGS.update(EVIDENCEEVAL)
CONFIGS.update(EVIDENCESDREVAL)
CONFIGS.update(POLICYEVAL)
CONFIGS.update(ROBUSTNESSEVAL)
CONFIGS.update(FOLLOW_UPS)
CONFIGS.update(VIEW_FINDER_IMAGES)
CONFIGS.update(TBP_ROBOT_LAB)