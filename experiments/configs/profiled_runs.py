# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy

from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
    MontySupervisedObjectPretrainingExperiment,
    ProfileExperimentMixin,
)

from .config_utils import import_config_from_monty
from .evidence_evaluation import partial_rotation_eval_5lms_elm
from .graph_experiments import (
    evidence_tests,
    feature_pred_tests,
    five_lm_feature_matching,
)
from .more_pretraining_experiments import supervised_pre_training_storeall

randrot_noise_10distinctobj_5lms_dist_agent = import_config_from_monty(
    "ycb_experiments.py",
    "randrot_noise_10distinctobj_5lms_dist_agent",
)


class MontyObjectRecognitionProfiledExperiment(
    ProfileExperimentMixin, MontyObjectRecognitionExperiment
):
    pass


class MontySupervisedObjectPretrainingProfiledExperiment(
    ProfileExperimentMixin, MontySupervisedObjectPretrainingExperiment
):
    pass


supervised_pre_training_base_dev = copy.deepcopy(
    supervised_pre_training_storeall
)
supervised_pre_training_base_dev.update(
    experiment_class=MontySupervisedObjectPretrainingProfiledExperiment
)
supervised_pre_training_base_dev["monty_config"].monty_args.num_exploratory_steps = 20

feature_pred_tests_dev = copy.deepcopy(feature_pred_tests)
feature_pred_tests_dev.update(experiment_class=MontyObjectRecognitionProfiledExperiment)

evidence_profiled = copy.deepcopy(evidence_tests)
evidence_profiled.update(
    experiment_class=MontyObjectRecognitionProfiledExperiment
)

five_lm_feature_matching_dev = copy.deepcopy(five_lm_feature_matching)
five_lm_feature_matching_dev.update(
    experiment_class=MontyObjectRecognitionProfiledExperiment
)

five_lm_evidence_profiled = copy.deepcopy(partial_rotation_eval_5lms_elm)
five_lm_evidence_profiled.update(
    experiment_class=MontyObjectRecognitionProfiledExperiment
)

# NOTE: its best to set use_multithreading=False when running the profiler
# since it doesn't track the threaded tasks correctly.
randrot_noise_10distinctobj_5lms_dist_agent_profiled = copy.deepcopy(
    randrot_noise_10distinctobj_5lms_dist_agent
)
randrot_noise_10distinctobj_5lms_dist_agent_profiled.update(
    experiment_class=MontyObjectRecognitionProfiledExperiment
)

CONFIGS = dict(
    feature_pred_tests_dev=feature_pred_tests_dev,
    evidence_profiled=evidence_profiled,
    supervised_pre_training_base_dev=supervised_pre_training_base_dev,
    five_lm_evidence_profiled=five_lm_evidence_profiled,
    randrot_noise_10distinctobj_5lms_dist_agent_profiled=randrot_noise_10distinctobj_5lms_dist_agent_profiled,  # noqa E501
)
