# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 4: Structured Object Representations

This module defines the following experiments:
 - `dist_agent_1lm_randrot_noise_10simobj`

 Experiments use:
 - 10 similar objects (but using the 77-object pretrained model)
 - 5 random rotations
 - Sensor noise
 - Hypothesis-testing policy active
 - No voting
 - SELECTIVE evidence logging
 - Run in serial due to memory needed for detailed logging
"""

from copy import deepcopy
from typing import Mapping

import numpy as np
from tbp.monty.frameworks.environments.ycb import SIMILAR_OBJECTS
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler

from .common import (
    SelectiveEvidenceHandler,
    SelectiveEvidenceLoggingConfig,
)
from .fig5_rapid_inference_with_voting import dist_agent_1lm_randrot_noise
from .fig6_rapid_inference_with_model_based_policies import (
    surf_agent_1lm_randrot_noise,
)


class SimilarObjectsEvidenceHandler(SelectiveEvidenceHandler):
    """Logging handler that only saves terminal evidence data for the MLH object.

    A lean logger handler for the symmetry experiment (which are full-length runs,
    and so we need to be very selective about which data to log).

    """

    def report_episode(
        self,
        data: Mapping,
        output_dir: str,
        episode: int,
        mode: str = "train",
        **kwargs,
    ) -> None:
        """Store only final evidence data and no sensor data."""

        # Initialize output data.
        self.handler_args["last_evidence"] = True  # Required for this handler.
        episode_total, buffer_data = self.init_buffer_data(
            data, episode, mode, **kwargs
        )
        # Only store last evidence, and only for the 10 similar objects.
        evidences_ls = buffer_data["LM_0"]["evidences_ls"]
        output_data = {"LM_0": {}}
        output_data["LM_0"]["max_evidences_ls"] = {
            obj: np.max(arr) for obj, arr in evidences_ls.items()
        }
        self.save(episode_total, output_data, output_dir)


dist_agent_1lm_randrot_noise_10simobj = deepcopy(dist_agent_1lm_randrot_noise)
dist_agent_1lm_randrot_noise_10simobj["logging_config"] = (
    SelectiveEvidenceLoggingConfig(
        run_name="dist_agent_1lm_randrot_noise_10simobj",
        monty_handlers=[
            BasicCSVStatsHandler,
            SimilarObjectsEvidenceHandler,
        ],
    )
)
dist_agent_1lm_randrot_noise_10simobj[
    "eval_dataloader_args"
].object_names = SIMILAR_OBJECTS

surf_agent_1lm_randrot_noise_10simobj = deepcopy(surf_agent_1lm_randrot_noise)
surf_agent_1lm_randrot_noise_10simobj["logging_config"] = (
    SelectiveEvidenceLoggingConfig(
        run_name="surf_agent_1lm_randrot_noise_10simobj",
        monty_handlers=[
            BasicCSVStatsHandler,
            SimilarObjectsEvidenceHandler,
        ],
    )
)
surf_agent_1lm_randrot_noise_10simobj[
    "eval_dataloader_args"
].object_names = SIMILAR_OBJECTS

CONFIGS = {
    "dist_agent_1lm_randrot_noise_10simobj": dist_agent_1lm_randrot_noise_10simobj,
    "surf_agent_1lm_randrot_noise_10simobj": surf_agent_1lm_randrot_noise_10simobj,
}
