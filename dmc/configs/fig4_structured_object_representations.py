# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 4: Structured Object Representations

This module defines the following experiment:
 - `surf_agent_1lm_randrot_noise_10simobj`

 Experiments use:
 - 10 similar objects (but using the 77-object pretrained model)
 - 5 random rotations
 - Sensor noise
 - Hypothesis-testing policy active
 - No voting

This experiment should be run in serial due to the memory needs of detailed logging.
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
from .fig6_rapid_inference_with_model_based_policies import (
    surf_agent_1lm_randrot_noise,
)


class LastMaxEvidenceHandler(SelectiveEvidenceHandler):
    """Logging handler that stores terminal evidence values for each object's MLH.

    This handler stores the evidence value of the most-likely hypothesis for each
    object for only the final step of an episode.
    """

    def report_episode(
        self,
        data: Mapping,
        output_dir: str,
        episode: int,
        mode: str = "train",
        **kwargs,
    ) -> None:
        """Store only maxima of final evidence values."""

        # Initialize output data.
        episode_total, buffer_data = self.init_buffer_data(
            data, episode, mode, **kwargs
        )
        self.take_last_evidences(buffer_data)

        # Only store maximum of final evidence values.
        evidences_ls = buffer_data["LM_0"]["evidences_ls"]
        output_data = {"LM_0": {}}
        output_data["LM_0"]["max_evidences_ls"] = {
            obj: np.max(arr) for obj, arr in evidences_ls.items()
        }
        self.save(episode_total, output_data, output_dir)


surf_agent_1lm_randrot_noise_10simobj = deepcopy(surf_agent_1lm_randrot_noise)
surf_agent_1lm_randrot_noise_10simobj["logging_config"] = (
    SelectiveEvidenceLoggingConfig(
        run_name="surf_agent_1lm_randrot_noise_10simobj",
        monty_handlers=[
            BasicCSVStatsHandler,
            LastMaxEvidenceHandler,
        ],
    )
)
surf_agent_1lm_randrot_noise_10simobj[
    "eval_dataloader_args"
].object_names = SIMILAR_OBJECTS

CONFIGS = {
    "surf_agent_1lm_randrot_noise_10simobj": surf_agent_1lm_randrot_noise_10simobj,
}
