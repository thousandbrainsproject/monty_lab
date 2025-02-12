# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 9: Structured Object Representations

This module defines the following experiments:
 - `dist_agent_1lm_randrot_noise_10simobj`

 Experiments use:
 - 10 similar objects (but using the 77-object pretrained model)
 - 5 random rotations
 - Sensor noise
 - Hypothesis-testing policy active
 - No voting
 - SELECTIVE evidence logging
 - Probably best run in serial.
"""

import copy
import datetime
import importlib
import logging
import os
import pprint

from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.environments.ycb import SIMILAR_OBJECTS
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.loggers.exp_logger import (
    BaseMontyLogger,
    LoggingCallbackHandler,
)
from tbp.monty.frameworks.loggers.monty_handlers import DetailedJSONHandler
from tbp.monty.frameworks.loggers.wandb_handlers import WandbWrapper
from tbp.monty.frameworks.models.monty_base import MontyBase
from tbp.monty.frameworks.utils.dataclass_utils import (
    get_subset_of_args,
)

from .common import DMC_RESULTS_DIR
from .fig4_rapid_inference_with_voting import dist_agent_1lm_randrot_noise

"""
We have to use the following class to enable selective logging of evidence or 
else apply this change to tbp.monty. Some details:

Expected behavior:
- Set `monty_log_level` to `"SELECTIVE"`, causing `MontyExperiment` to use the 
  `SelectiveEvidenceLogger`, and specify we want a `DetailedJSONHandler` to store
  the results.
- The `SelectiveEvidenceLogger` adds evidence-specific data and removes or filters 
  unwanted data (i.e., sensor module's `raw_observations`).

Actual behavior:
- When `MontyExperiment.init_loggers` sees we have a `DetailedJSONHandler`, it 
  infers that the actual `monty_log_level` is `"DETAILED"`, and uses 
  `DetailedGraphMatchingLogger` instead of `SelectiveEvidenceLogger`.

To get around this, the following class overrides `init_loggers` which does everything
a detailed log level would require (i.e., telling sensor modules and learning modules
to store detailed data) without upgrading to "DETAILED". This patch could be applied
to tbp.monty.
"""


class EvidenceLoggingMontyObjectRecognitionExperiment(MontyObjectRecognitionExperiment):
    def init_loggers(self, logging_config):
        """Initialize logger with specified log level."""
        # Add experiment config so config can be passed to wandb
        all_logging_args = logging_config
        # all_logging_args.update(config=self.config)

        # Unpack individual logging arguments
        self.monty_log_level = all_logging_args["monty_log_level"]
        self.monty_handlers = all_logging_args["monty_handlers"]
        self.wandb_handlers = all_logging_args["wandb_handlers"]
        self.python_log_level = all_logging_args["python_log_level"]
        self.log_to_file = all_logging_args["python_log_to_file"]
        self.log_to_stdout = all_logging_args["python_log_to_stdout"]
        self.output_dir = all_logging_args["output_dir"]
        self.run_name = all_logging_args["run_name"]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # If basic config has been set by a previous experiment, ipython, code editor,
        # or anything else, the config will not be properly set. importlib.reload gets
        # around this and ensures
        importlib.reload(logging)

        # Create basic python logging handlers
        python_logging_handlers = []
        if self.log_to_file:
            python_logging_handlers.append(
                logging.FileHandler(os.path.join(self.output_dir, "log.txt"), mode="w")
            )
        if self.log_to_stdout:
            python_logging_handlers.append(logging.StreamHandler())

        # Configure basic python logging
        logging.basicConfig(
            level=self.python_log_level,
            handlers=python_logging_handlers,
        )
        logging.info(f"Logger initialized at {datetime.datetime.now()}")
        logging.debug(pprint.pformat(self.config))

        # Configure Monty logging
        monty_handlers = []
        has_detailed_logger = False
        for handler in self.monty_handlers:
            if handler.log_level() == "DETAILED":
                has_detailed_logger = True
            handler_args = get_subset_of_args(all_logging_args, handler.__init__)
            monty_handler = handler(**handler_args)
            monty_handlers.append(monty_handler)

        # Configure wandb logging
        if len(self.wandb_handlers) > 0:
            wandb_args = get_subset_of_args(all_logging_args, WandbWrapper.__init__)
            wandb_args.update(
                config=self.config,
                run_name=wandb_args["run_name"] + "_" + wandb_args["wandb_id"],
            )
            monty_handlers.append(WandbWrapper(**wandb_args))
            for handler in self.wandb_handlers:
                if handler.log_level() == "DETAILED":
                    has_detailed_logger = True

        # Only upgrade to DETAILED if the logging level is "less than" SELECTIVE.
        if has_detailed_logger and self.monty_log_level not in (
            "SELECTIVE",
            "DETAILED",
        ):
            logging.warning(
                f"Log level is set to {self.monty_log_level} but you "
                "specified a detailed logging handler. Setting log level "
                "to detailed."
            )
            self.monty_log_level = "DETAILED"

        if (
            self.monty_log_level not in ("SELECTIVE", "DETAILED")
            and not has_detailed_logger
        ):
            logging.warning(
                "You are setting the monty logging level to DETAILED, but all your"
                "handlers are BASIC. Consider setting the level to BASIC, or adding a"
                "DETAILED handler"
            )

        for lm in self.model.learning_modules:
            lm.has_detailed_logger = has_detailed_logger

        if has_detailed_logger or self.show_sensor_output:
            # If we log detailed stats we want to save sm raw obs by default.
            for sm in self.model.sensor_modules:
                sm.save_raw_obs = True

        # monty_log_level determines if we used Basic or Detailed logger
        # TODO: only defined for MontyForGraphMatching right now, need to add TM later
        # NOTE: later, more levels that Basic or Detailed could be added

        if isinstance(self.model, MontyBase):
            if self.monty_log_level in self.model.LOGGING_REGISTRY:
                logger_class = self.model.LOGGING_REGISTRY[self.monty_log_level]
                self.monty_logger = logger_class(handlers=monty_handlers)

            else:
                logging.warning(
                    "Unable to match monty logger to log level"
                    "An empty logger will be used as a placeholder"
                )
                self.monty_logger = BaseMontyLogger(handlers=[])
        else:
            raise (
                NotImplementedError,
                "Please implement a mapping from monty_log_level to a logger class"
                f"for models of type {type(self.model)}",
            )

        if "log_parallel_wandb" in all_logging_args.keys():
            self.monty_logger.use_parallel_wandb_logging = all_logging_args[
                "log_parallel_wandb"
            ]
        # Instantiate logging callback handler for custom monty loggers
        self.logger_handler = LoggingCallbackHandler(
            self.monty_logger, self.model, output_dir=self.output_dir
        )


dist_agent_1lm_randrot_noise_10simobj = copy.deepcopy(dist_agent_1lm_randrot_noise)
dist_agent_1lm_randrot_noise_10simobj["experiment_class"] = (
    EvidenceLoggingMontyObjectRecognitionExperiment
)
dist_agent_1lm_randrot_noise_10simobj["logging_config"] = (
    DetailedEvidenceLMLoggingConfig(
        output_dir=str(DMC_RESULTS_DIR),
        run_name="dist_agent_1lm_randrot_noise_10simobj",
        wandb_group="dmc",
        monty_log_level="SELECTIVE",
    )
)
dist_agent_1lm_randrot_noise_10simobj[
    "eval_dataloader_args"
].object_names = SIMILAR_OBJECTS

CONFIGS = {
    "dist_agent_1lm_randrot_noise_10simobj": dist_agent_1lm_randrot_noise_10simobj,
}
