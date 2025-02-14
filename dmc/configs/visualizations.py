# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Configs for visualizations (not core experiments)

This file contains configs defined solely for making visualizations that go into
paper figures. The configs defined are:

- `visualize_8lm_patches`: An experiment that is used to save patch view of an object
for the 8-patch distant agent model. Only runs one episode (using the mug). The
output is read and plotted by `scripts/visualize_multilm_patches.py`.

"""
import os
from copy import deepcopy

from tbp.monty.frameworks.loggers.monty_handlers import DetailedJSONHandler

from .common import DMC_ROOT_DIR
from .fig4_rapid_inference_with_voting import dist_agent_8lm_half_lms_match

VISUALIZATIONS_DIR = os.path.join(DMC_ROOT_DIR, "visualizations")

config = deepcopy(dist_agent_8lm_half_lms_match)
config["logging_config"].run_name = "visualize_8lm_patches"
config["logging_config"].output_dir = VISUALIZATIONS_DIR
config["logging_config"].monty_handlers.append(DetailedJSONHandler)
config["experiment_args"].n_eval_epochs = 1
config["experiment_args"].max_total_steps = 1
config["experiment_args"].max_eval_steps = 1
config["monty_config"].monty_args.num_exploration_steps = 1
config["eval_dataloader_args"].object_names = ["mug"]
config["eval_dataloader_args"].object_init_sampler.rotations = [[0, 0, 0]]
# Set viewfinder resolution to 256 x 256 for a denser "background" image. The remaining
# patches uses the same resolution as the original 64 x 64 patches.
dataset_args = config["dataset_args"]
resolutions = [[64, 64]] * 9
resolutions[-1] = [256, 256]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = resolutions
dataset_args.__post_init__()


CONFIGS = {"visualize_8lm_patches": config}
