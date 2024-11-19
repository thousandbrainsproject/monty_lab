# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import os

import torch
from tbp.monty.frameworks.config_utils.config_args import PatchAndViewMontyConfig
from tbp.monty.frameworks.config_utils.monty_parser import create_monty_instance

cfg = PatchAndViewMontyConfig()
model_class = cfg.monty_class
model = create_monty_instance(cfg)
model_path = os.path.expanduser(
    "~/tbp/tbp.monty/projects/monty_runs/pretrained_models_ycb/supervised_pre_training/7/model.pt"  # noqa E501
)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
