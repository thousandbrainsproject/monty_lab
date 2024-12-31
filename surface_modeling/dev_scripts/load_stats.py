# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os

from tbp.monty.frameworks.utils.logging_utils import load_stats

exp_path = os.path.expanduser(
    "~/tbp/tbp.monty/projects/monty_runs/logs/feature_pred_tests/"
)
outputs = load_stats(exp_path)
