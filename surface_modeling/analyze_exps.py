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

import matplotlib.pyplot as plt
import torch
from analyze_pose_optimization import (
    get_error_per_exp,
    get_euler_angles_per_transform,
    plot_error_vs_transform_degrees,
    plot_mean_error_per_class,
)

path = os.path.expanduser("~/tbp/tbp.monty/projects/monty_runs/logs")
icp_path = os.path.join(path, "icp_align_vectors_x", "stats.pt")
data = torch.load(icp_path)

plot_mean_error_per_class(data)
plt.show()

angles = get_euler_angles_per_transform(data)
all_errors = get_error_per_exp(data)
plot_error_vs_transform_degrees(all_errors, angles[:, 0])
plt.show()
