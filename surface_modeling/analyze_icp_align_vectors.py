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
import numpy as np
import torch
from analyze_pose_optimization import (
    get_error_per_exp,
    get_euler_angles_per_transform,
    get_time_per_exp,
)

# TODO: code for analyzing all runs from run_single_axis_rotation_exps.sh

path = os.path.expanduser("~/tbp/tbp.monty/projects/monty_runs/logs")

icp_x_path = os.path.join(path, "icp_x", "stats.pt")
icp_x_1_path = os.path.join(path, "icp_x_1_step", "stats.pt")

icp_av_x_path = os.path.join(path, "icp_align_vectors_x", "stats.pt")
icp_av_x_1_path = os.path.join(path, "icp_x_align_vectors_1_step", "stats.pt")

av_x_path = os.path.join(path, "align_vectors_x", "stats.pt")

paths = [icp_x_path,
         icp_x_1_path,
         icp_av_x_path,
         icp_av_x_1_path,
         av_x_path]

for p in paths:
    assert os.path.exists(p)

results = dict(
    icp=torch.load(icp_x_path),
    icp1=torch.load(icp_x_1_path),

    icpav=torch.load(icp_av_x_path),
    icpav1=torch.load(icp_av_x_1_path),

    av=torch.load(av_x_path),
)

analysis_dir = os.path.expanduser("~/tbp/tbp.monty/projects/surface_modeling/results")
os.makedirs(analysis_dir, exist_ok=True)

int_to_xyz = {i: j for i, j in zip([0, 1, 2], ["x", "y", "z"])}
k_2_error = dict()
k_2_angles = dict()
k_2_time = dict()
k_2_exp_name = dict(

    icp="ICP, manual SVD, 20 steps",
    icp1="ICP, manual SVD, 1 step",

    icpav="ICP, scipy align_vectors, 20 steps",
    icpav1="ICP, scipy align_vectors, 1 step",

    av="No ICP, scipy align_vectors"
)

for k in results.keys():

    data = results[k]
    angles = get_euler_angles_per_transform(data)
    all_errors = get_error_per_exp(data)
    t = get_time_per_exp(data)
    k_2_error[k] = all_errors
    k_2_angles[k] = angles
    k_2_time[k] = t


fig, ax = plt.subplots(figsize=(8, 8))

for k in k_2_error.keys():

    all_errors = k_2_error[k]
    rots = k_2_angles[k][:, 0]
    em = all_errors.mean(axis=1)
    ev = all_errors.var(axis=1)
    ax.errorbar(rots,
                em,
                yerr=ev,
                marker=".",
                ls="none",
                label=k_2_exp_name[k],
                alpha=0.5)

ax.set_ylabel("Mean pointwise error")
ax.set_xlabel("Degrees of rotation")
ax.set_title("Rotation vs Error: Multiple Alignment Methods")
plt.legend()

plt.savefig(os.path.join(analysis_dir, "ICP_vs_align_vectors.png"))

kl = np.array(list(k_2_time.keys()))
mean_times = np.array([k_2_time[k].mean() for k in kl])
var_times = np.array([k_2_time[k].var() for k in kl])
ksrt = np.argsort(mean_times)
mnames = [k_2_exp_name[kl[i]] for i in ksrt]

fig, ax = plt.subplots()

ax.bar(mnames, height=mean_times[ksrt], yerr=var_times[ksrt])
ax.set_xticklabels(mnames, rotation=70)
ax.set_ylabel("Time (s) per sample")
ax.set_title("Speed of different pose estimation methods")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "speed_of_methods.png"))
