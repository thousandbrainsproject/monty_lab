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
    plot_error_vs_2_transform_degrees,
    plot_error_vs_3_transform_degrees,
    plot_error_vs_transform_degrees,
    plot_mean_error_per_class,
)

path = os.path.expanduser("~/tbp/tbp.monty/projects/monty_runs/logs")
av_x_path = os.path.join(path, "align_vectors_x", "stats.pt")
icp_av_x_path = os.path.join(path, "icp_align_vectors_x", "stats.pt")
icp_x_path = os.path.join(path, "icp_x", "stats.pt")
icp_xy_path = os.path.join(path, "icp_xy", "stats.pt")
icp_xyz_path = icp_xy_path = os.path.join(path, "icp_xyz", "stats.pt")
icp_rx_path = os.path.join(path, "icp_rigid_body_xrot", "stats.pt")
icp_rxy_path = os.path.join(path, "icp_rigid_body_xyrot", "stats.pt")
icp_rxyz_path = os.path.join(path, "icp_rigid_body_xyzrot", "stats.pt")


paths = [av_x_path,
         icp_x_path,
         icp_av_x_path,
         icp_xy_path,
         icp_xyz_path,
         icp_rx_path,
         icp_rxy_path,
         icp_rxyz_path]

for p in paths:
    assert os.path.exists(p)

results = dict(
    icpavx=torch.load(icp_av_x_path),
    avx=torch.load(av_x_path),
    x=torch.load(icp_x_path),
    xy=torch.load(icp_xy_path),
    xyz=torch.load(icp_xyz_path),
    rx=torch.load(icp_rx_path),
    rxy=torch.load(icp_rxy_path),
    rxyz=torch.load(icp_rxyz_path),
)

analysis_dir = os.path.expanduser("~/tbp/tbp.monty/projects/surface_modeling/results")
os.makedirs(analysis_dir, exist_ok=True)

for k, v in results.items():
    plot_mean_error_per_class(v, exp_name=f"Exp = {k}")
    plt.savefig(os.path.join(analysis_dir, f"icp_error_per_class_{k}.png"))


int_to_xyz = {i: j for i, j in zip([0, 1, 2], ["x", "y", "z"])}

for k in results.keys():

    data = results[k]
    angles = get_euler_angles_per_transform(data)
    all_errors = get_error_per_exp(data)
    plot_error_vs_transform_degrees(
        all_errors,
        angles[:, 0],
        exp_name=f"Exp = {k}, rotation=x"
    )
    plt.savefig(os.path.join(analysis_dir, f"icp_error_per_rotation_{k}.png"))


for k in ["xy", "rxy"]:

    data = results[k]
    angles = get_euler_angles_per_transform(data)
    all_errors = get_error_per_exp(data)
    plot_error_vs_2_transform_degrees(
        all_errors,
        angles[:, 0],
        angles[:, 1],
        "x",
        "y",
        exp_name=k)
    plt.savefig(os.path.join(analysis_dir, f"icp_error_per_rotation_2d_{k}.png"))


for k in ["xyz", "rxyz"]:

    data = results[k]
    angles = get_euler_angles_per_transform(data)
    all_errors = get_error_per_exp(data)
    plot_error_vs_3_transform_degrees(
        all_errors,
        angles,
        ["x", "y", "z"],
        exp_name=k)
    plt.savefig(os.path.join(analysis_dir, f"icp_error_per_rotation_3d_{k}.png"))
