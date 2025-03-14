# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""
Figure 4: Visualize 8-patch view finder
"""

import copy
import os
from numbers import Number
from pathlib import Path
from typing import (
    Container,
    List,
    Optional,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from data_utils import (
    DMC_ANALYSIS_DIR,
    DMC_RESULTS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    get_frequency,
    load_eval_stats,
    load_object_model,
)
from matplotlib.lines import Line2D
from plot_utils import (
    TBP_COLORS,
    add_legend,
    axes3d_clean,
    axes3d_set_aspect_equal,
    violinplot,
)

plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"

OUT_DIR = DMC_ANALYSIS_DIR / "fig6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
%matplotlib qt
def plot_curvature_guided_policy():
    exp_dir = VISUALIZATION_RESULTS_DIR / "fig6_curvature_guided_policy"
    detailed_stats_path = exp_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    locations = [obs["location"] for obs in stats["SM_0"]["processed_observations"]]
    locations = np.array(locations)[:14]

    # %matplotlib qt
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), subplot_kw={"projection": "3d"})

    model = load_object_model("dist_agent_1lm", "mug")
    ax.scatter(
        model.x,
        model.y,
        model.z,
        color=model.rgba,
        alpha=0.35,
        s=3,
        edgecolor="none",
    )

    ax.scatter(
        locations[:, 0],
        locations[:, 1],
        locations[:, 2],
        color=TBP_COLORS["blue"],
        alpha=1,
        s=20,
        marker="v",
        zorder=10,
    )
    ax.plot(
        locations[:, 0],
        locations[:, 1],
        locations[:, 2],
        color=TBP_COLORS["blue"],
        alpha=1,
        zorder=10,
        lw=2,
    )
    ax.set_proj_type("persp", focal_length=0.5)
    axes3d_clean(ax, grid_color=(1, 1, 1, 1))
    axes3d_set_aspect_equal(ax)
    ax.view_init(elev=54, azim=-36, roll=60)
    plt.show()
    fig.savefig(OUT_DIR / "curvature_guided_policy.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "curvature_guided_policy.svg", bbox_inches="tight")

plot_curvature_guided_policy()