# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

sys.path.append("./")
from utils import checkdir  # noqa: E402

sns.set(style="dark", context="talk")


def plot_box(log_path, exp_ids, var, var_name, save_path, show_ax2=False):
    """
    plots a bar chart for the final overlap error averaged over some steps.
    This compares different setting for different experiments on the same plot.
    """

    log_paths = [
        os.path.join(log_path, f"exp_{str(exp_id).zfill(3)}") for exp_id in exp_ids
    ]

    # get logs
    df_dict = {}
    for log_path in log_paths:
        files = glob(os.path.join(log_path, "*/pth/overlap.json"))
        overlap = [json.load(open(file, "r"))["overlap_distance"] for file in files]
        config_file = glob(os.path.join(log_path, "*/pth/configs.yaml"))[0]
        configs = yaml.safe_load(open(config_file, "r"))
        df_dict[sum(configs[var])] = overlap

    df = pd.DataFrame(df_dict)
    fig, ax1 = plt.subplots()
    sns.boxplot(data=df, ax=ax1)
    ax1.set_ylabel("Normalized Overlap Error")
    ax1.set_xlabel(var_name)

    # Add a second y-axis (twin axis)
    if show_ax2:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Overlap error")
        ylim = ax1.get_ylim()
        ax2.set_ylim(ylim[0] * 41, ylim[1] * 41)

    plt.savefig(save_path, bbox_inches="tight")


# run plots
save_dir = "results/single_plots"
checkdir(save_dir)  # This will delete plots in the save folder


# example of plot box usage
plot_box(
    log_path="logs",
    exp_ids=[1, 2, 3],
    var="num_objects",
    var_name="Number of Objects",
    save_path=os.path.join(save_dir, "vary_M_box.png"),
    show_ax2=True,
)
