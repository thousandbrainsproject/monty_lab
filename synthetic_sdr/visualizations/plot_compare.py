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


def plot_box(log_paths, exp_ids, var, var_name, save_path, show_ax2=False):
    """
    plots a bar chart for the final overlap error averaged over some steps.

    * This compares different setting for different experiments for different
    algorithm on the same plot.

    * We used this to compare the old way of encoding with an AutoEncoder to
    the new parameterless method.
    """

    df_dicts = []
    for log_path in log_paths:
        exp_paths = [os.path.join(log_path, f"exp_{str(i).zfill(3)}") for i in exp_ids]

        df_dict = {}
        for exp_path in exp_paths:
            files = glob(os.path.join(exp_path, "*/pth/overlap.json"))
            overlap = [json.load(open(file, "r"))["overlap_distance"] for file in files]
            config_file = glob(os.path.join(log_path, "*/pth/configs.yaml"))[0]
            configs = yaml.safe_load(open(config_file, "r"))
            df_dict[sum(configs[var])] = overlap

        df_dicts.append(df_dict)

    # concatenate the dataframes
    dfs = [pd.DataFrame(df) for df in df_dicts]
    cols = dfs[0].columns
    dfs[0]["Encoder"] = "baseline"
    dfs[1]["Encoder"] = "simplified"
    df = pd.concat(dfs)

    # melt the dataframe for seaborn
    df = pd.melt(df, id_vars="Encoder", value_vars=cols)

    fig, ax1 = plt.subplots()
    sns.boxplot(data=df, x="variable", y="value", hue="Encoder", ax=ax1)
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
save_dir = "results/comparison_plots"
checkdir(save_dir)  # This will delete plots in the save folder


# exmple of plot_box usage to compare multiple experiments
plot_box(
    ["logs_backup/logs_2", "logs_backup/logs_3"],
    [1, 2, 3],
    "num_objects",
    "Number of Objects",
    save_path=os.path.join(save_dir, "vary_M_box.png"),
    show_ax2=True,
)
