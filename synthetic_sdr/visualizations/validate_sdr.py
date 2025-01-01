# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

sys.path.append("./")
from utils import checkdir, set_seed  # noqa: E402

sns.set(style="dark", context="talk")
set_seed(0)


def binarize(emb, sdr_on_bits):
    topk_indices = torch.topk(emb, k=sdr_on_bits, dim=1)[1]
    mask = torch.full_like(emb, -1)
    mask.scatter_(1, topk_indices, 1.0)
    return mask


def pairwise_overlaps(x, y, convert_to_sdr=False):
    if convert_to_sdr:
        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0
    return x @ y.t()


def plot_overlaps(d):
    overlaps_df = pd.DataFrame.from_dict(d, orient="index").T

    fig, ax1 = plt.subplots()
    ax1.set_ylabel("Overlap with Object SDRs")
    ax1.set_xlabel("Number of Random SDRs")
    sns.boxplot(
        data=overlaps_df,
        ax=ax1,
        showmeans=True,
        meanline=True,
        meanprops=dict(linestyle="solid", linewidth=1, color="k"),
        medianprops=dict(linewidth=0),
    )


def plot_union(obj_sdrs, r_bin, union, save_dir):
    """
    plots the overlap of random SDRs with unions of trained SDRs
    """

    r_bin = (r_bin + 1.0) / 2.0
    obj_sdrs = (obj_sdrs + 1.0) / 2.0

    if union > 1:
        obj_sdrs = torch.clamp_max(
            obj_sdrs.reshape(union, -1, obj_sdrs.shape[-1]).sum(0), 1
        )
    overlaps = r_bin @ obj_sdrs.t()
    overlaps_dict = {n: overlaps[:n].reshape(-1).numpy().tolist() for n in ranges}
    plot_overlaps(overlaps_dict)
    plt.savefig(os.path.join(save_dir, f"sdr_union_{union}.png"), bbox_inches="tight")


def compare_plot(obj_sdrs1, obj_sdrs2, r_bin, union, save_dir):
    """
    makes a comparison between overlap of random SDRs with trained SDRs, and
    random SDRs with other random SDRs
    """

    r_bin = (r_bin + 1.0) / 2.0
    obj_sdrs1 = (obj_sdrs1 + 1.0) / 2.0
    obj_sdrs2 = (obj_sdrs2 + 1.0) / 2.0

    if union > 1:
        obj_sdrs1 = torch.clamp_max(
            obj_sdrs1.reshape(union, -1, obj_sdrs1.shape[-1]).sum(0), 1
        )
        obj_sdrs2 = torch.clamp_max(
            obj_sdrs2.reshape(union, -1, obj_sdrs2.shape[-1]).sum(0), 1
        )

    overlaps1 = r_bin @ obj_sdrs1.t()
    overlaps2 = r_bin @ obj_sdrs2.t()
    overlaps1_dict = {n: overlaps1[:n].reshape(-1).numpy().tolist() for n in ranges}
    overlaps2_dict = {n: overlaps2[:n].reshape(-1).numpy().tolist() for n in ranges}

    overlaps1_df = pd.DataFrame.from_dict(overlaps1_dict, orient="index").T
    overlaps2_df = pd.DataFrame.from_dict(overlaps2_dict, orient="index").T

    overlaps1_df["type"] = "Evidence SDR"
    overlaps2_df["type"] = "Random SDR"

    df = pd.concat([overlaps1_df, overlaps2_df])
    df = pd.melt(
        df,
        id_vars="type",
        value_vars=[10, 100, 1000, 10000],
        var_name="column",
        value_name="value",
    )

    fig, ax1 = plt.subplots()
    ax1.set_ylabel("Overlap with Object SDRs")
    ax1.set_xlabel("Number of Random SDRs")

    sns.boxplot(
        data=df,
        x="column",
        y="value",
        hue="type",
        ax=ax1,
        showmeans=True,
        meanline=True,
        meanprops=dict(linestyle="solid", linewidth=1, color="k"),
        medianprops=dict(linewidth=0),
    )

    plt.legend(title=None, ncols=2, bbox_to_anchor=(0.5, 1.1), loc="center")
    plt.savefig(
        os.path.join(save_dir, f"sdr_union_{union}_comparison.png"), bbox_inches="tight"
    )


save_dir = "results/sdr_validation"
checkdir(save_dir)  # This will delete plots in the save folder

# random sdrs
ranges = [10, 100, 1000, 10000]
r = torch.randn(ranges[-1], 2048)
r_bin = binarize(r, 41)

# obj sdrs
exp_path = "logs/exp_004/seed_000/pth"
preds = torch.load(os.path.join(exp_path, "preds.pth"))
obj_sdrs = preds["sparse"][-1]

plot_union(obj_sdrs, r_bin, 1, save_dir)
plot_union(obj_sdrs, r_bin, 5, save_dir)
plot_union(obj_sdrs, r_bin, 20, save_dir)

# random obj sdrs
obj_sdrs_random = torch.randn(20 * 5, 2048)
obj_sdrs_random = binarize(obj_sdrs_random, 41)

compare_plot(obj_sdrs, obj_sdrs_random, r_bin, 20, save_dir)
