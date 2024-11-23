"""Contains functions for generating summaries of multi-LM results."""

import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monty_capabilities_analysis.data_utils import (
    OUT_DIR,
    get_percent_correct,
    load_eval_stats,
)
from monty_capabilities_analysis.plot_utils import (
    TBP_COLORS,
    violinplot,
)

plt.rcParams["font.size"] = 8


# Directories to save plots and tables to.
OUT_DIR = OUT_DIR / "multi_lm_summaries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Additional output directories depending on format.
PNG_DIR = OUT_DIR / "png"
PNG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR = OUT_DIR / "svg"
SVG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR = OUT_DIR / "pdf"
PDF_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR = OUT_DIR / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)
TXT_DIR = OUT_DIR / "txt"
TXT_DIR.mkdir(parents=True, exist_ok=True)


def get_summary_stats(
    dataframes: List[pd.DataFrame], conditions: List[str]
) -> pd.DataFrame:
    """Get a dataframe with basic stats.

    Args:
        dataframes (List[pd.DataFrame]): Dataframes for different conditions.
            Typically base, noise, RR, and noise + RR.
        conditions (List[str]): Conditions for each dataframe. Conditions will be the
            index of the returned dataframe.

    Returns:
        pd.DataFrame: stats
    """
    table = pd.DataFrame(index=conditions)
    table.index.name = "Num. LMs"

    num_steps_all = [df.num_steps for df in dataframes]
    num_steps_median = [np.median(arr) for arr in num_steps_all]
    table["Med. Steps"] = num_steps_median

    accuracy_all = [get_percent_correct(df) for df in dataframes]
    accuracy_mean = [np.mean(arr) for arr in accuracy_all]
    table["Mean Accuracy"] = accuracy_mean

    rotation_errors_all = [np.rad2deg(df.rotation_error.dropna()) for df in dataframes]
    rotation_errors_median = [np.median(arr) for arr in rotation_errors_all]
    table["Med. Rotation Error (deg)"] = rotation_errors_median

    return table


def write_latex_table(
    path: os.PathLike,
    dataframes: List[pd.DataFrame],
    conditions: List[str],
    caption: str,
    label: str,
) -> None:
    """Write a latex table with summary stats.

    Args:
        dataframes (List[pd.DataFrame]): Dataframes for different conditions.
            Typically base, noise, RR, and noise + RR.
        conditions (List[str]): Conditions for each dataframe. Conditions will be the
            index of the returned dataframe.

    Returns:
        pd.DataFrame: stats
    """
    table = pd.DataFrame(index=conditions)
    table.index.name = "Num. LMs"

    num_steps_all = [df.num_steps for df in dataframes]
    num_steps_median = [np.median(arr) for arr in num_steps_all]
    table["Med. Steps"] = num_steps_median

    accuracy_all = [get_percent_correct(df) for df in dataframes]
    accuracy_mean = [np.mean(arr) for arr in accuracy_all]
    table["Mean Accuracy"] = accuracy_mean

    rotation_errors_all = [np.rad2deg(df.rotation_error.dropna()) for df in dataframes]
    rotation_errors_median = [np.median(arr) for arr in rotation_errors_all]
    table["Med. Rotation Error (deg)"] = rotation_errors_median

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{llll}")
    lines.append("\\toprule")

    # Header
    line_items = [table.index.name] + list(table.columns)
    line_items_tex = [f"\\textbf{{{name}}}" for name in line_items]
    line = " & ".join(line_items_tex) + " \\\\"
    lines.append(line)
    lines.append("\\midrule")

    # Rows
    for row_num in range(len(table)):
        row_name = table.index[row_num]
        row_items = [row_name] + [f"{val:.2f}" for val in table.iloc[row_num]]
        line = " & ".join(row_items) + " \\\\"
        lines.append(line)

    # Footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def init_multi_lm_plot(
    dataframes: List[pd.DataFrame],
    conditions: List[str],
    figsize=(3.5, 2),
) -> matplotlib.figure.Figure:
    """Initialize a plot with violin plots for steps and accuracy.

    Used by other functions to generate plots for specific datasets.

    Args:
        dataframes (List[pd.DataFrame]): Dataframes for different conditions.
            Typically ["1", "2", ...] for the number of LMs.
        conditions (List[str]): Conditions/labels associated with each dataframe.
        figsize (tuple, optional): Figure size. Defaults to (3.5, 2).

    Returns:
        matplotlib.figure.Figure: _description_
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    xticks = list(range(len(conditions)))

    # Plot distribution of num_steps
    ax = axes[0]
    num_steps = [df.num_steps for df in dataframes]
    violinplot(ax, num_steps, conditions)
    ax.set_xlabel("LMs")
    ax.set_ylabel("Steps")
    ax.set_ylim(0, 500)
    ax.set_title("Steps")

    # Plot accuracy
    ax = axes[1]
    xticks = list(range(len(conditions)))
    ax.bar(
        xticks,
        [get_percent_correct(df) for df in dataframes],
        color=TBP_COLORS["blue"],
    )
    ax.set_xlabel("LMs")
    ax.set_xticks(xticks, conditions)
    ax.set_ylabel("% Correct")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy")

    return fig


def plot_multi_lm_base(save: bool = False):
    dataframes = [
        load_eval_stats("dist_agent_1lm"),
        load_eval_stats("dist_agent_2lm"),
        load_eval_stats("dist_agent_5lm"),
        load_eval_stats("dist_agent_9lm"),
        load_eval_stats("dist_agent_10lm"),
    ]
    conditions = ["1", "2", "5", "9", "10"]
    fig = init_multi_lm_plot(dataframes, conditions)
    fig.suptitle("Multiple LMs")
    fig.tight_layout()
    if save:
        fig.savefig(PNG_DIR / "multi_lm_base.png")
        fig.savefig(SVG_DIR / "multi_lm_base.svg")
        fig.savefig(PDF_DIR / "multi_lm_base.pdf")
        stats = get_summary_stats(dataframes, conditions)
        stats.to_csv(CSV_DIR / "multi_lm_base.csv", float_format="%.2f")
        write_latex_table(
            TXT_DIR / "multi_lm_base.txt",
            dataframes,
            conditions,
            "Multiple LMs",
            "tab:multi-lm-base",
        )
    return fig


def plot_multi_lm_noise(save: bool = False):
    dataframes = [
        load_eval_stats("dist_agent_1lm_noise"),
        load_eval_stats("dist_agent_2lm_noise"),
        load_eval_stats("dist_agent_5lm_noise"),
        load_eval_stats("dist_agent_9lm_noise"),
        load_eval_stats("dist_agent_10lm_noise"),
    ]
    conditions = ["1", "2", "5", "9", "10"]
    fig = init_multi_lm_plot(dataframes, conditions)
    fig.suptitle("Multiple LMs with Noise")
    fig.tight_layout()
    if save:
        fig.savefig(PNG_DIR / "multi_lm_noise.png")
        fig.savefig(SVG_DIR / "multi_lm_noise.svg")
        fig.savefig(PDF_DIR / "multi_lm_noise.pdf")
        stats = get_summary_stats(dataframes, conditions)
        stats.to_csv(CSV_DIR / "multi_lm_noise.csv", float_format="%.2f")
        write_latex_table(
            TXT_DIR / "multi_lm_noise.txt",
            dataframes,
            conditions,
            "Multiple LMs with Noise",
            "tab:multi-lm-noise",
        )
    return fig


def plot_multi_lm_randrot(save: bool = False):
    dataframes = [
        load_eval_stats("dist_agent_1lm_randrot"),
        load_eval_stats("dist_agent_2lm_randrot"),
        load_eval_stats("dist_agent_5lm_randrot"),
        load_eval_stats("dist_agent_9lm_randrot"),
        load_eval_stats("dist_agent_10lm_randrot"),
    ]
    conditions = ["1", "2", "5", "9", "10"]
    fig = init_multi_lm_plot(dataframes, conditions)
    fig.suptitle("Multiple LMs with Random Rotations")
    fig.tight_layout()
    if save:
        fig.savefig(PNG_DIR / "multi_lm_randrot.png")
        fig.savefig(SVG_DIR / "multi_lm_randrot.svg")
        fig.savefig(PDF_DIR / "multi_lm_randrot.pdf")
        stats = get_summary_stats(dataframes, conditions)
        stats.to_csv(CSV_DIR / "multi_lm_randrot.csv", float_format="%.2f")
        write_latex_table(
            TXT_DIR / "multi_lm_randrot.txt",
            dataframes,
            conditions,
            "Multiple LMs with Random Rotations",
            "tab:multi-lm-randrot",
        )
    return fig


def plot_multi_lm_randrot_noise(save: bool = False):
    dataframes = [
        load_eval_stats("dist_agent_1lm_randrot_noise"),
        load_eval_stats("dist_agent_2lm_randrot_noise"),
        load_eval_stats("dist_agent_5lm_randrot_noise"),
        load_eval_stats("dist_agent_9lm_randrot_noise"),
        load_eval_stats("dist_agent_10lm_randrot_noise"),
    ]
    conditions = ["1", "2", "5", "9", "10"]
    fig = init_multi_lm_plot(dataframes, conditions)
    fig.suptitle("Multiple LMs with Random Rotations + Noise")
    fig.tight_layout()
    if save:
        fig.savefig(PNG_DIR / "multi_lm_randrot_noise.png")
        fig.savefig(SVG_DIR / "multi_lm_randrot_noise.svg")
        fig.savefig(PDF_DIR / "multi_lm_randrot_noise.pdf")
        stats = get_summary_stats(dataframes, conditions)
        stats.to_csv(CSV_DIR / "multi_lm_randrot_noise.csv", float_format="%.2f")
        write_latex_table(
            TXT_DIR / "multi_lm_randrot_noise.txt",
            dataframes,
            conditions,
            "Multiple LMs with Random Rotations + Noise",
            "tab:multi-lm-randrot-noise",
        )
    return fig


def plot_multi_lm_num_steps_all_conditions(save: bool = False):
    base_model_names = [
        "dist_agent_1lm",
        "dist_agent_2lm",
        "dist_agent_5lm",
        "dist_agent_9lm",
        "dist_agent_10lm",
    ]
    conditions = ["1", "2", "5", "9", "10"]

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.5))

    ax = axes[0, 0]
    model_names = base_model_names
    dataframes = [load_eval_stats(name) for name in model_names]

    num_steps = [df.num_steps for df in dataframes]
    violinplot(ax, num_steps, conditions)
    ax.set_title("Base")

    ax = axes[0, 1]
    model_names = [name + "_noise" for name in base_model_names]
    dataframes = [load_eval_stats(name) for name in model_names]
    num_steps = [df.num_steps for df in dataframes]
    violinplot(ax, num_steps, conditions)
    ax.set_title("Noise")

    ax = axes[1, 0]
    model_names = [name + "_randrot" for name in base_model_names]
    dataframes = [load_eval_stats(name) for name in model_names]
    num_steps = [df.num_steps for df in dataframes]
    violinplot(ax, num_steps, conditions)
    ax.set_title("Random Rotations")

    ax = axes[1, 1]
    model_names = [name + "_randrot_noise" for name in base_model_names]
    dataframes = [load_eval_stats(name) for name in model_names]
    num_steps = [df.num_steps for df in dataframes]
    violinplot(ax, num_steps, conditions)
    ax.set_title("Random Rotations + Noise")

    for ax in axes.flatten():
        ax.set_ylabel("Num Steps")
        ax.set_ylim(0, 100)

    fig.suptitle("Multiple LMs")
    fig.tight_layout()
    if save:
        fig.savefig(PNG_DIR / "multi_lm_num_steps.png")
        fig.savefig(SVG_DIR / "multi_lm_num_steps.svg")
        fig.savefig(PDF_DIR / "multi_lm_num_steps.pdf")
    return fig


if __name__ == "__main__":
    plot_multi_lm_base(save=True)
    plot_multi_lm_noise(save=True)
    plot_multi_lm_randrot(save=True)
    plot_multi_lm_randrot_noise(save=True)
    plot_multi_lm_num_steps_all_conditions(save=True)
