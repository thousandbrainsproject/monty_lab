"""Contains functions for generating summaries of single-LM results."""

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
OUT_DIR = OUT_DIR / "one_lm_summaries"
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
    table.index.name = "Condition"

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
    table.index.name = "Condition"

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


def init_1lm_plot(
    dataframes: List[pd.DataFrame],
    conditions: List[str],
    figsize=(6, 3),
) -> matplotlib.figure.Figure:
    """Initialize a plot with violin plots for steps, accuracy, and rotation error.

    Used by other functions to generate plots for specific datasets.

    Args:
        dataframes (List[pd.DataFrame]): Dataframes for different conditions.
            Typically base, noise, RR, and noise + RR.
        conditions (List[str]): Conditions/labels associated with each dataframe.
        figsize (tuple, optional): Figure size. Defaults to (4, 2).

    Returns:
        matplotlib.figure.Figure: _description_
    """

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    xticks = list(range(len(conditions)))
    # Plot distribution of num_steps
    ax = axes[0]
    num_steps = [df.num_steps for df in dataframes]
    violinplot(ax, num_steps, conditions, rotation=45)
    ax.set_ylabel("Steps")
    ax.set_ylim(0, 500)
    ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.set_title("Steps")

    # Plot object detection accuracy
    ax = axes[1]
    ax.bar(
        xticks,
        [get_percent_correct(df) for df in dataframes],
        color=TBP_COLORS["blue"],
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("% Correct")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy")

    # Plot rotation error
    ax = axes[2]
    rotation_errors = [np.rad2deg(df.rotation_error.dropna()) for df in dataframes]
    violinplot(ax, rotation_errors, conditions, rotation=45)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_ylim(0, 180)
    ax.set_ylabel("Error (degrees)")
    ax.set_title("Rotation Error")

    return fig


# def plot_multimodal_transfer_base(save: bool = False):
#     dataframes = [
#         load_eval_stats("dist_agent_1lm"),
#         load_eval_stats("dist_on_touch"),
#         load_eval_stats("touch_agent_1lm"),
#         load_eval_stats("touch_on_dist"),
#     ]
#     conditions = ["dist on dist", "dist on touch", "touch on touch", "touch on dist"]
#     fig = init_1lm_plot(dataframes, conditions, figsize=(4, 2.15))
#     fig.suptitle("Multimodal Transfer")
#     fig.tight_layout()
#     if save:
#         fig.savefig(PNG_DIR / "multimodal_transfer_base.png", dpi=300)
#         fig.savefig(SVG_DIR / "multimodal_transfer_base.svg")
#         fig.savefig(PDF_DIR / "multimodal_transfer_base.pdf")
#         stats = get_summary_stats(dataframes, conditions)
#         stats.to_csv(CSV_DIR / "multimodal_transfer_base.csv", float_format="%.2f")
#         write_latex_table(
#             TXT_DIR / "multimodal_transfer_base.txt",
#             dataframes,
#             conditions,
#             "Multimodal Transfer Performance",
#             "tab:multimodal-transfer-performance",
#         )
#     return fig


def plot_fig3():
    dataframes = [
        load_eval_stats("dist_agent_1lm"),
        load_eval_stats("dist_agent_1lm_noise"),
        load_eval_stats("dist_agent_1lm_randrot_all"),
        load_eval_stats("dist_agent_1lm_randrot_all_noise"),
    ]
    conditions = ["base", "noise", "RR", "noise + RR"]
    fig = init_1lm_plot(dataframes, conditions)

    fig.suptitle("Fig 3: Robust Sensorimotor Inference")
    fig.tight_layout()
    return fig


def plot_fig4():
    dataframes = [
        load_eval_stats("dist_agent_1lm_randrot_noise"),
        load_eval_stats("dist_agent_2lm_randrot_noise"),
        load_eval_stats("dist_agent_4lm_randrot_noise"),
        load_eval_stats("dist_agent_8lm_randrot_noise"),
        load_eval_stats("dist_agent_16lm_randrot_noise"),
    ]
    conditions = ["1", "2", "4", "8", "16"]
    fig = init_1lm_plot(dataframes, conditions, figsize=(7, 3))

    fig.suptitle("Fig 4: Voting")
    fig.tight_layout()
    return fig


def plot_fig5():
    dataframes = [
        load_eval_stats("dist_agent_1lm_randrot_noise"),
        load_eval_stats("dist_agent_1lm_randrot_noise_nohyp"),
        load_eval_stats("dist_agent_1lm_randrot_noise_moderatehyp"),
    ]
    conditions = ["std", "no hyp", "mod hyp"]
    fig = init_1lm_plot(dataframes, conditions)
    fig.suptitle("Fig 5: Model-Based Policies")
    fig.tight_layout()
    return fig


def plot_fig6():
    dataframes = [
        load_eval_stats("dist_agent_1lm_randrot_nohyp_1rot_trained"),
        load_eval_stats("dist_agent_1lm_randrot_nohyp_2rot_trained"),
        load_eval_stats("dist_agent_1lm_randrot_nohyp_4rot_trained"),
        load_eval_stats("dist_agent_1lm_randrot_nohyp_8rot_trained"),
        load_eval_stats("dist_agent_1lm_randrot_nohyp_16rot_trained"),
        load_eval_stats("dist_agent_1lm_randrot_nohyp_32rot_trained"),
    ]
    conditions = ["1", "2", "4", "8", "16", "32"]
    fig = init_1lm_plot(dataframes, conditions, figsize=(7, 3))
    fig.suptitle("Fig 6: Rapid Learning")
    fig.tight_layout()
    return fig


def plot_fig7():
    dataframes = [
        load_eval_stats("dist_agent_1lm_randrot_nohyp_x_percent_5p"),
        load_eval_stats("dist_agent_1lm_randrot_nohyp_x_percent_10p"),
        load_eval_stats("dist_agent_1lm_randrot_nohyp_x_percent_20p"),
        load_eval_stats("dist_agent_1lm_randrot_nohyp_x_percent_30p"),
        # load_eval_stats("dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all"),
    ]
    conditions = ["5%", "10%", "20%", "30%"]
    fig = init_1lm_plot(dataframes, conditions, figsize=(7, 3))
    fig.suptitle("Fig 7: Flops Comparison")
    fig.axes[0].set_xlabel("x_percent_threshold")
    fig.tight_layout()


def plot_fig8():
    dataframes = [
        load_eval_stats("dist_agent_1lm_randrot_noise"),
        load_eval_stats("touch_agent_1lm_randrot_noise"),
        load_eval_stats("dist_on_touch_1lm_randrot_noise"),
        load_eval_stats("touch_on_dist_1lm_randrot_noise"),
    ]
    conditions = ["dist", "touch", "dist on touch", "touch on dist"]
    fig = init_1lm_plot(dataframes, conditions)

    fig.suptitle("Fig 8: Multimodal Transfer")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    pass
    # plot_1lm_distant_agent(save=True)
    # plot_1lm_distant_agent_nohyp(save=True)
    # plot_1lm_surface_agent(save=True)
    # plot_1lm_touch_agent(save=True)
    # plot_dist_on_touch(save=True)
    # plot_touch_on_dist(save=True)
    # plot_multimodal_transfer_base(save=True)

    # if save:
    #     fig.savefig(PNG_DIR / "1lm_surface_agent.png", dpi=300)
    #     fig.savefig(SVG_DIR / "1lm_surface_agent.svg")
    #     fig.savefig(PDF_DIR / "1lm_surface_agent.pdf")
    #     stats = get_summary_stats(dataframes, conditions)
    #     stats.to_csv(CSV_DIR / "1lm_surface_agent.csv", float_format="%.2f")
    #     write_latex_table(
    #         TXT_DIR / "1lm_surface_agent.txt",
    #         dataframes,
    #         conditions,
    #         "Surface Agent Performance",
    #         "tab:surface-agent-performance",
    #     )
    fig3 = plot_fig3()
    fig4 = plot_fig4()
    fig5 = plot_fig5()
    fig6 = plot_fig6()
    fig7 = plot_fig7()
    fig8 = plot_fig8()
