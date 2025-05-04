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
Makes Bubble plots of FLOPs vs. accuracy and FLOPs vs. rotation error for Monty and ViT.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def simplify_name(name):
    if "dist_agent" in name:
        if "evidence" in name:
            return "30p_evidence"
        return name.split("_")[-1]  # Gets the Xp part
    else:
        return (
            name.split("-")[:2][0] + "-" + name.split("-")[:2][1]
        )  # Gets vit-bXX part


def create_comparison_plot(comparison_df, metric_col, metric_name, output_filename):
    """Create a bubble plot comparing FLOPs vs the specified metric.

    Args:
        comparison_df: DataFrame containing the data
        metric_col: Column name for the metric to plot on y-axis
        metric_name: Display name for the metric (used in labels)
        output_filename: Name of the output file
    """
    plt.figure(figsize=(8, 6))

    # Adjust size scaling using log10
    relative_sizes = comparison_df["flops"] / comparison_df["flops"].min() * 50

    # Create scatter plot for Monty (hyp), Monty (nohyp) and ViT separately
    nohyp_mask = comparison_df["experiment"].str.contains("nohyp", case=False)
    vit_mask = comparison_df["experiment"].str.contains("vit", case=False)
    hyp_mask = ~(nohyp_mask | vit_mask)  # Everything else is regular Monty (hyp)

    # Create legend elements list dynamically based on what's in the data
    legend_elements = []

    if hyp_mask.any():
        plt.scatter(
            comparison_df[hyp_mask]["flops"],
            comparison_df[hyp_mask][metric_col],
            s=relative_sizes[hyp_mask],
            alpha=0.8,
            c="#2F2B5C",
            label="Monty (hyp)",
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#2F2B5C",
                label="Monty (hyp)",
                markersize=15,
                alpha=0.8,
            )
        )

    if nohyp_mask.any():
        plt.scatter(
            comparison_df[nohyp_mask]["flops"],
            comparison_df[nohyp_mask][metric_col],
            s=relative_sizes[nohyp_mask],
            alpha=0.8,
            c="#5C2F2B",
            label="Monty (nohyp)",
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#5C2F2B",
                label="Monty (nohyp)",
                markersize=15,
                alpha=0.8,
            )
        )

    if vit_mask.any():
        plt.scatter(
            comparison_df[vit_mask]["flops"],
            comparison_df[vit_mask][metric_col],
            s=relative_sizes[vit_mask],
            alpha=0.8,
            c="#2B5C2F",
            label="ViT",
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#2B5C2F",
                label="ViT",
                markersize=15,
                alpha=0.8,
            )
        )

    # Add labels for each point
    for idx, row in comparison_df.iterrows():
        y_offset = -np.sqrt(relative_sizes.iloc[idx]) * 1.0
        plt.annotate(
            row["simple_name"],
            (row["flops"], row[metric_col]),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=11,
        )

    plt.title(f"Inference FLOPs vs. {metric_name}")
    plt.xlabel("FLOPs")
    plt.ylabel(metric_name)
    if metric_col == "accuracy_mean":
        plt.ylim(0, 100)
    plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.3)

    # Create custom legend
    plt.legend(handles=legend_elements, fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


def create_accuracy_vs_rotation_plot(comparison_df, output_filename):
    """Create a bubble plot comparing accuracy vs rotation error, with bubble size representing FLOPs.

    Args:
        comparison_df: DataFrame containing the data
        output_filename: Name of the output file
    """
    plt.figure(figsize=(8, 6))

    # Adjust size scaling using log10
    relative_sizes = comparison_df["flops"] / comparison_df["flops"].min() * 50

    # Create scatter plot for Monty (hyp), Monty (nohyp) and ViT separately
    nohyp_mask = comparison_df["experiment"].str.contains("nohyp", case=False)
    vit_mask = comparison_df["experiment"].str.contains("vit", case=False)
    hyp_mask = ~(nohyp_mask | vit_mask)

    legend_elements = []

    if hyp_mask.any():
        plt.scatter(
            comparison_df[hyp_mask]["rotation_error_mean"],
            comparison_df[hyp_mask]["accuracy_mean"],
            s=relative_sizes[hyp_mask],
            alpha=0.8,
            c="#2F2B5C",
            label="Monty (hyp)",
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#2F2B5C",
                label="Monty (hyp)",
                markersize=15,
                alpha=0.8,
            )
        )

    if nohyp_mask.any():
        plt.scatter(
            comparison_df[nohyp_mask]["rotation_error_mean"],
            comparison_df[nohyp_mask]["accuracy_mean"],
            s=relative_sizes[nohyp_mask],
            alpha=0.8,
            c="#5C2F2B",
            label="Monty (nohyp)",
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#5C2F2B",
                label="Monty (nohyp)",
                markersize=15,
                alpha=0.8,
            )
        )

    if vit_mask.any():
        plt.scatter(
            comparison_df[vit_mask]["rotation_error_mean"],
            comparison_df[vit_mask]["accuracy_mean"],
            s=relative_sizes[vit_mask],
            alpha=0.8,
            c="#2B5C2F",
            label="ViT",
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#2B5C2F",
                label="ViT",
                markersize=15,
                alpha=0.8,
            )
        )

    # Add labels for each point
    for idx, row in comparison_df.iterrows():
        y_offset = -np.sqrt(relative_sizes.iloc[idx]) * 1.0
        plt.annotate(
            row["simple_name"],
            (row["rotation_error_mean"], row["accuracy_mean"]),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=11,
        )

    plt.title("Accuracy vs. Rotation Error (bubble size = FLOPs)")
    plt.xlabel("Rotation Error")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.legend(handles=legend_elements, fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


def create_grouped_bar_plot(comparison_df, output_filename):
    """Create grouped bar plots comparing FLOPs, Accuracy, and Rotation Error for each model.

    Args:
        comparison_df: DataFrame containing the data
        output_filename: Name of the output file
    """
    plt.figure(figsize=(12, 6))

    # Normalize the metrics to [0,1] scale for better visualization
    normalized_df = comparison_df.copy()
    normalized_df["flops_norm"] = (
        np.log10(comparison_df["flops"]) - np.log10(comparison_df["flops"]).min()
    ) / (
        np.log10(comparison_df["flops"]).max() - np.log10(comparison_df["flops"]).min()
    )
    normalized_df["accuracy_norm"] = (
        comparison_df["accuracy_mean"] / 100
    )  # Already in percentage
    normalized_df["rotation_norm"] = (
        comparison_df["rotation_error_mean"]
        - comparison_df["rotation_error_mean"].min()
    ) / (
        comparison_df["rotation_error_mean"].max()
        - comparison_df["rotation_error_mean"].min()
    )

    # Set up the bar positions
    models = normalized_df["simple_name"]
    x = np.arange(len(models))
    width = 0.25

    # Create bars
    plt.bar(
        x - width,
        normalized_df["flops_norm"],
        width,
        label="FLOPs (log-normalized)",
        color="#2F2B5C",
        alpha=0.8,
    )
    plt.bar(
        x,
        normalized_df["accuracy_norm"],
        width,
        label="Accuracy (normalized)",
        color="#5C2F2B",
        alpha=0.8,
    )
    plt.bar(
        x + width,
        normalized_df["rotation_norm"],
        width,
        label="Rotation Error (normalized)",
        color="#2B5C2F",
        alpha=0.8,
    )

    # Customize the plot
    plt.xlabel("Model")
    plt.ylabel("Normalized Value")
    plt.title("Comparison of FLOPs, Accuracy, and Rotation Error")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    # Add value annotations on top of each bar
    for i in x:
        plt.text(
            i - width,
            normalized_df["flops_norm"].iloc[i],
            f"{comparison_df['flops'].iloc[i]:.1e}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )
        plt.text(
            i,
            normalized_df["accuracy_norm"].iloc[i],
            f"{comparison_df['accuracy_mean'].iloc[i]:.1f}%",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )
        plt.text(
            i + width,
            normalized_df["rotation_norm"].iloc[i],
            f"{comparison_df['rotation_error_mean'].iloc[i]:.2f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


def create_flops_bar_plot(comparison_df, output_filename, filter_dict=None):
    """Create horizontal bar plot of FLOPs for models.

    Args:
        comparison_df: DataFrame containing the data
        output_filename: Name of the output file
        filter_dict: Optional dictionary of form {'pattern': 'regex'} to filter experiments
                    e.g., {'pattern': '20p|vit-b16'} for 20p and vit-b16 only
    """
    # Filter data if requested
    plot_df = comparison_df.copy()
    if filter_dict is not None:
        pattern = filter_dict.get("pattern")
        if pattern:
            plot_df = plot_df[plot_df["experiment"].str.contains(pattern, regex=True)]

    # Sort by FLOPs
    plot_df = plot_df.sort_values("flops", ascending=True)

    plt.figure(figsize=(10, max(6, len(plot_df) * 0.4)))

    # Create color mapping
    colors = []
    for exp in plot_df["experiment"]:
        if "nohyp" in exp.lower():
            colors.append("#5C2F2B")  # Monty (nohyp) color
        elif "vit" in exp.lower():
            colors.append("#2B5C2F")  # ViT color
        else:
            colors.append("#2F2B5C")  # Monty (hyp) color

    # Create horizontal bars
    bars = plt.barh(range(len(plot_df)), plot_df["flops"], alpha=0.8, color=colors)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(
            width * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2e}",
            va="center",
            fontsize=10,
        )

    # Customize the plot
    plt.ylabel("Model")
    plt.xlabel("FLOPs")
    plt.title("Model FLOPs Comparison")
    plt.yticks(range(len(plot_df)), plot_df["simple_name"])
    plt.xscale("log")
    plt.grid(True, axis="x", linestyle="--", alpha=0.3)

    # Add legend
    legend_elements = []
    if any("nohyp" in exp.lower() for exp in plot_df["experiment"]):
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, fc="#5C2F2B", alpha=0.8, label="Monty (nohyp)")
        )
    if any("vit" in exp.lower() for exp in plot_df["experiment"]):
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, fc="#2B5C2F", alpha=0.8, label="ViT")
        )
    if any(
        ("vit" not in exp.lower() and "nohyp" not in exp.lower())
        for exp in plot_df["experiment"]
    ):
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, fc="#2F2B5C", alpha=0.8, label="Monty (hyp)")
        )

    if legend_elements:
        plt.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    csv_dir = Path("~/tbp/results/dmc/results").expanduser()
    save_dir = Path("~/tbp/results/dmc/results/floppy").expanduser()

    # Load all data
    monty_df = pd.read_csv(
        csv_dir / "floppy" / "flops_aws" / "floppy_flops_accuracy_rotation_error.csv"
    )
    vit_df = pd.read_csv(csv_dir / "vit" / "vit_flops_accuracy_rotation_error.csv")

    # Prepare data once
    monty_df.rename(columns={"total_train_flops_per_epoch": "flops"}, inplace=True)
    vit_df.rename(columns={"total_train_flops_per_epoch": "flops"}, inplace=True)
    monty_df["flops"] = monty_df["flops"] / 77 / 5
    vit_df["flops"] = vit_df["flops"] / 3 / 77 / 14 / 0.8

    # Convert accuracy to percentage
    monty_df["accuracy_mean"] *= 100
    vit_df["accuracy_mean"] *= 100

    # Convert rotation error from radians to degrees for monty_df
    monty_df["rotation_error_mean"] = monty_df["rotation_error_mean"] * (180 / np.pi)

    # Separate monty_df into nohyp and hyp
    monty_df["is_nohyp"] = monty_df["experiment"].str.contains("nohyp")
    nohyp_df = monty_df[monty_df["is_nohyp"] == True].copy()
    hyp_df = monty_df[monty_df["is_nohyp"] == False].copy()

    # Add simple names
    nohyp_df["simple_name"] = nohyp_df["experiment"].apply(simplify_name)
    hyp_df["simple_name"] = hyp_df["experiment"].apply(simplify_name)
    vit_df["simple_name"] = vit_df["experiment"].apply(simplify_name)

    # Create individual comparison dataframes
    nohyp_comparison_df = pd.concat([nohyp_df, vit_df.copy()])
    hyp_comparison_df = pd.concat([hyp_df, vit_df.copy()])
    combined_comparison_df = pd.concat([hyp_df, nohyp_df, vit_df.copy()])

    # Create individual comparison plots (hyp vs. vit and nohyp vs. vit)
    create_comparison_plot(
        nohyp_comparison_df,
        "accuracy_mean",
        "Accuracy (%)",
        save_dir / "fig7_flops_nohyp_accuracy.png",
    )
    create_comparison_plot(
        nohyp_comparison_df,
        "rotation_error_mean",
        "Rotation Error",
        save_dir / "fig7_flops_nohyp_rotation.png",
    )
    create_comparison_plot(
        hyp_comparison_df,
        "accuracy_mean",
        "Accuracy (%)",
        save_dir / "fig7_flops_hyp_accuracy.png",
    )
    create_comparison_plot(
        hyp_comparison_df,
        "rotation_error_mean",
        "Rotation Error",
        save_dir / "fig7_flops_hyp_rotation.png",
    )

    # Create new combined plots (hyp + nohyp + vit)
    create_comparison_plot(
        combined_comparison_df,
        "accuracy_mean",
        "Accuracy (%)",
        save_dir / "fig7_flops_combined_accuracy.png",
    )
    create_comparison_plot(
        combined_comparison_df,
        "rotation_error_mean",
        "Rotation Error",
        save_dir / "fig7_flops_combined_rotation.png",
    )

    # Add new plot for accuracy vs rotation error
    create_accuracy_vs_rotation_plot(
        combined_comparison_df,
        save_dir / "fig7_accuracy_vs_rotation.png",
    )

    # Add new grouped bar plot
    create_grouped_bar_plot(
        combined_comparison_df,
        save_dir / "fig7_grouped_comparison.png",
    )

    # Create FLOPs bar plots
    # Plot for all models
    create_flops_bar_plot(combined_comparison_df, save_dir / "fig7_flops_bars_all.png")

    # Plot for 20p hyp vs vit-b16
    create_flops_bar_plot(
        combined_comparison_df,
        save_dir / "fig7_flops_bars_filtered.png",
        filter_dict={
            "pattern": "20p(?!.*nohyp)|vit-b16"
        },  # matches 20p (not nohyp) and vit-b16
    )


if __name__ == "__main__":
    main()