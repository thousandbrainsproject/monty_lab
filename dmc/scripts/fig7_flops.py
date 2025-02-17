import os
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

    # Create scatter plot for Monty and ViT separately
    monty_mask = comparison_df["experiment"].str.contains("dist_agent", case=False)
    vit_mask = comparison_df["experiment"].str.contains("vit", case=False)

    # Plot points
    plt.scatter(
        comparison_df[monty_mask]["flops"],
        comparison_df[monty_mask][metric_col],
        s=relative_sizes[monty_mask],
        alpha=0.8,
        c="#2F2B5C",
        label="Monty",
    )
    plt.scatter(
        comparison_df[vit_mask]["flops"],
        comparison_df[vit_mask][metric_col],
        s=relative_sizes[vit_mask],
        alpha=0.8,
        c="#2B5C2F",
        label="ViT",
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
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2F2B5C",
            label="Monty",
            markersize=15,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2B5C2F",
            label="ViT",
            markersize=15,
            alpha=0.8,
        ),
    ]
    plt.legend(handles=legend_elements, fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

def prepare_dataframe(monty_df, vit_df):
    """Prepare and combine the Monty and ViT dataframes."""
    # Make the column names consistent
    vit_df.rename(columns={"total_train_flops_per_epoch": "flops"}, inplace=True)
    vit_df["flops"] = vit_df["flops"] / 3 / 77 / 14

    # Convert accuracy to percentage
    monty_df["accuracy_mean"] *= 100
    vit_df["accuracy_mean"] *= 100

    # Concatenate the dataframes
    comparison_df = pd.concat([monty_df, vit_df])
    comparison_df["simple_name"] = comparison_df["experiment"].apply(simplify_name)

    return comparison_df

def main():
    csv_dir = Path("~/tbp/results/dmc/results").expanduser()

    # Load all data
    monty_nohyp_df = pd.read_csv(
        csv_dir / "floppy" / "nohyp" / "flops_accuracy_rotation_error.csv"
    )
    monty_hyp_df = pd.read_csv(
        csv_dir / "floppy" / "hyp" / "flops_accuracy_rotation_error.csv"
    )
    vit_df = pd.read_csv(csv_dir / "vit" / "vit_flops_accuracy_rotation_error.csv")

    # Prepare dataframes
    nohyp_comparison_df = prepare_dataframe(monty_nohyp_df, vit_df.copy())
    hyp_comparison_df = prepare_dataframe(monty_hyp_df, vit_df.copy())

    # Create all plots
    create_comparison_plot(
        nohyp_comparison_df,
        "accuracy_mean",
        "Accuracy (%)",
        "fig7_flops_nohyp_accuracy.png",
    )
    create_comparison_plot(
        nohyp_comparison_df,
        "rotation_error_mean",
        "Rotation Error",
        "fig7_flops_nohyp_rotation.png",
    )
    create_comparison_plot(
        hyp_comparison_df,
        "accuracy_mean",
        "Accuracy (%)",
        "fig7_flops_hyp_accuracy.png",
    )
    create_comparison_plot(
        hyp_comparison_df,
        "rotation_error_mean",
        "Rotation Error",
        "fig7_flops_hyp_rotation.png",
    )

if __name__ == "__main__":
    main()