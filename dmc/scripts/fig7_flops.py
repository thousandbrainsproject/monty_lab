"""
This script reads the flops_accuracy_rotation_error.csv file and plots the flops vs accuracy and flops vs rotation error.

It colors by Monty or ViT.

It saves the bubble plot to fig7_flops.png
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    csv_dir = Path("~/tbp/results/dmc/results").expanduser()
    monty_nohyp_results_df = pd.read_csv(
        csv_dir / "floppy" / "nohyp" / "flops_accuracy_rotation_error.csv"
    )
    monty_hyp_results_df = pd.read_csv(
        csv_dir / "floppy" / "hyp" / "flops_accuracy_rotation_error.csv"
    )
    vit_results_df = pd.read_csv(csv_dir / "vit" / "flops_accuracy_rotation_error.csv"12])
    # For inference, divide vit_results_df's flops column by 3 
    # Concatenate the two dataframes
    monty_results_df = pd.concat([monty_nohyp_results_df, monty_hyp_results_df])

    plt.figure(figsize=(10, 5))
    plt.scatter(df["flops"], df["accuracy_mean"], c=df["model"], cmap="viridis")
    plt.colorbar(label="Model")
    plt.xlabel("Flops")
    plt.ylabel("Accuracy")

    plt.savefig("fig7_flops.png")


if __name__ == "__main__":
    main()
