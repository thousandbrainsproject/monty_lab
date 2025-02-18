"""
This script reads the raw results from FLOPs measurements and accuracy and rotation errors and combines them together for easy visualization.

FLOPs results are stored in `~/tbp/results/dmc/results/floppy/flops_aws/*.csv`
Rotation error results are stored in `~/tbp/results/dmc/results/floppy/results/predictions.csv`
"""

import os
from pathlib import Path
from typing import Tuple

import pandas as pd


def compute_accuracy(predictions_df: pd.DataFrame) -> Tuple[float, float]:
    """
    If primary_performance is correct or correct_mlh, then it is considered correct.
    """
    correct = predictions_df[
        predictions_df["primary_performance"].isin(["correct", "correct_mlh"])
    ]
    accuracy = correct.shape[0] / predictions_df.shape[0]
    std = 0
    return accuracy, std


def compute_rotation_error(predictions_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute mean and std of rotation error from predictions dataframe
    """
    mean_rotation_error = predictions_df["rotation_error"].mean()
    std_rotation_error = predictions_df["rotation_error"].std()
    return mean_rotation_error, std_rotation_error


def compute_flops(flops_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute mean and std of flops from flops dataframe from perf stat
    """
    flops_per_operation = {
        "fp_arith_inst_retired.128b_packed_double": 2,
        "fp_arith_inst_retired.128b_packed_single": 4,
        "fp_arith_inst_retired.256b_packed_double": 4,
        "fp_arith_inst_retired.256b_packed_single": 8,
        "fp_arith_inst_retired.512b_packed_double": 8,
        "fp_arith_inst_retired.512b_packed_single": 16,
        "fp_arith_inst_retired.scalar_double": 1,
        "fp_arith_inst_retired.scalar_single": 1,
    }
    # Compute weighted sum of flops, the count is in first column, the operation in 3rd column, and weights in the dict
    total_flops = 0
    for index, row in flops_df.iterrows():
        count = row[0]
        operation = row[2]
        if count > 0:
            if operation in flops_per_operation:
                total_flops += count * flops_per_operation[operation]
            else:
                print(f"Operation {operation} not found in flops_per_operation")
    return total_flops


def main():
    save_dir = "~/tbp/results/dmc/results/floppy/flops_aws"
    experiments = [
        "dist_agent_1lm_randrot_x_percent_5p",
        "dist_agent_1lm_randrot_x_percent_10p",
        "dist_agent_1lm_randrot_x_percent_20p",
        "dist_agent_1lm_randrot_x_percent_30p",
        "dist_agent_1lm_randrot_nohyp_x_percent_5p",
        "dist_agent_1lm_randrot_nohyp_x_percent_10p",
        # "dist_agent_1lm_randrot_nohyp_x_percent_20p",
        "dist_agent_1lm_randrot_nohyp_x_percent_30p",
    ]

    results = pd.DataFrame()
    for experiment in experiments:
        # if experiment doesn't exist, skip
        flops_path = (
            Path(f"~/tbp/results/dmc/results/floppy/flops_aws/{experiment}.csv")
            .expanduser()
            .resolve()
        )
        eval_path = (
            Path(f"~/tbp/results/dmc/results/{experiment}_perf/eval_stats.csv")
            .expanduser()
            .resolve()
        )
        if not flops_path.exists():
            print(f"{flops_path} does not exist")
            continue
        flops_df = pd.read_csv(flops_path, comment="#", header=None)
        eval_df = pd.read_csv(eval_path)
        result = {}
        result["experiment"] = experiment
        result["accuracy_mean"], result["accuracy_std"] = compute_accuracy(eval_df)
        result["rotation_error_mean"], result["rotation_error_std"] = (
            compute_rotation_error(eval_df)
        )
        result["total_train_flops_per_epoch"] = compute_flops(flops_df)
        # Concatenate results
        results = pd.concat([results, pd.DataFrame([result])])

    results.to_csv(f"{save_dir}/floppy_flops_accuracy_rotation_error.csv", index=False)


if __name__ == "__main__":
    main()
