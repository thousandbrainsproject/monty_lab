"""
This script reads the raw data for experiments related to Floppy on Monty.

Results are stored in `~/tbp/results/dmc/results`
"""

import os
from typing import List

import numpy as np
import pandas as pd


def read_flop_traces(df: pd.DataFrame) -> float:
    """
    Read the flop traces from a file.
    """
    # Get average of flops for experiment.run_episode in method column
    run_episode_df = df[df["method"] == "experiment.run_episode"]
    # Return as a list
    return run_episode_df["flops"].tolist()


def compute_accuracy(df: pd.DataFrame) -> float:
    """
    Compute the accuracy from the eval_stats.csv file.

    It is considered accurate if primary_performance is correct or correct_mlh
    """
    correct = df[df["primary_performance"].isin(["correct", "correct_mlh"])]
    return correct.shape[0] / df.shape[0]


def compute_quaternion_error(df: pd.DataFrame) -> float:
    """
    Compute the quaternion error from the eval_stats.csv file.
    average the rotation_error column
    """
    return df["rotation_error"].mean()


def main(experiments: List[str], save_dir: str):
    data_dir = "~/tbp/results/dmc/results"
    data_dir = os.path.expanduser(data_dir)
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize results DataFrame
    results = pd.DataFrame(
        columns=[
            "experiment",
            "accuracy_mean",
            "accuracy_std",
            "rotation_error_mean",
            "rotation_error_std",
            "flops_sum",
            "flops_std",
        ]
    )

    for experiment in experiments:
        # First collect all flops data
        experiment_flops = []
        # Read all csv files that start with "flop_traces" in results_dir/experiment
        files = os.listdir(os.path.join(data_dir, experiment))
        for file in files:
            if file.startswith("flop_traces"):
                flops_df = pd.read_csv(os.path.join(data_dir, experiment, file))
                flops = read_flop_traces(flops_df)
                experiment_flops.extend(flops)

        # Calculate flops statistics
        flops_sum = np.sum(experiment_flops) if experiment_flops else np.nan
        flops_std = np.std(experiment_flops) if experiment_flops else np.nan

        # Now collect all eval stats
        accuracies = []
        rotation_errors = []
        for file in files:
            if file.startswith("eval_stats.csv"):
                eval_df = pd.read_csv(os.path.join(data_dir, experiment, file))
                accuracies.append(compute_accuracy(eval_df))
                rotation_errors.append(compute_quaternion_error(eval_df))

        # Calculate statistics
        accuracy_mean = np.mean(accuracies) if accuracies else np.nan
        accuracy_std = np.std(accuracies) if accuracies else np.nan
        rotation_error_mean = np.mean(rotation_errors) if rotation_errors else np.nan
        rotation_error_std = np.std(rotation_errors) if rotation_errors else np.nan

        # Add to results
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        "experiment": [experiment],
                        "accuracy_mean": [accuracy_mean],
                        "accuracy_std": [accuracy_std],
                        "rotation_error_mean": [rotation_error_mean],
                        "rotation_error_std": [rotation_error_std],
                        "flops_sum": [flops_sum],
                        "flops_std": [flops_std],
                    }
                ),
            ]
        )

    results.to_csv(
        os.path.join(save_dir, "flops_accuracy_rotation_error.csv"), index=False
    )


if __name__ == "__main__":
    nohyp_experiments = [
        "dist_agent_1lm_randrot_nohyp_x_percent_5p",
        "dist_agent_1lm_randrot_nohyp_x_percent_10p",
        "dist_agent_1lm_randrot_nohyp_x_percent_20p",
        "dist_agent_1lm_randrot_nohyp_x_percent_30p",
        # "dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all",
    ]
    hyp_experiments = [
        "dist_agent_1lm_randrot_x_percent_5p_perf",
        "dist_agent_1lm_randrot_x_percent_10p_perf",
        "dist_agent_1lm_randrot_x_percent_20p_perf",
        "dist_agent_1lm_randrot_x_percent_30p_perf",
        "dist_agent_1lm_randrot_x_percent_30p_evidence_update_all",
    ]
    save_dir = "~/tbp/results/dmc/results/floppy"

    main(nohyp_experiments, os.path.join(save_dir, "nohyp"))
    # main(hyp_experiments, os.path.join(save_dir, "hyp"))
