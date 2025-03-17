"""This script reads the raw data for experiments related to Floppy on Monty.

Results are stored in `~/tbp/results/dmc/results`
"""

import os
from typing import List

import numpy as np
import pandas as pd


def read_flop_traces(df: pd.DataFrame) -> float:
    """Read the flop traces from a file.
    """
    # Get average of flops for experiment.run_episode in method column
    run_episode_df = df[df["method"] == "experiment.run_episode"]
    # Return as a list
    return run_episode_df["flops"].tolist()


def compute_accuracy(df: pd.DataFrame) -> float:
    """Compute the accuracy from the eval_stats.csv file.

    It is considered accurate if primary_performance is correct or correct_mlh
    """
    correct = df[df["primary_performance"].isin(["correct", "correct_mlh"])]
    return correct.shape[0] / df.shape[0]


def compute_quaternion_error(df: pd.DataFrame) -> float:
    """Compute the quaternion error from the eval_stats.csv file.
    average the rotation_error column
    """
    return df["rotation_error"].mean()


def read_total_flops(df: pd.DataFrame, method: str) -> float:
    """Read the total flops from a file for a specific method.
    """
    method_df = df[df["method"] == method]
    return method_df["flops"].sum()


def main(exp_type: str, experiments: List[str], save_dir: str, pretrain: bool = False):
    data_dir = f"~/tbp/results/dmc/results/floppy/{exp_type}"
    data_dir = os.path.expanduser(data_dir)
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize results DataFrame with appropriate columns
    columns = ["experiment", "flops_mean", "flops_std"]
    if pretrain:
        columns.extend(["total_flops_episode", "total_flops_epoch"])
    else:
        columns.extend(
            [
                "accuracy_mean",
                "accuracy_std",
                "rotation_error_mean",
                "rotation_error_std",
            ]
        )
    results = pd.DataFrame(columns=columns)

    for experiment in experiments:
        # First collect all flops data
        experiment_flops = []
        total_flops_episode = 0
        total_flops_epoch = 0

        # Read all csv files that start with "flop_traces" in results_dir/experiment
        files = os.listdir(os.path.join(data_dir, experiment))
        for file in files:
            if file.startswith("flop_traces"):
                flops_df = pd.read_csv(os.path.join(data_dir, experiment, file))
                flops = read_flop_traces(flops_df)
                experiment_flops.extend(flops)

                if pretrain:
                    total_flops_episode += read_total_flops(
                        flops_df, "experiment.run_episode"
                    )
                    total_flops_epoch += read_total_flops(
                        flops_df, "experiment.run_epoch"
                    )

        # Calculate flops statistics
        flops_mean = np.mean(experiment_flops) if experiment_flops else np.nan
        flops_std = np.std(experiment_flops) if experiment_flops else np.nan

        # Prepare results dictionary
        result_dict = {
            "experiment": [experiment],
            "flops_mean": [flops_mean],
            "flops_std": [flops_std],
        }

        if pretrain:
            result_dict.update(
                {
                    "total_flops_episode": [total_flops_episode],
                    "total_flops_epoch": [total_flops_epoch],
                }
            )
        else:
            accuracies = []
            rotation_errors = []
            for file in files:
                if file.startswith("eval_stats.csv"):
                    eval_df = pd.read_csv(os.path.join(data_dir, experiment, file))
                    accuracies.append(compute_accuracy(eval_df))
                    rotation_errors.append(compute_quaternion_error(eval_df))

            result_dict.update(
                {
                    "accuracy_mean": [np.mean(accuracies) if accuracies else np.nan],
                    "accuracy_std": [np.std(accuracies) if accuracies else np.nan],
                    "rotation_error_mean": [
                        np.mean(rotation_errors) if rotation_errors else np.nan
                    ],
                    "rotation_error_std": [
                        np.std(rotation_errors) if rotation_errors else np.nan
                    ],
                }
            )

        # Add to results
        results = pd.concat([results, pd.DataFrame(result_dict)])

    # Save with appropriate filename
    filename = "flops.csv" if pretrain else "flops_accuracy_rotation_error.csv"
    results.to_csv(os.path.join(save_dir, filename), index=False)


if __name__ == "__main__":
    nohyp_experiments = [
        "dist_agent_1lm_randrot_nohyp_x_percent_5_floppy",
        "dist_agent_1lm_randrot_nohyp_x_percent_10_floppy",
        "dist_agent_1lm_randrot_nohyp_x_percent_20_floppy",
        "dist_agent_1lm_randrot_nohyp_x_percent_40_floppy",
        "dist_agent_1lm_randrot_nohyp_x_percent_60_floppy",
        "dist_agent_1lm_randrot_nohyp_x_percent_80_floppy",
    ]
    hyp_experiments = [
        "dist_agent_1lm_randrot_x_percent_5_floppy",
        "dist_agent_1lm_randrot_x_percent_10_floppy",
        "dist_agent_1lm_randrot_x_percent_20_floppy",
        "dist_agent_1lm_randrot_x_percent_40_floppy",
        "dist_agent_1lm_randrot_x_percent_60_floppy",
        "dist_agent_1lm_randrot_x_percent_80_floppy",
    ]
    save_dir = "~/tbp/results/dmc/results/floppy"

    # main("nohyp", nohyp_experiments, os.path.join(save_dir, "nohyp"))
    # main("hyp", hyp_experiments, os.path.join(save_dir, "hyp"))

    # Add pretrain experiment
    pretrain_experiments = ["pretrain_dist_agent_1lm"]
    main(
        "pretrain",
        pretrain_experiments,
        os.path.join(save_dir, "pretrain"),
        pretrain=True,
    )
