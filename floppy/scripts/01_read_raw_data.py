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
    return run_episode_df["flops"].mean()


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

    results = pd.DataFrame()
    for experiment in experiments:
        # Read all csv files that start with "flop_traces" in results_dir/experiment
        files = os.listdir(os.path.join(data_dir, experiment))
        for file in files:
            if file.startswith("flop_traces"):
                flops_df = pd.read_csv(os.path.join(data_dir, experiment, file))
                flops = read_flop_traces(flops_df)
            if file.startswith("eval_stats.csv"):
                eval_df = pd.read_csv(os.path.join(data_dir, experiment, file))
                accuracy = compute_accuracy(eval_df)
                quaternion_error_deg = compute_quaternion_error(eval_df)

                results = pd.concat(
                    [
                        results,
                        pd.DataFrame(
                            {
                                "experiment": [experiment],
                                "accuracy_mean": [accuracy],
                                "accuracy_std": [np.nan],
                                "rotation_error_mean": [quaternion_error_deg],
                                "rotation_error_std": [np.nan],
                                "flops": [flops],
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
        "dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all",
    ]
    hyp_experiments = [
        "dist_agent_1lm_randrot_x_percent_5p",
        "dist_agent_1lm_randrot_x_percent_10p",
        "dist_agent_1lm_randrot_x_percent_20p",
        "dist_agent_1lm_randrot_x_percent_30p_evidence_update_all",
    ]
    save_dir = "~/tbp/results/dmc/results/floppy"

    main(nohyp_experiments, os.path.join(save_dir, "nohyp"))
    main(hyp_experiments, os.path.join(save_dir, "hyp"))
