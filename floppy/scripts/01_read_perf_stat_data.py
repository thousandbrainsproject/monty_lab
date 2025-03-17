"""This script reads the raw results from FLOPs measurements and accuracy and rotation errors and combines them together for easy visualization.

FLOPs results are stored in `~/tbp/results/dmc/results/perf/monty/raw/{experiment}.csv`
Eval Stats results are stored in `/tbp/results/dmc/results/perf/monty/raw/{experiment}/eval_stats.csv`
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


def compute_accuracy(predictions_df: pd.DataFrame) -> Tuple[float, float]:
    """If primary_performance is correct or correct_mlh, then it is considered correct.
    """
    correct = predictions_df[
        predictions_df["primary_performance"].isin(["correct", "correct_mlh"])
    ]
    accuracy = correct.shape[0] / predictions_df.shape[0]
    std = 0
    return accuracy, std


def compute_rotation_error(predictions_df: pd.DataFrame) -> Tuple[float, float]:
    """Compute mean and std of rotation error from predictions dataframe
    """
    mean_rotation_error = predictions_df["rotation_error"].mean()
    std_rotation_error = predictions_df["rotation_error"].std()
    return mean_rotation_error, std_rotation_error


def compute_flops(flops_df: pd.DataFrame) -> float:
    """Compute total flops from flops dataframe from perf stat
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

    total_flops = sum(
        row[0] * flops_per_operation[row[2]]
        for _, row in flops_df.iterrows()
        if row[0] > 0 and row[2] in flops_per_operation
    )
    return total_flops


def main():
    base_dir = Path("~/tbp/results/dmc/results").expanduser().resolve()
    save_dir = base_dir / "perf/monty/raw"
    save_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        "dist_agent_1lm_randrot_x_percent_5",
        "dist_agent_1lm_randrot_x_percent_10",
        "dist_agent_1lm_randrot_x_percent_20",
        "dist_agent_1lm_randrot_x_percent_40",
        "dist_agent_1lm_randrot_x_percent_60",
        "dist_agent_1lm_randrot_x_percent_80",
        "dist_agent_1lm_randrot_nohyp_x_percent_5",
        "dist_agent_1lm_randrot_nohyp_x_percent_10",
        "dist_agent_1lm_randrot_nohyp_x_percent_20",
        "dist_agent_1lm_randrot_nohyp_x_percent_40",
        "dist_agent_1lm_randrot_nohyp_x_percent_60",
        "dist_agent_1lm_randrot_nohyp_x_percent_80",
    ]

    results = []
    for experiment in experiments:
        flops_path = (
            base_dir / "perf/monty/raw/max_nneighbors=10" / f"{experiment}p_perf.csv"
        )
        eval_path = (
            base_dir
            / "perf/monty/raw/max_nneighbors=10"
            / experiment
            / "eval_stats.csv"
        )

        if not all(p.exists() for p in [flops_path, eval_path]):
            print(f"Skipping {experiment}: required files not found")
            continue

        try:
            flops_df = pd.read_csv(flops_path, comment="#", header=None)
            eval_df = pd.read_csv(eval_path)

            results.append(
                {
                    "experiment": experiment,
                    "accuracy_mean": compute_accuracy(eval_df)[0],
                    "accuracy_std": compute_accuracy(eval_df)[1],
                    "rotation_error_mean": compute_rotation_error(eval_df)[0],
                    "rotation_error_std": compute_rotation_error(eval_df)[1],
                    "total_train_flops_per_epoch": compute_flops(flops_df),
                }
            )
        except Exception as e:
            print(f"Error processing {experiment}: {e!s}")
            continue

    results_df = pd.DataFrame(results)
    output_path = save_dir / "aws_perf_flops_accuracy_rotation_error.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
