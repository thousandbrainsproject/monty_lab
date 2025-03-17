"""This script reads the raw results from FLOPs measurements for pretraining.
FLOPs results are stored in `/Users/hlee/tbp/results/dmc/results/floppy/pretrain/pretrain_dist_agent_1lm_parallel_perf.csv`
"""

from pathlib import Path

import pandas as pd


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
    flops_path = Path(
        "/Users/hlee/tbp/results/dmc/results/floppy/pretrain/pretrain_dist_agent_1lm_perf.csv"
    )

    if not flops_path.exists():
        print(f"Error: File not found at {flops_path}")
        return

    try:
        flops_df = pd.read_csv(flops_path, comment="#", header=None)
        total_flops = compute_flops(flops_df)
        print(f"Total training FLOPs: {total_flops:,}")
    except Exception as e:
        print(f"Error processing file: {e!s}")


if __name__ == "__main__":
    main()
