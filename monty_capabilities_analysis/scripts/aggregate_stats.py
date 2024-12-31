# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import glob
import os

import pandas as pd

# Results directory
results_parent_dir = os.path.join(
    os.path.expanduser("~"),
    "tbp/monty_lab/monty_capabilities_analysis/results/dmc",
)
results_dirs = glob.glob(os.path.join(results_parent_dir, "*"))

# Read analysis/stats.csv from each results directory
stats_dfs = []
for results_dir in results_dirs:
    stats_path = os.path.join(results_dir, "analysis", "stats_summary.csv")
    if not os.path.exists(stats_path):
        print(f"Stats file not found in {results_dir}")
        continue
    stats_df = pd.read_csv(stats_path)
    stats_df["experiment_name"] = results_dir.split("/")[-1]
    stats_dfs.append(stats_df)

# Concatenate all stats dataframes
all_stats_df = pd.concat(stats_dfs, ignore_index=True)

# Move experiment_name to the first column
all_stats_df = all_stats_df[
    ["experiment_name"]
    + [col for col in all_stats_df.columns if col != "experiment_name"]
]
# Sort by experiment_name
all_stats_df = all_stats_df.sort_values(by="experiment_name")

# Save to csv
all_stats_path = os.path.join(results_parent_dir, "all_stats_summary.csv")
all_stats_df.to_csv(all_stats_path, index=False)
