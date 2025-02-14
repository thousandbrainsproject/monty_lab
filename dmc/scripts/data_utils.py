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
Data I/O, filesystem, and other utilities.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

DMC_ROOT_DIR = Path(os.environ.get("DMC_ROOT_DIR", "~/tbp/results/dmc")).expanduser()
DMC_PRETRAIN_DIR = DMC_ROOT_DIR / "pretrained_models"
DMC_RESULTS_DIR = DMC_ROOT_DIR / "results"
DMC_ANALYSIS_DIR = Path(
    os.environ.get("DMC_ANALYSIS_DIR", "~/tbp/results/dmc_analysis")
).expanduser()
DMC_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_eval_stats(exp: os.PathLike) -> pd.DataFrame:
    """Load `eval_stats.csv`

    Convenience function for loading `eval_stats.csv` from a DMC experiment name.

    Args:
        exp (os.PathLike): Name of a DMC experiment, a directory containing
        `eval_stats.csv`, or a complete path to `eval_stats.csv`.

    Returns:
        pd.DataFrame: The loaded dataframe. Includes generated columns `episode` and
        `epoch`.
    """

    path = Path(exp).expanduser()

    if path.exists():
        # Case 1: Given a path to a csv file.
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(exp, index_col=0)
        # Case 2: Given a path to a directory containing eval_stats.csv.
        elif (path / "eval_stats.csv").exists():
            df = pd.read_csv(path / "eval_stats.csv", index_col=0)
        else:
            raise FileNotFoundError(f"No eval_stats.csv found for {exp}")
    else:
        # Given a run name. Look in DMC folder.
        df = pd.read_csv(DMC_RESULTS_DIR / path / "eval_stats.csv", index_col=0)

    # Collect basic info, like number of LMs, objects, number of episodes, etc.
    n_lms = len(np.unique(df["lm_id"]))
    object_names = np.unique(df["primary_target_object"])
    n_objects = len(object_names)

    # Add 'episode' column.
    assert len(df) % n_lms == 0
    n_episodes = int(len(df) / n_lms)
    df["episode"] = np.repeat(np.arange(n_episodes), n_lms)

    # Add 'epoch' column.
    rows_per_epoch = n_objects * n_lms
    assert len(df) % rows_per_epoch == 0
    n_epochs = int(len(df) / rows_per_epoch)
    df["epoch"] = np.repeat(np.arange(n_epochs), rows_per_epoch)

    return df


def get_percent_correct(df: pd.DataFrame) -> float:
    """Get percent of correct object recognition for an `eval_stats` dataframe.

    Uses the 'primary_performance' column. Values 'correct' or 'correct_mlh' count
    as correct.

    """
    n_correct = df.primary_performance.str.startswith("correct").sum()
    return 100 * n_correct / len(df)
