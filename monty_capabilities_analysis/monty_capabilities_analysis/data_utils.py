"""
Data I/O, filesystem, and other utilities.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Directory containing DMC results.
# RESULTS_DIR = Path(
#     "~/tbp/monty_lab/monty_capabilities_analysis/results/dmc"
# ).expanduser()
DMC_ROOT = Path("~/tbp/results/dmc").expanduser()
PRETRAIN_DIR = DMC_ROOT / "pretrained_models"
RESULTS_DIR = DMC_ROOT / "results"

OUT_DIR = Path("~/tbp/dmc_analysis").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_eval_stats(exp: os.PathLike) -> pd.DataFrame:
    """Load `eval_stats.csv`

    Args:
        exp (os.PathLike): Path to a csv-file or a directory containing
        `eval_stats.csv`.

    Returns:
        pd.DataFrame: The loaded dataframe. Includes generated columns `episode` and
        `epoch`.
    """

    path = Path(exp).expanduser()

    if path.exists():
        # Case 1: Given a path to a csv file.
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(exp, index_col=0)
        # Case 2: Given a path to an experiment directory containing eval_stats.csv.
        elif (path / "eval_stats.csv").exists():
            df = pd.read_csv(path / "eval_stats.csv", index_col=0)
        else:
            raise FileNotFoundError(f"No eval_stats.csv found for {exp}")
    else:
        # Given a run name. Look in DMC folder.
        df = pd.read_csv(RESULTS_DIR / path / "eval_stats.csv", index_col=0)

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
