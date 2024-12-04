# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""This script takes in an eval folder and performs various analyses.

Inputs:
    experiment_name: The name of the experiment, e.g. dist_agent_1lm.
    - This will search for the eval_stats.csv file in the results directory (tbp/monty_lab/monty_capabilities_analysis/results/dmc/<experiment_name>)

Outputs:
    - Creates a save_dir in the results directory (tbp/monty_lab/monty_capabilities_analysis/results/dmc/<experiment_name>/analysis)
    - Saves the following files to save_dir:
        - frequent_mistakes.csv: A table of the most common mistakes.
        - mistakes_by_primary_target_object.csv: A table of mistakes grouped by primary target object.
        - stats_summary.csv: A table of the stats summary (accuracy, precision, recall, f1, etc.)
        - confusion_matrix_<num_objects>objs.csv: A table of the confusion matrix.
        - rotation_error_distribution.png: A plot of the rotation error distribution.

Example usage:
python analyze.py --experiment_name=dist_agent_1lm
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

ycb_groupings = {
    "cans": ["master_chef_can", "potted_meat_can", "tomato_soup_can", "tuna_fish_can"],
    "boxes": ["cracker_box", "sugar_box", "pudding_box", "gelatin_box"],
    "fruits": [
        "apple",
        "banana",
        "lemon",
        "strawberry",
        "peach",
        "pear",
        "plum",
        "orange",
    ],
    "kitchenware": [
        "pitcher_base",
        "bleach_cleanser",
        "bowl",
        "mug",
        "sponge",
        "skillet_lid",
        "plate",
        "fork",
        "spoon",
        "knife",
        "spatula",
        "mustard_bottle",
    ],
    "clamps": ["medium_clamp", "large_clamp", "extra_large_clamp"],
    "screwdrivers": ["flat_screwdriver", "phillips_screwdriver"],
    "tools": [
        "chain",
        "foam_brick",
        "power_drill",
        "hammer",
        "scissors",
        "padlock",
        "large_marker",
        "adjustable_wrench",
    ],
    "balls": [
        "golf_ball",
        "baseball",
        "tennis_ball",
        "racquetball",
        "softball",
        "mini_soccer_ball",
    ],
    "wood_blocks": ["wood_block", "a_colored_wood_blocks", "b_colored_wood_blocks"],
    "toy_airplanes": [
        "a_toy_airplane",
        "b_toy_airplane",
        "c_toy_airplane",
        "d_toy_airplane",
        "e_toy_airplane",
    ],
    "marbles": ["a_marbles", "b_marbles"],
    "cups": [
        "a_cups",
        "b_cups",
        "c_cups",
        "d_cups",
        "e_cups",
        "f_cups",
        "g_cups",
        "h_cups",
        "i_cups",
        "j_cups",
    ],
    "lego_duplo": [
        "a_lego_duplo",
        "b_lego_duplo",
        "c_lego_duplo",
        "d_lego_duplo",
        "e_lego_duplo",
        "f_lego_duplo",
        "g_lego_duplo",
    ],
    "cubes": ["rubiks_cube", "dice", "nine_hole_peg_test"],
    "no_observations_yet": ["no_observations_yet"],
}


# @deprecated(
#     reason="The list in result is not sorted by confidence, so we cannot calculate top-k accuracy."
# )
def calculate_top_k_accuracy(df: pd.DataFrame, k: int = 1) -> float:
    """Calculates the top-k accuracy for predictions.

    Args:
        df (pd.DataFrame): DataFrame containing 'primary_target_object' as ground truth
                           and 'result' as prediction(s).
        k (int): Number of top predictions to consider.

    Returns:
        float: Top-k accuracy.
    """
    correct = 0
    for i, row in df.iterrows():
        gt = row["primary_target_object"]

        # Check if result is a list or a single prediction
        if (
            isinstance(row["result"], str)
            and row["result"].startswith("[")
            and row["result"].endswith("]")
        ):
            preds = eval(row["result"])  # Convert string to list if formatted as such
        else:
            preds = [row["result"]]  # Treat single prediction as a list with one item

        # Calculate if ground truth is within the top-k predictions
        if gt in preds[: min(k, len(preds))]:  # Ensure we don't exceed list length
            correct += 1

    return correct / len(df)


def get_mean_std_rotation_error_degrees(df: pd.DataFrame) -> dict:
    """Gets the mean and std of the rotation error in degrees."""
    primary_target_rotation_euler = extract_column_as_array(
        df, "primary_target_rotation_euler"
    )
    most_likely_rotation = extract_column_as_array(df, "most_likely_rotation")
    # Convert the string into numpy array, these start with "[" and end with "]" with three values, e.g. "[0 0 0]"
    # for loop is used to convert each string into numpy array
    # Convert each string to a NumPy array and stack them
    primary_target_rotation_euler = np.array(
        [
            np.fromstring(rotation.strip("[]"), sep=" ")
            for rotation in primary_target_rotation_euler
        ]
    )
    most_likely_rotation = np.array(
        [
            np.fromstring(rotation.strip("[]"), sep=" ")
            for rotation in most_likely_rotation
        ]
    )
    # Convert to quaternions
    primary_target_quaternion = R.from_euler(
        "xyz", primary_target_rotation_euler, degrees=True
    )
    most_likely_quaternion = R.from_euler("xyz", most_likely_rotation, degrees=True)
    rotation_error_quaternion = primary_target_quaternion * most_likely_quaternion.inv()
    # get magnitude of the quaternion
    rotation_error_quaternion = rotation_error_quaternion.magnitude()
    rotation_error_degrees = rotation_error_quaternion * 180 / np.pi
    return {
        "errors": rotation_error_degrees,
        "mean": rotation_error_degrees.mean(),
        "std": rotation_error_degrees.std(),
    }


def extract_column_as_array(
    df: pd.DataFrame, column_name: str = "primary_target_object"
) -> np.ndarray:
    """Extracts a column from a DataFrame as a numpy array. This is a helper function for analyzing results using sklearn.
    Since certain columns can contain lists,

    Args:
        df: DataFrame containing the eval_stats.csv file.
        column_name: Name of the column to extract.
    Returns:
        A numpy array containing the column.
    """
    return df[column_name].to_numpy()


def get_precision_recall_f1_accuracy(
    gt_objects: np.ndarray, pred_objects: np.ndarray
) -> dict:
    """Gets the precision, recall, f1, and accuracy from the gt and pred objects.

    Args:
        gt_objects: Ground truth objects.
        pred_objects: Predicted objects.
    Returns:
        A dictionary containing the precision, recall, and f1 score.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_objects, pred_objects, average="macro"
    )
    accuracy = accuracy_score(gt_objects, pred_objects)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

def calculate_object_recognition_incorrect_percentage(df: pd.DataFrame) -> float:
    """Calculates the percentage of incorrect predictions of Monty model. A prediction is considered incorrect if the primary_performance is in "confused" or "confused_mlh".

    Note that this does not include "patch_off_object" as a mistake.

    Args:
        df: DataFrame containing the eval_stats.csv file.
    Returns:
        A DataFrame containing the mistakes.
    """
    incorrect_predictions = df["primary_performance"].isin(["confused", "confused_mlh"])
    incorrect_percentage = incorrect_predictions.mean()
    return incorrect_percentage


def calculate_object_recognition_correct_percentage(df: pd.DataFrame) -> float:
    """Calculates the percentage of correct predictions of Monty model. A prediction is considered correct if the primary_performance is in "correct" or "correct_mlh".

    Args:
        df: DataFrame containing the eval_stats.csv file.
    Returns:
        The accuracy of the model as a float.
    """
    correct_predictions = df["primary_performance"].isin(["correct", "correct_mlh"])
    correct_percentage = correct_predictions.mean()
    return correct_percentage

def get_frequent_object_recognition_mistakes(df: pd.DataFrame) -> pd.DataFrame:
    """Gets the most common mistakes in object recognition. This will sort the mistakes by frequency.

    Args:
        df: DataFrame containing the eval_stats.csv file.
    Returns:
        A DataFrame containing the most common mistakes sorted by frequency.
    """
    mistakes = df[df["primary_performance"].isin(["confused", "confused_mlh"])]
    mistake_counts = (
        mistakes.groupby("primary_target_object").size().sort_values(ascending=False)
    )
    mistake_counts = mistake_counts.reset_index()
    mistake_counts.columns = ["primary_target_object", "count"]
    return mistake_counts


def get_most_frequent_object_recognition_mistake(df: pd.DataFrame) -> dict:
    """Gets the most frequent object recognition mistake and its frequency.

    Args:
        df: DataFrame containing the eval_stats.csv file.
    Returns:
        A dictionary containing the most frequent mistake and its frequency.
    """
    mistake_counts = get_frequent_object_recognition_mistakes(df)
    most_frequent_mistake = mistake_counts.iloc[0]["primary_target_object"]
    most_frequent_mistake_frequency = mistake_counts.iloc[0]["count"]
    return {
        "most_frequent_mistake": most_frequent_mistake,
        "most_frequent_mistake_frequency": most_frequent_mistake_frequency,
    }


def get_mistakes_by_primary_target_object(df: pd.DataFrame) -> pd.DataFrame:
    """Gets the mistakes grouped by primary_target_object.

    Args:
        df: DataFrame containing the eval_stats.csv file.
    Returns:
        A DataFrame containing the mistakes grouped by primary_target_object.
    """
    mistakes = df[df["primary_performance"].isin(["confused", "confused_mlh"])]
    evidence_counts = mistakes.groupby("primary_target_object")[
        [
            "primary_performance",
            "highest_evidence",
            "num_steps",
            "result",
            "individual_ts_performance",
        ]
    ].value_counts()
    evidence_counts = evidence_counts.reset_index()
    evidence_counts = evidence_counts.sort_values("primary_performance")
    return evidence_counts

def calculate_confusion_matrix(gt_objects, pred_objects) -> np.ndarray:
    """Calculates the confusion matrix."""
    # get unique labels
    labels = np.unique(gt_objects)
    # calculate confusion matrix
    cm = confusion_matrix(gt_objects, pred_objects, labels=labels)
    return cm


def plot_rotation_errors(
    errors,
    n_bins=36,
    figsize=(16, 8),
    color_palette="coolwarm",
    max_degree=180,
    show_stats=True,
    title="Rotation Error Distribution",
    save_path=None,
):
    """
    Create a circular histogram (rose plot) for rotation errors spanning 0-180 degrees.

    Parameters:
    -----------
    errors : array-like
        Array of rotation errors in degrees (0-180)
    n_bins : int
        Number of bins in the histogram
    figsize : tuple
        Figure size (width, height)
    color_palette : str
        Name of the seaborn color palette to use
    max_degree : float
        Maximum degree to show in the plot (default 180)
    show_stats : bool
        Whether to show statistics
    title : str
        Plot title
    """
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    # Circular histogram (left subplot)
    ax1 = fig.add_subplot(gs[0], projection="polar")

    # Calculate histogram
    bins = np.linspace(0, max_degree, n_bins + 1)
    hist, bin_edges = np.histogram(np.abs(errors), bins=bins)

    # Calculate bin centers and widths
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    # Convert to radians for plotting
    # Map 0-180 degrees to 0-360 degrees in the plot (double the angles)
    angles = np.deg2rad(bin_centers * 2)  # Multiply by 2 to span full circle
    widths = np.deg2rad(bin_widths * 2)  # Multiply widths by 2 as well

    # Create color gradient
    colors = sns.color_palette(color_palette, n_colors=len(hist))

    # Plot bars
    bars = ax1.bar(angles, hist, width=widths, bottom=0.0)

    # Color the bars according to the gradient
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
        bar.set_alpha(0.7)

    # Customize the polar plot
    ax1.set_theta_zero_location("N")  # 0 degrees at the top
    ax1.set_theta_direction(-1)  # Clockwise direction

    # Set the rlabel position to 0 degrees
    ax1.set_rlabel_position(0)

    # Add degree labels (0 to 180, mapped to 0 to 360)
    degrees = np.linspace(0, 180, 9)  # 9 labels from 0 to 180
    angles = np.deg2rad(degrees * 2)  # Convert to the doubled angles we're using
    labels = [f"{int(deg)}°" for deg in degrees]
    ax1.set_xticks(angles)
    ax1.set_xticklabels(labels)

    # Regular histogram (right subplot)
    ax2 = fig.add_subplot(gs[1])

    # Plot regular histogram with KDE
    sns.histplot(
        data=np.abs(errors), bins=n_bins, color="gray", alpha=0.5, kde=True, ax=ax2
    )
    ax2.set_xlabel("Rotation Error (degrees)")
    ax2.set_ylabel("Count")

    # Set x-axis limit to match max_degree
    ax2.set_xlim(0, max_degree)

    # Add statistics if requested
    if show_stats:
        stats_text = (
            f"Mean: {np.mean(np.abs(errors)):.2f}°\n"
            f"Median: {np.median(np.abs(errors)):.2f}°\n"
            f"Std Dev: {np.std(np.abs(errors)):.2f}°\n"
            f"90th percentile: {np.percentile(np.abs(errors), 90):.2f}°"
        )
        ax2.text(
            0.95,
            0.95,
            stats_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Set title for the entire figure
    fig.suptitle(title, y=1.02, fontsize=14)

    # Adjust layout
    plt.tight_layout()
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_large_confusion_matrix(
    cm: np.ndarray,
    labels=None,
    figsize=(20, 20),
    min_display_value=0.01,
    log_scale=True,
    save_path=None,
    show_colorbar=True,
):
    """
    Create a readable confusion matrix visualization for large number of classes.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        List of label names
    figsize : tuple, optional
        Figure size in inches
    min_display_value : float, optional
        Minimum value to display in the matrix (smaller values shown as blank)
    log_scale : bool, optional
        Whether to use logarithmic color scaling
    save_path : str, optional
        Path to save the figure
    show_colorbar : bool, optional
        Whether to show the colorbar

    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the confusion matrix
    """
    # Convert to percentages per row
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Define color mapping
    if log_scale:
        norm = LogNorm(
            vmin=max(min_display_value, cm_normalized[cm_normalized > 0].min()),
            vmax=cm_normalized.max(),
        )
    else:
        norm = None

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        ax=ax,
        cmap="YlOrRd",
        norm=norm,
        square=True,
        cbar=show_colorbar,
        xticklabels=labels if labels is not None else range(cm.shape[1]),
        yticklabels=labels if labels is not None else range(cm.shape[0]),
    )
    # Calculate appropriate font size for labels based on figsize
    # Calculate base font size based on figure width
    base_font_size = figsize[0] * 1.5  # A multiplier for readability; adjust as needed

    # Apply font sizes
    plt.rcParams.update(
        {
            "font.size": base_font_size,  # General font size
            "axes.titlesize": base_font_size * 1.2,  # Axis title font size
            "axes.labelsize": base_font_size,  # Axis label font size
            "xtick.labelsize": base_font_size * 0.8,  # X tick font size
            "ytick.labelsize": base_font_size * 0.8,  # Y tick font size
            "legend.fontsize": base_font_size * 0.8,  # Legend font size
        }
    )
    # Customize appearance
    plt.title("Confusion Matrix (Normalized %)", pad=20)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add grid to help with readability
    ax.grid(False)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def main(results_dir: str):
    # Check existence of results_dir
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory {results_dir} does not exist")

    # Load eval_stats.csv
    eval_stats_path = os.path.join(results_dir, "eval_stats.csv")
    eval_stats = pd.read_csv(eval_stats_path)

    # Create save directory
    save_dir = os.path.join(results_dir, "analysis")
    os.makedirs(save_dir, exist_ok=True)

    #### Saving additional csvs from eval_stats.csv ####
    # Get frequent object recognition mistakes and save to csv
    frequent_mistakes = get_frequent_object_recognition_mistakes(eval_stats)
    frequent_mistakes.to_csv(
        os.path.join(save_dir, "frequent_mistakes.csv"), index=False
    )

    # Get mistakes by primary target object and save to csv
    mistakes_by_primary_target_object = get_mistakes_by_primary_target_object(
        eval_stats
    )
    mistakes_by_primary_target_object.to_csv(
        os.path.join(save_dir, "mistakes_by_primary_target_object.csv"), index=False
    )

    #### Collecting stats from eval_stats.csv ####
    stats_summary = {}
    stats_summary["correct_percentage"] = (
        calculate_object_recognition_correct_percentage(eval_stats)
    )
    stats_summary["incorrect_percentage"] = (
        calculate_object_recognition_incorrect_percentage(eval_stats)
    )
    most_frequent_mistake = get_most_frequent_object_recognition_mistake(eval_stats)
    stats_summary["most_frequent_mistake"] = most_frequent_mistake[
        "most_frequent_mistake"
    ]
    stats_summary["most_frequent_mistake_frequency"] = most_frequent_mistake[
        "most_frequent_mistake_frequency"
    ]
    gt_objects = extract_column_as_array(eval_stats, "primary_target_object")
    pred_objects_top_1 = extract_column_as_array(eval_stats, "most_likely_object")

    precision_recall_f1 = get_precision_recall_f1_accuracy(
        gt_objects, pred_objects_top_1
    )
    stats_summary["precision"] = precision_recall_f1["precision"]
    stats_summary["recall"] = precision_recall_f1["recall"]
    stats_summary["f1"] = precision_recall_f1["f1"]
    stats_summary["accuracy"] = precision_recall_f1["accuracy"]
    rotation_error_degrees = get_mean_std_rotation_error_degrees(eval_stats)
    # Save rotation_error degrees with eval_stats
    rotation_error_degrees_df = pd.DataFrame(
        rotation_error_degrees["errors"], columns=["calculated_rotation_error"]
    )
    # Add "primary_performance" column, num_steps, rotation_error, result, most_likely_object, primary_target_object
    rotation_error_degrees_df["primary_performance"] = eval_stats["primary_performance"]
    rotation_error_degrees_df["num_steps"] = eval_stats["num_steps"]
    rotation_error_degrees_df["rotation_error"] = eval_stats["rotation_error"]
    rotation_error_degrees_df["result"] = eval_stats["result"]
    rotation_error_degrees_df["most_likely_object"] = eval_stats["most_likely_object"]
    rotation_error_degrees_df["primary_target_object"] = eval_stats[
        "primary_target_object"
    ]
    rotation_error_degrees_df.to_csv(
        os.path.join(save_dir, "rotation_error_degrees.csv"), index=False
    )
    stats_summary["rotation_error_degrees_mean"] = rotation_error_degrees["mean"]
    stats_summary["rotation_error_degrees_std"] = rotation_error_degrees["std"]

    # Get rotation error for correct predictions
    correct_predictions = eval_stats[
        eval_stats["primary_performance"].isin(["correct", "correct_mlh"])
    ]
    rotation_error_degrees_correct = get_mean_std_rotation_error_degrees(
        correct_predictions
    )
    stats_summary["rotation_error_degrees_correct_mean"] = (
        rotation_error_degrees_correct["mean"]
    )
    stats_summary["rotation_error_degrees_correct_std"] = (
        rotation_error_degrees_correct["std"]
    )

    # Get rotation error for incorrect predictions
    incorrect_predictions = eval_stats[
        eval_stats["primary_performance"].isin(["confused", "confused_mlh"])
    ]
    rotation_error_degrees_incorrect = get_mean_std_rotation_error_degrees(
        incorrect_predictions
    )
    stats_summary["rotation_error_degrees_incorrect_mean"] = (
        rotation_error_degrees_incorrect["mean"]
    )
    stats_summary["rotation_error_degrees_incorrect_std"] = (
        rotation_error_degrees_incorrect["std"]
    )

    # Convert to df and save to csv
    stats_summary_df = pd.DataFrame([stats_summary])
    stats_summary_df.to_csv(os.path.join(save_dir, "stats_summary.csv"), index=False)

    # Calculate confusion matrix
    num_objects = 77
    labels = np.unique(gt_objects)
    cm = calculate_confusion_matrix(gt_objects, pred_objects_top_1)
    # Convert to a DataFrame for better readability
    confusion_matrix_df = pd.DataFrame(cm, index=labels, columns=labels)
    confusion_matrix_df.to_csv(
        os.path.join(save_dir, f"confusion_matrix_{num_objects}objs.csv"), index=False
    )

    # Plot and save confusion matrix
    _ = plot_large_confusion_matrix(
        cm,
        labels=labels,
        save_path=os.path.join(save_dir, f"confusion_matrix_{num_objects}objs.png"),
    )
    # invert ycb_groupings to get the reverse mapping
    ycb_groupings_inv = {}
    for group, objects in ycb_groupings.items():
        for obj in objects:
            ycb_groupings_inv[obj] = group
    # Get coarse-grained gt_objects and pred_objects_top_1
    gt_objects_coarse = [ycb_groupings_inv[gt_object] for gt_object in gt_objects]
    pred_objects_top_1_coarse = [
        ycb_groupings_inv[pred_object] for pred_object in pred_objects_top_1
    ]
    # Calculate coarse-grained confusion matrix
    cm_coarse = calculate_confusion_matrix(gt_objects_coarse, pred_objects_top_1_coarse)
    num_objects_coarse = len(np.unique(gt_objects_coarse))
    coarse_labels = np.unique(gt_objects_coarse)
    # Convert to a DataFrame for better readability
    confusion_matrix_df_coarse = pd.DataFrame(
        cm_coarse, index=coarse_labels, columns=coarse_labels
    )
    confusion_matrix_df_coarse.to_csv(
        os.path.join(save_dir, f"confusion_matrix_{num_objects_coarse}objs_coarse.csv"),
        index=False,
    )
    # Plot and save coarse-grained confusion matrix
    _ = plot_large_confusion_matrix(
        cm_coarse,
        labels=coarse_labels,
        save_path=os.path.join(
            save_dir, f"confusion_matrix_{num_objects_coarse}objs_coarse.png"
        ),
    )

    # Plot rotation errors
    _ = plot_rotation_errors(
        rotation_error_degrees["errors"],
        title="Rotation Error Distribution",
        save_path=os.path.join(save_dir, "rotation_error_distribution.png"),
    )

    # Plot without zeros use epsilon for very small values
    epsilon = 1e-3
    rotation_error_degrees_no_zeros = rotation_error_degrees["errors"][
        rotation_error_degrees["errors"] > epsilon
    ]
    _ = plot_rotation_errors(
        rotation_error_degrees_no_zeros,
        title="Rotation Error Distribution (without zeros)",
        save_path=os.path.join(save_dir, "rotation_error_distribution_no_zeros.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        default="dist_agent_1lm",
        help="Name of the experiment, e.g. dist_agent_1lm",
    )
    args = parser.parse_args()
    results_dir = os.path.join(
        os.path.expanduser("~"),
        "tbp/monty_lab/monty_capabilities_analysis/results/dmc",
        args.experiment_name,
    )
    main(results_dir)
