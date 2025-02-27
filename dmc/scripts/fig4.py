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
Figure 4: Visualize 8-patch view finder
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage
from data_utils import (
    DMC_ANALYSIS_DIR,
    VISUALIZATION_RESULTS_DIR,
    DetailedJSONStatsInterface,
    load_eval_stats,
)
from plot_utils import axes3d_set_aspect_equal

OUT_DIR = DMC_ANALYSIS_DIR / "fig4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_8lm_patches():
    """Plot the 8-SM + view_finder visualization for figure 4.

    Uses data from the experiment `fig4_visualize_8lm_patches` defined in
    `configs/visualizations.py`. This function renders the sensor module
    RGBA data in the scene (in 3D) and overlays the sensor module's patch
    boundaries.

    Creates:
     - $DMC_ANALYSIS_DIR/fig4/8lm_patches.png
     - $DMC_ANALYSIS_DIR/fig4/8lm_patches.svg

    """

    # Load the detailed stats.
    experiment_dir = VISUALIZATION_RESULTS_DIR / "fig4_visualize_8lm_patches"
    detailed_stats_path = experiment_dir / "detailed_run_stats.json"
    detailed_stats_interface = DetailedJSONStatsInterface(detailed_stats_path)
    stats = detailed_stats_interface[0]

    def pull(sm_num: int):
        """Helper function for extracting necessary sensor data."""
        sm_dict = stats[f"SM_{sm_num}"]
        # Extract RGBA sensor patch.
        rgba_2d = np.array(sm_dict["raw_observations"][0]["rgba"]) / 255.0
        n_rows, n_cols = rgba_2d.shape[0], rgba_2d.shape[1]

        # Extract locations and on-object filter.
        semantic_3d = np.array(sm_dict["raw_observations"][0]["semantic_3d"])
        pos_1d = semantic_3d[:, 0:3]
        pos_2d = pos_1d.reshape(n_rows, n_cols, 3)
        on_object_1d = semantic_3d[:, 3].astype(int) > 0
        on_object_2d = on_object_1d.reshape(n_rows, n_cols)

        # Filter out points that aren't on-object. Yields a flat list of points/colors.
        return rgba_2d, pos_2d, on_object_2d

    # Create a 3D plot of the semantic point cloud
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=90, azim=-90, roll=0)
    ax.set_proj_type("persp", focal_length=0.125)
    ax.dist = 4.55

    # Render the view finder's RGBA data in the scene.
    rgba_2d, pos_2d, on_object_2d = pull(8)
    rows, cols = np.where(on_object_2d)
    pos_valid_1d = pos_2d[on_object_2d]
    rgba_valid_1d = rgba_2d[on_object_2d]
    ax.scatter(
        pos_valid_1d[:, 0],
        pos_valid_1d[:, 1],
        pos_valid_1d[:, 2],
        c=rgba_valid_1d,
        marker="o",
        alpha=0.3,
        zorder=5,
        s=10,
        edgecolors="none",
    )

    # Render patches and patch boundaries for all sensors.
    for i in range(8):
        # Load sensor data.
        rgba_2d, pos_2d, on_object_2d = pull(i)
        rows, cols = np.where(on_object_2d)
        pos_valid_1d = pos_2d[on_object_2d]
        rgba_valid_1d = rgba_2d[on_object_2d]

        # Render the patch.
        ax.scatter(
            pos_valid_1d[:, 0],
            pos_valid_1d[:, 1],
            pos_valid_1d[:, 2],
            c=rgba_valid_1d,
            marker="o",
            alpha=1,
            zorder=10,
            edgecolors="none",
            s=1,
        )

        # Draw the patch boundaries (complicated).
        n_rows, n_cols = on_object_2d.shape
        row_mid, col_mid = n_rows // 2, n_cols // 2
        n_pix_on_object = on_object_2d.sum()

        if n_pix_on_object == 0:
            contours = []
        elif n_pix_on_object == on_object_2d.size:
            temp = np.zeros((n_rows, n_cols), dtype=bool)
            temp[0, :] = True
            temp[-1, :] = True
            temp[:, 0] = True
            temp[:, -1] = True
            contours = [np.argwhere(temp)]
        else:
            contours = skimage.measure.find_contours(
                on_object_2d, level=0.5, positive_orientation="low"
            )
            contours = [] if contours is None else contours

        for ct in contours:
            row_mid, col_mid = n_rows // 2, n_cols // 2

            # Contour may be floating point (fractional indices from scipy). If so,
            # round rows/columns towards the center of the patch.
            if not np.issubdtype(ct.dtype, np.integer):
                # Round towards the center.
                rows, cols = ct[:, 0], ct[:, 1]
                rows_new, cols_new = np.zeros_like(rows), np.zeros_like(cols)
                rows_new[rows >= row_mid] = np.floor(rows[rows >= row_mid])
                rows_new[rows < row_mid] = np.ceil(rows[rows < row_mid])
                cols_new[cols >= col_mid] = np.floor(cols[cols >= col_mid])
                cols_new[cols < col_mid] = np.ceil(cols[cols < col_mid])
                ct_new = np.zeros_like(ct, dtype=int)
                ct_new[:, 0] = rows_new.astype(int)
                ct_new[:, 1] = cols_new.astype(int)
                ct = ct_new

            # Drop any points that happen to be off-object (it's possible that
            # some boundary points got rounded off-object).
            points_on_object = on_object_2d[ct[:, 0], ct[:, 1]]
            ct = ct[points_on_object]

            # In order to plot the boundary as a line, we need the points to
            # be in order. We can order them by associating each point with its
            # angle from the center of the patch. This isn't a general solution,
            # but it works here.
            Y, X = row_mid - ct[:, 0], ct[:, 1] - col_mid  # pixel to X/Y coords.
            theta = np.arctan2(Y, X)
            sort_order = np.argsort(theta)
            ct = ct[sort_order]

            # Finally, plot the contour.
            xyz = pos_2d[ct[:, 0], ct[:, 1]]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="k", linewidth=3, zorder=20)

    axes3d_set_aspect_equal(ax)
    ax.axis("off")
    plt.show()

    fig.savefig(OUT_DIR / "8lm_patches.png", dpi=300)
    fig.savefig(OUT_DIR / "8lm_patches.svg")


half_lms_match = {
    "dist_agent_1lm_randrot_noise": {"n_lms_match": 1},
    "dist_agent_2lm_half_lms_match_randrot_noise": {"n_lms_match": 1},
    "dist_agent_4lm_half_lms_match_randrot_noise": {"n_lms_match": 2},
    "dist_agent_8lm_half_lms_match_randrot_noise": {"n_lms_match": 4},
    "dist_agent_16lm_half_lms_match_randrot_noise": {"n_lms_match": 8},
}

fixed_min_lms_match = {
    "dist_agent_1lm_randrot_noise": {"n_lms_match": 1},
    "dist_agent_2lm_fixed_min_lms_match_randrot_noise": {"n_lms_match": 2},
    "dist_agent_4lm_fixed_min_lms_match_randrot_noise": {"n_lms_match": 2},
    "dist_agent_8lm_fixed_min_lms_match_randrot_noise": {"n_lms_match": 2},
    "dist_agent_16lm_fixed_min_lms_match_randrot_noise": {"n_lms_match": 2},
}

groups = {
    "half_lms_match": half_lms_match,
    "fixed_min_lms_match": fixed_min_lms_match,
}

for g in groups.values():
    for experiment, dct in g.items():
        dct["dataframe"] = load_eval_stats(experiment)
