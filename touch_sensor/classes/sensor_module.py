# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
from skimage.color import rgb2hsv

from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatVisionPatchSM,
)
from tbp.monty.frameworks.utils.sensor_processing import (
    get_curvature_at_point,
    scale_clip,
)
from utils.touch_utils import get_point_normal


class HabitatTouchPatchSM(HabitatVisionPatchSM):
    """Just adding a depth feature to existing sensor module class

    Sensor Module that turns Habitat camery obs into features at locations.

    Takes in camera rgba and depth input and calculates locations from this.
    It also extracts features which are currently: on_object, rgba, point_normal,
    curvature.
    """

    def __init__(self, sensor_module_id, features, save_raw_obs=False):
        """Initialize Sensor Module.

        :param sensor_module_id: Name of sensor module.
        :param features: Which features to extract. In [on_object, rgba, point_normal,
                principal_curvatures, curvature_directions, gaussian_curvature,
                mean_curvature]
        :param save_raw_obs: Whether to save raw sensory input for logging.

        NOTE: When using feature at location matching with graphs, point_normal
             and on_object needs to be in the list of features.
        NOTE: gaussian_curvature and mean_curvature should be used together to
            contain the same information as principal_curvatures.
        NOTE: point_normal and curvature_directions will be provided in the sensors
            reference frame -> remember to transform them.
        """
        super(HabitatVisionPatchSM, self).__init__(sensor_module_id, save_raw_obs)
        possible_features = [
            "on_object",
            "object_coverage",
            "rgba",
            "depth",
            "hsv",
            "point_normal",
            "principal_curvatures",
            "curvature_directions",
            "gaussian_curvature",
            "mean_curvature",
            "gaussian_curvature_sc",
            "mean_curvature_sc",
        ]
        for feature in features:
            assert feature in possible_features

        self.features = features
        self.processed_obs = []
        self.states = []

    def step(self, data):
        """Turn raw observations into dict of features at location.
        ADDED: depth feature, modified on_object feature
        """
        DetailedLoggingSM.step(self, data)  # for logging

        obs_3d = data["semantic_3d"]
        rgba_feat = data["rgba"]
        depth_feat = data["depth"].reshape(data["depth"].size, 1)
        # Assuming squared patches
        center_row_col = rgba_feat.shape[0] // 2
        # Calculate center ID for flat semantic obs
        obs_dim = int(np.sqrt(obs_3d.shape[0]))
        half_obs_dim = obs_dim // 2
        center_id = half_obs_dim + obs_dim * half_obs_dim
        # Extract all specified features
        features = dict()
        if "point_normal" in self.features:
            # Need to get point normal for graph matching with features
            point_normal = get_point_normal(
                obs_3d, center_id, sensor_location=self.state["location"]
            )
            features["point_normal"] = point_normal
        if "rgba" in self.features:
            features["rgba"] = rgba_feat[center_row_col, center_row_col]
        if "depth" in self.features:
            # depth at 'center' (just using a heuristic for now) of points on object
            # semantic_only_depth_feat = depth_feat[obs_3d[:, 3] != 0]
            # features["depth"] = np.min(semantic_only_depth_feat)
            features["depth"] = depth_feat[obs_3d[:, 3] != 0]
        if "hsv" in self.features:
            rgba = rgba_feat[center_row_col, center_row_col]
            hsv = rgb2hsv(rgba[:3])
            features["hsv"] = hsv
        if any("curvature" in feat for feat in self.features):
            k1, k2, dir1, dir2 = get_curvature_at_point(obs_3d, center_id, point_normal)
        if "principal_curvatures" in self.features:
            features["principal_curvatures"] = [k1, k2]

        if "curvature_directions" in self.features:
            features["curvature_directions"] = np.array([dir1, dir2]).flatten()

        if "gaussian_curvature" in self.features:
            features["gaussian_curvature"] = k1 * k2

        if "mean_curvature" in self.features:
            features["mean_curvature"] = (k1 + k2) / 2

        if "gaussian_curvature_sc" in self.features:
            gc = k1 * k2
            gc_scaled_clipped = scale_clip(gc, 4096)
            features["gaussian_curvature_sc"] = gc_scaled_clipped

        if "mean_curvature_sc" in self.features:
            mc = (k1 + k2) / 2
            mc_scaled_clipped = scale_clip(mc, 256)
            features["mean_curvature_sc"] = mc_scaled_clipped

        obs_3d_center = obs_3d[center_id]
        x, y, z, on_obj = obs_3d_center
        if "on_object" in self.features:
            features["on_object"] = on_obj

        if "object_coverage" in self.features:
            features["object_coverage"] = sum(obs_3d[:, 3]) / len(
                obs_3d[:, 3]
            )  # modified for touch sensor

        # Sensor module returns features at locations.
        patch_observation = {
            "location": [x, y, z],
            "features": features,
        }
        # Save raw observations and state for logging.
        if not self.is_exploring:
            self.processed_obs.append(
                patch_observation
            )  # consider changing to self.provessed_obs = patch_observation
            self.states.append(self.state)

        return patch_observation
