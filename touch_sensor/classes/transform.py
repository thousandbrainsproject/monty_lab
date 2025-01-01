# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
import quaternion as qt


class DepthTo3DLocationsTouch:
    """
    Transform semantic and depth observations from camera coordinate (2D) into
    agent (or world) coordinate (3D). This transform will add the transformed
    results as a new observation called "semantic_3d" which will contain the 3d
    coordinates relative to the agent (or world) with the semantic ID and 3D
    location of every object observed::

        "semantic_3d" : [
        #    x-pos      , y-pos     , z-pos      , semantic_id
            [-0.06000001, 1.56666668, -0.30000007, 25.],
            [ 0.06000001, 1.56666668, -0.30000007, 25.],
            [-0.06000001, 1.43333332, -0.30000007, 25.],
            [ 0.06000001, 1.43333332, -0.30000007, 25.]])
        ]

    :param agent_id: Agent ID to get observations from
    :param resolution: Camera resolution (H, W)
    :param zoom: Camera zoom factor. Defaul 1.0 (no zoom)
    :param hfov: Camera HFOV, default 90 degrees
    :param semantic_sensor: Semantic sensor id. Default "semantic"
    :param depth_sensor: Depth sensor id. Default "depth"
    :param world_coord: Whether to return 3D locations in world coordinates.
                        If enabled, then :meth:`__call__` must be called with
                        the agent and sensor states in addition to observations.
                        Default False. Return coordinated relative to the agent
    :param get_all_points: Whether to return all 3D coordinates or only the ones
                        that land on an object.

    .. warning:: This transformation is only valid for pinhole cameras

    And add depth clipping to simulate touch sensation
    """

    def __init__(
        self,
        agent_id,
        sensor_ids,
        resolutions,
        zooms=1.0,
        hfov=90.0,
        world_coord=False,
        get_all_points=False,
        clip_value=0.005,
    ):
        self.inv_k = []
        self.h, self.w = [], []

        if isinstance(zooms, (int, float)):
            zooms = [zooms] * len(sensor_ids)

        if isinstance(hfov, (int, float)):
            hfov = [hfov] * len(sensor_ids)

        for i, zoom in enumerate(zooms):
            # Pinhole camera, focal length fx = fy
            hfov[i] = float(hfov[i] * np.pi / 180.0)

            fx = np.tan(hfov[i] / 2.0) / zoom
            fy = fx

            # Adjust fy for aspect ratio
            self.h.append(resolutions[i][0])
            self.w.append(resolutions[i][1])
            fy = fy * self.h[i] / self.w[i]

            # Intrinsic matrix, K
            # Assuming skew is 0 for pinhole camera and center at (0,0)
            k = np.array(
                [
                    [1.0 / fx, 0.0, 0.0, 0.0],
                    [0.0, 1 / fy, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            # Inverse K
            self.inv_k.append(np.linalg.inv(k))

        self.agent_id = agent_id
        self.sensor_ids = sensor_ids
        self.clip_value = clip_value
        self.world_coord = world_coord
        self.get_all_points = get_all_points

    def clip(self, agent_obs):
        """
        Clip the depth and semantic data that lie beyond a certain depth threshold
        Set the values of 0 (infinite depth) to the clip value
        """
        agent_obs["semantic"][agent_obs["depth"] > self.clip_value] = 0
        agent_obs["depth"][agent_obs["depth"] > self.clip_value] = self.clip_value
        agent_obs["depth"][agent_obs["depth"] == 0] = self.clip_value

    def __call__(self, observations, state=None):
        for i, sensor_id in enumerate(self.sensor_ids):
            agent_obs = observations[self.agent_id][sensor_id]
            if sensor_id == "patch":
                self.clip(agent_obs)
            depth_obs = agent_obs["depth"]
            semantic_obs = agent_obs["semantic"]

            # Approximate true world coordinates
            x, y = np.meshgrid(
                np.linspace(-1, 1, self.w[i]), np.linspace(1, -1, self.h[i])
            )
            x = x.reshape(1, self.h[i], self.w[i])
            y = y.reshape(1, self.h[i], self.w[i])

            # Unproject 2D camera coordinates into 3D coordinates relative to the agent
            depth = depth_obs.reshape(1, self.h[i], self.w[i])
            xyz = np.vstack((x * depth, y * depth, -depth, np.ones(depth.shape)))
            xyz = xyz.reshape(4, -1)
            xyz = np.matmul(self.inv_k[i], xyz)

            if self.world_coord and state is not None:
                # Apply camera transformations to get world coordinates
                agent_state = state[self.agent_id]
                depth_state = agent_state["sensors"][sensor_id + ".depth"]
                agent_rotation = agent_state["rotation"]
                sensor_rotation = depth_state["rotation"]
                # Combine body and sensor rotation
                rotation = agent_rotation * sensor_rotation

                translation = agent_state["position"] + depth_state["position"]
                rotation_matrix = qt.as_rotation_matrix(rotation)
                world_camera = np.eye(4)
                world_camera[0:3, 0:3] = rotation_matrix
                world_camera[0:3, 3] = translation
                xyz = np.matmul(world_camera, xyz)

            # Extract 3D coordinates of detected objects (semantic_id != 0)
            semantic = semantic_obs.reshape(1, -1)
            if self.get_all_points:
                semantic_3d = xyz.transpose(1, 0)
                semantic_3d[:, 3] = semantic[0]
            else:
                detected = semantic.any(axis=0)
                xyz = xyz.transpose(1, 0)
                semantic_3d = xyz[detected]
                semantic_3d[:, 3] = semantic[0, detected]

            # Add transformed observation to existing dict. We don't need to create
            # a deepcopy because we are appending a new observation
            observations[self.agent_id][sensor_id]["semantic_3d"] = semantic_3d
        return observations
