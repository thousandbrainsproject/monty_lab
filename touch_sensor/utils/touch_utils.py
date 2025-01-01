# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import open3d as o3d


def get_point_normal(point_cloud, center_id, sensor_location):
    """Estimate point normal at the center point of a point cloud.

    :param point_cloud: List of 3D locations
    :param center_id: ID of center point in the point cloud
    :param sensor_location: location of sensor. Used to have the point normal
            point towards the sensor.
    :return: Point normal at center_id
    """

    # on_obj = point_cloud[:, 3] > 0
    # adjusted_center_id = sum(on_obj[:center_id])
    # point_cloud[on_obj, :3] to restrict to points on the object
    point_cloud = point_cloud[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=8 * 8)
    )
    pcd.orient_normals_towards_camera_location(camera_location=sensor_location)

    # adjusted_center_id when restricted to points on the object
    point_normal = pcd.normals[center_id]

    return point_normal
