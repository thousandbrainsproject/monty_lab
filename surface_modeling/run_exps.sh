#!/bin/zsh

# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

python run.py -e icp_x_rotation,
                 icp_xy_rotation,
                 icp_xyz_rotation,
                 icp_rigid_body_x,
                 icp_rigid_body_xy,
                 icp_rigid_body_xyz
