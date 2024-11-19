# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from pose_estimation_models import IterativeClosestPoint


class IterativeClosestPointStoreSources(IterativeClosestPoint):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.source_point_clouds = []

    def step(self):

        super().step()
        self.source_point_clouds.append(self.src)
