# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np


class DisparitySrcTgt:
    """
    Temp class for just getting metrics for exp before stepping back making cleaner
    """

    def __call__(self, **kwargs):
        """
        target is the target returned by a monty dataloader, ie dict with lotta info
        """

        target = kwargs["label"]
        euler_target = np.array([i.numpy() for i in target["euler_rotation"]])
        params = kwargs["params"][:3]

        return euler_target + params


class DetailedTargetToParams:
    """
    Take detailed target dictionary and unpack into pieces used in other
    OnlinePoseOptimization Experiments
    """

    def __call__(self, target):

        params = np.zeros(6)
        params[:3] = target["euler_rotations"]
