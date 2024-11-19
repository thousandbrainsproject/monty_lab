# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R



def get_rotation_error(predicted_quaternion: torch.Tensor, target_quaternion: torch.Tensor):
    """Get the rotation error between two quaternions.

    Args:
        predicted_quaternion (torch.Tensor): Predicted quaternion (batch_size, 4).
        target_quaternion (torch.Tensor): Target quaternion (batch_size, 4).

    Returns:
        torch.Tensor: Rotation error for each sample in the batch.
    """
    device = predicted_quaternion.device

    # Convert to numpy if not already
    if not isinstance(predicted_quaternion, np.ndarray):
        predicted_quaternion = predicted_quaternion.detach().cpu().numpy()
    if not isinstance(target_quaternion, np.ndarray):
        target_quaternion = target_quaternion.detach().cpu().numpy()

    predicted_quaternion = R.from_quat(predicted_quaternion)
    target_quaternion = R.from_quat(target_quaternion)

    difference = predicted_quaternion * target_quaternion.inv()
    errors = difference.magnitude()
    
    return torch.tensor(errors, dtype=torch.float32, device=device)
