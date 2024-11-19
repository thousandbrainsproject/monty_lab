# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import torch
import torch.nn.functional as F


def quaternion_geodesic_loss(pred_quaternion: torch.Tensor, target_quaternion: torch.Tensor) -> torch.Tensor:
    """Compute the geodesic distance between two quaternions.

    Args:
        pred_quaternion (torch.Tensor): Predicted quaternion of shape (N, 4)
        target_quaternion (torch.Tensor): Target quaternion of shape (N, 4)

    Returns:
        torch.Tensor: Geodesic loss between the quaternions
    """
    # Normalize the quaternions
    pred_norm = F.normalize(pred_quaternion, p=2, dim=-1)
    target_norm = F.normalize(target_quaternion, p=2, dim=-1)

    # Compute the dot product
    dot_product = torch.sum(pred_norm * target_norm, dim=-1)

    # Clamp the dot product to avoid numerical instability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute the geodesic distance (angle)
    angle = 2 * torch.acos(torch.abs(dot_product))

    # Return the mean loss
    return torch.mean(angle)
