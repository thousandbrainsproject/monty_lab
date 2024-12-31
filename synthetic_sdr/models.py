# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import torch
import torch.nn as nn

from utils import get_dists, get_overlaps


class Model(nn.Module):
    """
    *This model creates embeddings that turn into SDRs with desirable overlaps
    when binarized*

    * The model is initialized with 0 objects and the :py:meth:`~add_objects` can
    be used to add new objects in a streaming manner
    """

    def __init__(self, sdr_length, sdr_on_bits, lr=1e-3):
        r"""
        *Initialization function for the model. Inherits from `nn.Module`*

        :param int sdr_length: Size of the SDRs
        :param int sdr_on_bits: Number of active bits
        :param float lr: Learning rate for optimization

        """
        super().__init__()
        self.sdr_length = sdr_length
        self.sdr_on_bits = sdr_on_bits
        self.lr = lr
        self.obj_reps = nn.Embedding(0, self.sdr_length)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def add_objects(self, new_objects):
        """Adds new objects the the model"""

        old_data = self.obj_reps.weight.data.clone()

        self.obj_reps = nn.Embedding(old_data.shape[0] + new_objects, self.sdr_length)

        self.obj_reps.weight.data[: old_data.shape[0]] = old_data
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def binarize(self, emb):
        """
        Turns dense embeddings to SDRs by binarizing each object representation
        with top-K function, where K here is the sdr_on_bits.

        **Note**: We are only applying top-k on the second axes only. Therefore,
        each object representation is binarized independantly. `emb` is a tensor of
        shape (num_objects, sdr_length), the output mask is of the same shape.
        """

        topk_indices = torch.topk(emb, k=self.sdr_on_bits, dim=1)[1]
        mask = torch.full_like(emb, -1)
        mask.scatter_(1, topk_indices, 1.0)
        return mask, get_overlaps(mask)

    def optimize(self, loss):
        """optimization function. Performs backpropagation and gradient descent"""

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_reps(self, ix, target_overlaps):
        """
        *Main training function. Receives indices from data loader
        and their target overlaps*

        Note that the overlap_error is only used to weight the distance_matrix loss
        for each pair of objects, and gradients do not flow through the sparse overlap
        calculations. This is the reason for detaching the overlap error in the
        loss function. Also note that the binarization function is non-differentiable so
        gradients are not expected to flow back through this route anyway.

        The magnitude of the overlap error controls the strength of moving
        dense representations. Also the sign of the overlap error controls
        whether the representations will be moving towards or away of each other.


        :param Tensor ix: Indices to consider for training
        :param Tensor target_overlap: The overlap values for the target indices
        : returns:
            * (*dict*): Results of training for logging
        """

        reps = self.obj_reps(ix)

        distance_matrix = get_dists(reps)
        mask, emb_overlaps = self.binarize(reps)
        overlap_error = target_overlaps - emb_overlaps
        loss = (distance_matrix * overlap_error.detach()).mean()

        self.optimize(loss)

        overlap_sum = overlap_error.tril().abs().sum()
        overlap_count = (overlap_error.numel() - overlap_error.shape[0]) // 2
        overlap_distance = (overlap_sum / overlap_count) / self.sdr_on_bits

        results = dict(
            overlap_loss=loss,
            overlap_distance=overlap_distance,
        )
        return results

    def infer_rep(self, reps):
        """
        Calls binarization function and turns provided representations
        to SDRs
        """

        emb_bin, _ = self.binarize(reps)
        return emb_bin
