# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import torch


class ObjectsDataset(torch.utils.data.Dataset):
    """
    *This creates a dataset of indicies*

    * These indices are chunked and every chunk of indices
    becomes the data batch for a single iteration.
    * If batch_size is set to 0 in the configs, a single iteration
    will apply gradient step over the all the objects.
    """

    def __init__(self, batch_size=10):
        super().__init__()
        self.num_objects = 0
        self.batch_size = batch_size

        # invalid batch size because we need two or more objects
        # to calculate similarity
        if self.batch_size == 1:
            self.batch_size = 0

        self.refresh_batch_size = True if self.batch_size == 0 else False

    def get_batches(self):
        """
        Creates batches to be iterated over during data loading
        """
        if self.refresh_batch_size:
            self.batch_size = self.num_objects

        self.batches = torch.randperm(self.num_objects).chunk(
            int(torch.ceil(torch.tensor(self.num_objects) / self.batch_size))
        )

        if len(self.batches[-1]) == 1:
            self.batches = self.batches[:-1]

    def add_objects(self, new_objects):
        """
        Adds new objects and recreates the chuncks
        """
        self.num_objects += new_objects
        self.get_batches()
        return self.num_objects

    def __getitem__(self, index):
        """
        retrieves data from batches by the index of the data iterator
        """
        if not self.num_objects:
            return
        return self.batches[index]

    def __len__(self):
        return len(self.batches)
