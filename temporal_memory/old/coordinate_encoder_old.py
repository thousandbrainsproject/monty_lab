# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import hashlib
import itertools

import numpy as np
import torch


def tensor_to_string(tensor):
    """
    returns a string from a tensor.
    """

    s = [" " for _ in range(len(tensor))]

    for i in range(len(s)):
        if tensor[i].item() == 1:
            s[i] = "1"
        else:
            s[i] = "0"

    return "".join(s)


class CoordinateEncoder:
    """
    Given a coordinate in an N-dimensional space, and a radius around
    that coordinate, the Coordinate Encoder returns an SDR representation
    of that position.
    The Coordinate Encoder uses an N-dimensional integer coordinate space.
    For example, a valid coordinate in this space is (150, -49, 58), whereas
    an invalid coordinate would be (55.4, -5, 85.8475).
    It uses the following algorithm:

    1.  Find all the coordinates around the input coordinate, within the
        specified radius.
    2.  For each coordinate, use a uniform hash function to
        deterministically map it to a real number between 0 and 1. This is the
        "order" of the coordinate.
    3.  Of these coordinates, pick the top W by order, where W is the
        number of active bits desired in the SDR.
    4.  For each of these W coordinates, use a uniform hash function to
        deterministically map it to one of the bits in the SDR. Make this bit
        active.
    5.  This results in a final SDR with exactly W bits active (barring chance hash
        collisions).
    """

    def __init__(self, w=21, n=1000):
        if (w <= 0) or (w % 2 == 0):
            raise ValueError("w must be an odd positive integer")

        if (n <= 6 * w) or (not isinstance(n, int)):
            raise ValueError(
                "n must be an integer strictly greater than 6*w."
                "For good results, use n > 11*w."
            )

        self.w = w
        self.n = n

    def get_width(self):
        """
        return output width
        """

        return self.n

    def encode(self, coordinates, radius):
        assert isinstance(coordinates, torch.Tensor) or isinstance(
            coordinates, np.array
        )
        assert isinstance(radius, int)

        if not isinstance(coordinates, np.array):
            coordinates = np.array(coordinates)

        def bit_fn(x):
            return self.bit_for_coordinate(x)

        sdr = np.zeros((coordinates.shape[0], self.get_width()), dtype=np.uint8)

        for i in range(coordinates.shape[0]):
            coordinate = coordinates[i, :]

            winners = self.top_w_coordinates(self.neighbors(coordinate, radius))

            indices = [bit_fn(coord) for coord in winners]

            sdr[i, indices] = 1

        return sdr

    def neighbors(self, coordinate, radius):
        """
        returns coordinates around given coordinate, within given radius. includes
        given coordinate.
        """

        ranges = (np.arange(n - radius, n + radius + 1) for n in coordinate.tolist())

        return np.array(list(itertools.product(*ranges)))

    def top_w_coordinates(self, coordinates):
        """
        returns the top w coordinates by order.
        """

        orders = np.array([self.order_for_coordinates(c) for c in coordinates])

        indices = np.argsort(orders)[-self.w :]

        return coordinates[indices]

    def order_for_coordinates(self, coordinate):
        """
        returns the order (float in the interval [0, 1) = order) for a coordinate.
        """

        generator = np.random.default_rng(self.hash_coordinate(coordinate))

        return generator.random()

    def hash_coordinate(self, coordinate):
        """
        hash a coordinate to a 64 bit integer.
        """

        return int(
            int(hashlib.md5(coordinate.numpy().tobytes()).hexdigest(), 16) % (2 ** 64)
        )

    def bit_for_coordinate(self, coordinate):
        """
        maps the coordinate to a bit in the SDR.
        """

        rng = np.random.default_rng(self.hash_coordinate(coordinate))

        return rng.integers(0, self.get_width(), size=1)

    def __str__(self):
        string = "CoordinateEncoder:"
        string += "\n   w:  {w}".format(w=self.w)
        string += "\n   n:  {n}".format(n=self.get_width())

        return string

    def __repr__(self):
        return self.__str__()
