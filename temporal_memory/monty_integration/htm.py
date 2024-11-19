# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Used to be in frameworks/models/

from copy import deepcopy

import numpy as np
from nupic.bindings.math import Random, SparseMatrixConnections
from nupic.research.frameworks.columns import ApicalTiebreakPairMemoryWrapper


class L6a_2d():  # noqa N801
    """
    A specific version of Layer 6A, i.e. the location layer, which handles movement
    in 2 dimensions (x, y).

    A single layer comprises of multiple modules per dimension, where each module
    contains a set of grid cells that share the same scale and orientation.
    """
    def __init__(
        self,
        num_modules,
        cells_per_axis,
        anchor_input_size,
        cell_coordinate_offsets,
        activation_threshold,
        initial_permanence,
        connected_permanence,
        matching_threshold,
        sample_size,
        permanence_increment,
        permanence_decrement,
        anchoring_method,
        random_location,
        seed,
    ):
        scales = [1.0]
        orientations = [
            np.radians(
                (i * 90 / num_modules) + (90 / num_modules) / 2
            )
            for i in range(num_modules)
        ]

        self.modules_xy = []
        self.modules_yz = []
        self.modules_xz = []
        for n in range(num_modules):
            # from Gu et. al 2018 (Map-like Micro-Organization of Grid Cells in the MEC)
            # successive module have scales that increase by a factor of 1.5
            scale = scales[-1] * 1.5
            scales.append(scale)

            self.modules_xy.append(
                Simplified2DLocationModule(
                    cells_per_axis=cells_per_axis,
                    scale=scale,
                    orientation=orientations[n],
                    anchor_input_size=anchor_input_size,
                    cell_coordinate_offsets=cell_coordinate_offsets,
                    activation_threshold=activation_threshold,
                    initial_permanence=initial_permanence,
                    connected_permanence=connected_permanence,
                    matching_threshold=matching_threshold,
                    sample_size=sample_size,
                    permanence_increment=permanence_increment,
                    permanence_decrement=permanence_decrement,
                    anchoring_method=anchoring_method,
                    seed=seed,
                )
            )

        self.random_location = random_location

    def set_first_location(self):
        if self.random_location:
            self.last_location = np.array(
                [np.random.rand() * 100, np.random.rand() * 100]
            )
        else:
            self.last_location = np.array([50, 50])

    def activate_random_location(self):
        for module in (self.modules_xy):
            module.activate_random_location()

    def get_location_representation(self):
        """
        get full population representation of the location layer.
        """
        active_cells = np.array([], dtype="uint32")

        total_prev_cells = 0

        for module in (self.modules_xy):
            active_cells = np.append(
                active_cells,
                module.get_active_cells() + total_prev_cells
            )

            total_prev_cells += module.get_num_of_cells()

        return active_cells

    def get_learnable_location_representation(self):
        """
        get the cells in the location layer that should be associated with the
        sensory input layer representation. in some models, this is identical to the
        active cells. in others, it's a subset.
        """
        learnable_cells = np.array([], dtype="uint32")

        total_prev_cells = 0

        for module in (self.modules_xy):
            learnable_cells = np.append(
                learnable_cells,
                module.get_learnable_cells() + total_prev_cells
            )

            total_prev_cells += module.get_num_of_cells()

        return learnable_cells

    def get_sensory_associated_location_representation(self):
        """
        get the location cells in the location layer that were driven by the input
        layer (or, during learning, that was associated with this input).
        """
        sensory_associated_cells = np.array([], dtype="uint32")

        total_prev_cells = 0

        for module in (self.modules_xy):
            sensory_associated_cells = np.append(
                sensory_associated_cells,
                module.get_sensory_associated_cells() + total_prev_cells
            )

            total_prev_cells += module.get_num_of_cells()

        return sensory_associated_cells

    def movement_compute(self, next_location, noise_factor=0):
        displacement = (next_location - self.last_location)

        self.last_location = next_location

        # each module only computes displacement along a specific plane (xy, yz, and xz)
        for module in self.modules_xy:
            module.movement_compute(
                displacement=displacement[[0, 1]],
                noise_factor=noise_factor
            )

    def sensory_compute(self, anchor_input, anchor_growth_candidates, learn):
        for module in (self.modules_xy):
            module.sensory_compute(
                anchor_input=anchor_input,
                anchor_growth_candidates=anchor_growth_candidates,
                learn=learn
            )

    def reset(self):
        for module in (self.modules_xy):
            module.reset()


class L6a_3d:  # noqa N801
    """
    A specific version of Layer 6A, i.e. the location layer, which handles movement
    in 3 dimensions (x, y, and z) instead of 2.

    A single layer comprises of multiple modules per dimension, where each module
    contains a set of grid cells that share the same scale and orientation.
    """
    def __init__(
        self,
        num_modules,
        cells_per_axis,
        anchor_input_size,
        cell_coordinate_offsets,
        activation_threshold,
        initial_permanence,
        connected_permanence,
        matching_threshold,
        sample_size,
        permanence_increment,
        permanence_decrement,
        anchoring_method,
        random_location,
        seed,
    ):
        scales = [1.0]
        orientations = [
            np.radians(
                (i * 90 / num_modules) + (90 / num_modules) / 2
            )
            for i in range(num_modules)
        ]

        self.modules_xy = []
        self.modules_yz = []
        self.modules_xz = []
        for n in range(num_modules):
            # from Gu et. al 2018 (Map-like Micro-Organization of Grid Cells in the MEC)
            # successive module have scales that increase by a factor of 1.5
            scale = scales[-1] * 1.5
            scales.append(scale)

            self.modules_xy.append(
                Simplified2DLocationModule(
                    cells_per_axis=cells_per_axis,
                    scale=scale,
                    orientation=orientations[n],
                    anchor_input_size=anchor_input_size,
                    cell_coordinate_offsets=cell_coordinate_offsets,
                    activation_threshold=activation_threshold,
                    initial_permanence=initial_permanence,
                    connected_permanence=connected_permanence,
                    matching_threshold=matching_threshold,
                    sample_size=sample_size,
                    permanence_increment=permanence_increment,
                    permanence_decrement=permanence_decrement,
                    anchoring_method=anchoring_method,
                    seed=seed,
                )
            )

            self.modules_yz.append(
                Simplified2DLocationModule(
                    cells_per_axis=cells_per_axis,
                    scale=scale,
                    orientation=orientations[n],
                    anchor_input_size=anchor_input_size,
                    cell_coordinate_offsets=cell_coordinate_offsets,
                    activation_threshold=activation_threshold,
                    initial_permanence=initial_permanence,
                    connected_permanence=connected_permanence,
                    matching_threshold=matching_threshold,
                    sample_size=sample_size,
                    permanence_increment=permanence_increment,
                    permanence_decrement=permanence_decrement,
                    anchoring_method=anchoring_method,
                    seed=seed,
                )
            )

            self.modules_xz.append(
                Simplified2DLocationModule(
                    cells_per_axis=cells_per_axis,
                    scale=scale,
                    orientation=orientations[n],
                    anchor_input_size=anchor_input_size,
                    cell_coordinate_offsets=cell_coordinate_offsets,
                    activation_threshold=activation_threshold,
                    initial_permanence=initial_permanence,
                    connected_permanence=connected_permanence,
                    matching_threshold=matching_threshold,
                    sample_size=sample_size,
                    permanence_increment=permanence_increment,
                    permanence_decrement=permanence_decrement,
                    anchoring_method=anchoring_method,
                    seed=seed,
                )
            )

        self.random_location = random_location

    def set_first_location(self):
        if self.random_location:
            self.last_location = np.array(
                [np.random.rand() * 100, np.random.rand() * 100, np.random.rand() * 100]
            )
        else:
            self.last_location = np.array([50, 50, 50])

    def activate_random_location(self):
        for module in (self.modules_xy + self.modules_yz + self.modules_xz):
            module.activate_random_location()

    def get_location_representation(self):
        """
        get full population representation of the location layer.
        """
        active_cells = np.array([], dtype="uint32")

        total_prev_cells = 0

        for module in (self.modules_xy + self.modules_yz + self.modules_xz):
            active_cells = np.append(
                active_cells,
                module.get_active_cells() + total_prev_cells
            )

            total_prev_cells += module.get_num_of_cells()

        return active_cells

    def get_learnable_location_representation(self):
        """
        get the cells in the location layer that should be associated with the
        sensory input layer representation. in some models, this is identical to the
        active cells. in others, it's a subset.
        """
        learnable_cells = np.array([], dtype="uint32")

        total_prev_cells = 0

        for module in (self.modules_xy + self.modules_yz + self.modules_xz):
            learnable_cells = np.append(
                learnable_cells,
                module.get_learnable_cells() + total_prev_cells
            )

            total_prev_cells += module.get_num_of_cells()

        return learnable_cells

    def get_sensory_associated_location_representation(self):
        """
        get the location cells in the location layer that were driven by the input
        layer (or, during learning, that was associated with this input).
        """
        sensory_associated_cells = np.array([], dtype="uint32")

        total_prev_cells = 0

        for module in (self.modules_xy + self.modules_yz + self.modules_xz):
            sensory_associated_cells = np.append(
                sensory_associated_cells,
                module.get_sensory_associated_cells() + total_prev_cells
            )

            total_prev_cells += module.get_num_of_cells()

        return sensory_associated_cells

    def movement_compute(self, next_location, noise_factor=0):
        displacement = (next_location - self.last_location)

        self.last_location = next_location

        # each module only computes displacement along a specific plane (xy, yz, and xz)
        for module in self.modules_xy:
            module.movement_compute(
                displacement=displacement[[0, 1]],
                noise_factor=noise_factor
            )

        for module in self.modules_yz:
            module.movement_compute(
                displacement=displacement[[1, 2]],
                noise_factor=noise_factor
            )

        for module in self.modules_xz:
            module.movement_compute(
                displacement=displacement[[0, 2]],
                noise_factor=noise_factor
            )

    def sensory_compute(self, anchor_input, anchor_growth_candidates, learn):
        for module in (self.modules_xy + self.modules_yz + self.modules_xz):
            module.sensory_compute(
                anchor_input=anchor_input,
                anchor_growth_candidates=anchor_growth_candidates,
                learn=learn
            )

    def reset(self):
        for module in (self.modules_xy + self.modules_yz + self.modules_xz):
            module.reset()


class L4:
    """
    Classic Layer 4 Temporal Memory layer that interacts with the location layer L6a.
    """
    def __init__(
        self,
        proximal_n,
        proximal_w,
        basal_n,
        basal_w,
        apical_n,
        apical_w,
        cells_per_column,
        activation_threshold,
        reduced_basal_threshold,
        initial_permanence,
        connected_permanence,
        matching_threshold,
        sample_size,
        permanence_increment,
        permanence_decrement,
        seed
    ):
        self.tm = ApicalTiebreakPairMemoryWrapper(
            proximal_n=proximal_n,
            proximal_w=proximal_w,
            basal_n=basal_n,
            basal_w=basal_w,
            apical_n=apical_n,
            apical_w=apical_w,
            cells_per_column=cells_per_column,
            activation_threshold=activation_threshold,
            reduced_basal_threshold=reduced_basal_threshold,
            initial_permanence=initial_permanence,
            connected_permanence=connected_permanence,
            matching_threshold=matching_threshold,
            sample_size=sample_size,
            permanence_increment=permanence_increment,
            permanence_decrement=permanence_decrement,
            seed=seed
        )

    def sensory_compute(
        self,
        active_columns,
        basal_input,
        basal_growth_candidates,
        learn
    ):
        self.tm.compute(
            active_columns=active_columns,
            basal_input=basal_input,
            basal_growth_candidates=basal_growth_candidates,
            learn=learn
        )

    def get_winner_cells(self):
        return self.tm.get_winner_cells()

    def get_predicted_active_cells(self):
        return self.tm.get_predicted_active_cells()

    def get_active_cells(self):
        return self.tm.getActiveCells()

    def reset(self):
        self.tm.reset()


class ThresholdedGaussian2DLocationModule():
    """
    a model of a grid cell module. the module has one or more Gaussian activity bumps
    that move as the population receives motor input. when two bumps are near each
    other, the intermediate cells have higher firing rates than they would with a single
    bump. the cells with firing rates above a certain threshold are considered "active".

    doesn't model path integration. when the network receives a motor command, it shifts
    its bump. we track each bump as floating point coordinates and we shift the bumps
    with movement.
    this model isn't attempting to explain how path integration works.
    it's attempting to show how a population of cells that can path integrate are useful
    in a larger network.

    cells are distributed uniformly through the rhombus -- packed in the optimal
    hexagonal arrangement. during learning, the cell nearest to the current phase
    is associated with the sensed feature.

    cells_per_axis, active_firing_rate, and bump_sigma all must be chosen such that:
      - one cell fires at each location
      - inference accounts for uncertainty in the learned locations (must use a large
        enough set of active cells)
    """
    def __init__(
        self,
        cells_per_axis,
        scale,
        orientation,
        anchor_input_size,
        active_firing_rate,
        bump_sigma,
        activation_threshold=10,
        initial_permanence=0.21,
        connected_permanence=0.50,
        matching_threshold=10,
        sample_size=20,
        permanence_increment=0.1,
        permanence_decrement=0.0,
        max_synapses_per_segment=-1,
        bump_overlap_method="probabilistic",
        seed=42
    ):
        """
        @param cells_per_axis (int)
        Determines the number of cells.
        Determines how space is divided between the cells.

        @param scale (float)
        Determines the amount of world space covered by all the cells combined.
        This defines the "scale" of the module.

        @param orientation (float)
        Rotation of the map, measured in radians.

        @param anchor_input_size (int)
        Number of input bits in the anchor input.

        @param active_firing_rate (float)
        Between 0.0 and 1.0 -- a cell is considered active if its firing rate is at
        least this value.

        @param bump_sigma (float)
        Specifies the diameter of the gaussian bump, in units of "rhombus edges".
        A single edge of the rhombus has length 1. bump_sigma is typically less than 1.
        Often use 0.18172 as an estimate for the sigma of a rat's entorhinal bump.

        @param bump_overlap_method ("probabilistic" or "sum")
        Specifies the firing rate of a cell when it's part of two bumps.
        """

        self.cells_per_axis = cells_per_axis

        self.scale = scale
        self.orientation = orientation

        # matrix that converts world displacement into phase displacement
        self.world_to_phase = np.linalg.inv(
            scale * np.array(
                [
                    [np.cos(orientation), np.cos(orientation + np.radians(60.))],
                    [np.sin(orientation), np.sin(orientation + np.radians(60.))]
                ]
            )
        )

        # matrix that converts phase displacement to world displacement
        self.phase_to_world = np.array([
            [np.cos(np.radians(0.)), np.cos(np.radians(60.))],
            [np.sin(np.radians(0.)), np.sin(np.radians(60.))]
        ])

        # spatial phase is measured as two numbers in the range [0.0, 1.0) (for x and y)
        self.bump_phases = np.empty((2, 0), dtype="float")
        self.cells_for_active_phases = np.empty(0, dtype="int")
        self.phase_displacement = np.empty((0, 2), dtype="float")

        self.active_cells = np.empty(0, dtype="int")

        # learning cells == "winner" cells
        self.learning_cells = np.empty(0, dtype="int")

        # cells that were activated by sensory input in an learning / inference timestep
        self.sensory_associated_cells = np.empty(0, dtype="int")

        self.active_segments = np.empty(0, dtype="uint32")

        self.connections = SparseMatrixConnections(
            self.cells_per_axis * self.cells_per_axis,
            anchor_input_size
        )

        self.initial_permanence = initial_permanence
        self.connected_permanence = connected_permanence
        self.matching_threshold = matching_threshold
        self.sample_size = sample_size
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.activation_threshold = activation_threshold
        self.max_synapses_per_segment = max_synapses_per_segment

        self.bump_sigma = bump_sigma
        self.active_firing_rate = active_firing_rate
        self.bump_overlap_method = bump_overlap_method

        cell_phases_axis = np.linspace(0, 1, self.cells_per_axis, endpoint=False)

        # discretization of spatial phases in x and y dimensions
        self.cell_phases = np.array([
            np.repeat(cell_phases_axis, self.cells_per_axis),
            np.tile(cell_phases_axis, self.cells_per_axis)
        ])

        # shift the cells so they're visually/intuitively arranged in a rhombus
        self.cell_phases += [[0.5 / self.cells_per_axis], [0.5 / self.cells_per_axis]]

        self.nupic_rng = Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def reset(self):
        """
        clear the active cells.
        """

        self.bump_phases = np.empty((2, 0), dtype="float")
        self.phase_displacement = np.empty((0, 2), dtype="float")
        self.cells_for_active_phases = np.empty(0, dtype="int")
        self.active_cells = np.empty(0, dtype="int")
        self.learning_cells = np.empty(0, dtype="int")
        self.sensory_associated_cells = np.empty(0, dtype="int")

    def compute_active_cells(self):
        """
        compute which cells are active.
        """

        # for each cell, compute the phase displacement from each bump:
        #   - create an array of matrices, one per cell
        #   - each column in a matrix corresponds to the phase displacement from the
        #     bump to the cell
        cell_bump_positive_phase_displacement = np.mod(
            self.cell_phases.T[:, :, np.newaxis] - self.bump_phases,
            1.0
        )

        # for each cell / bump pair, consider phase displacement vectors reaching
        # that cell from that bump by moving up/right, up/left, down/right, or down/left
        #
        # create a 2D array of matrices arranged by cell then direction. each column
        # in a matrix corresponds to a phase displacement from the bump to the cell in
        # a particular direction
        cell_direction_bump_phase_displacement = (
            cell_bump_positive_phase_displacement[:, np.newaxis, :, :]
            - np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ])[:, :, np.newaxis]
        )

        # convert displacement in phase to a displacement in the world, with scale
        # normalized out. two vectors with the same phase distance will typically have
        # different world distances unless they are parallel.
        cell_direction_bump_world_displacement = np.matmul(
            self.phase_to_world,
            cell_direction_bump_phase_displacement
        )

        # measure length of each displacement vector.
        # create a 3D array of distances, organized by cell direction, then bump.
        cell_direction_bump_distance = np.linalg.norm(
            cell_direction_bump_world_displacement,
            axis=-2
        )

        # choose the shortest distance from each cell to each bump. create a 2D
        # array of distances, organized by cell then bump.
        cell_bump_distance = np.amin(cell_direction_bump_distance, axis=1)

        # compute the Gaussian of each of these distances
        cell_excitations_from_bumps = gaussian(self.bump_sigma, cell_bump_distance)

        # combine bumps: create an array of firing rates, organized by cell.
        if self.bump_overlap_method == "probabilistic":
            # a bump is a probability distribution. each cell's firing rate encodes
            # its relative probability that it's the correct location.
            #
            # e.g., for bump A, P(A = cell x) is encoded by cell x's firing rate.
            #
            # union of bumps A, B, etc. is not a probability distribution but a
            # set of independent events.
            #
            # when multiple bumps overlap, cell's firing rate should encode its relative
            # probability that it's correct in *any* bump:
            #       P( (A = cell x) or (B = cell x) or ... )
            # ==    1 - P( (A != cell x) and (B != cell x) and ...)
            # ==    1 - P(A != cell x) * P(B != cell x) * ...
            #
            # conclusion: as more and more bumps overlap, a cell's firing rate increases
            #             but not as quickly as it would with a sum.
            cell_excitations = 1. - np.prod(1. - cell_excitations_from_bumps, axis=1)
        elif self.bump_overlap_method == "sum":
            # sum the firing rates.
            #
            # this means that a union of bumps A, B, etc. is a set of equally probable
            # events of which only one is true.
            cell_excitations = np.sum(cell_excitations_from_bumps, axis=1)
        else:
            raise ValueError(
                "Unrecognized bump overlap strategy", self.bump_overlap_method
            )

        self.active_cells = np.where(cell_excitations >= self.active_firing_rate)[0]
        self.learning_cells = np.where(cell_excitations == cell_excitations.max())[0]

    def activate_random_location(self):
        """
        set the location to a random point.
        """

        self.bump_phases = np.array([self.np_rng.random(2)]).T
        self.compute_active_cells()

    def movement_compute(self, displacement, noise_factor=0):
        """
        shift the current active cells by a vector.

        @param displacement (pair of floats)
        A translation vector [di, dj].
        """

        if noise_factor != 0:
            displacement = deepcopy(displacement)
            x_noise = self.np_rng.normal(0, noise_factor)
            y_noise = self.np_rng.normal(0, noise_factor)

            displacement[0] += x_noise
            displacement[1] += y_noise

        # calculate delta in the module's coordinates
        phase_displacement = np.matmul(self.world_to_phase, displacement)

        # shift the active coordinates
        np.add(
            self.bump_phases,
            phase_displacement[:, np.newaxis],
            out=self.bump_phases
        )
        np.round(self.bump_phases, decimals=9, out=self.bump_phases)
        np.mod(self.bump_phases, 1.0, out=self.bump_phases)

        self.compute_active_cells()
        self.phase_displacement = phase_displacement

    def sensory_compute_inference_mode(self, anchor_input):
        """
        infer the location from sensory input. activate any cells with enough active
        synapses to this sensory input. deactivate all other cells.

        @param anchor_input (np.array)
        A sensory input. This will often come from a feature-location pair layer.
        """

        if len(anchor_input) == 0:
            return

        overlaps = self.connections.computeActivity(
            anchor_input,
            self.connected_permanence
        )
        active_segments = np.where(overlaps >= self.activation_threshold)[0]

        sensory_supported_cells = np.unique(
            self.connections.mapSegmentsToCells(active_segments)
        )

        self.bump_phases = self.cell_phases[:, sensory_supported_cells]
        self.compute_active_cells()

        self.active_segments = active_segments
        self.sensory_associated_cells = sensory_supported_cells

    def sensory_compute_learning_mode(self, anchor_input):
        """
        associate this location with a sensory input. subsequently, anchor input will
        activate the current location during sensory_compute_inference_mode().

        @param anchor_input (np.array)
        A sensory inut. this will often come from a feature-location pair layer.
        """

        # cells with an active segment: reinforce the segment
        overlaps = self.connections.computeActivity(
            anchor_input,
            self.connected_permanence
        )
        active_segments = np.where(overlaps >= self.activation_threshold)[0]

        cells_for_active_segments = self.connections.mapSegmentsToCells(active_segments)
        learning_active_segments = active_segments[
            np.isin(cells_for_active_segments, self.learning_cells)
        ]

        # remaining cells with a matching segment: reinforce the best matching segment
        potential_overlaps = self.connections.computeActivity(anchor_input)
        matching_segments = np.where(potential_overlaps >= self.matching_threshold)[0]

        remaining_cells = np.setdiff1d(self.learning_cells, cells_for_active_segments)

        candidate_segments = self.connections.filterSegmentsByCell(
            matching_segments,
            remaining_cells
        )
        cells_for_candidate_segments = self.connections.mapSegmentsToCells(
            candidate_segments
        )
        candidate_segments = candidate_segments[
            np.isin(cells_for_candidate_segments, remaining_cells)
        ]

        learning_matching_segments = candidate_segments[
            argmax_multi(
                potential_overlaps[candidate_segments],
                cells_for_candidate_segments
            )
        ]

        # learn on both chosen active segments and matching segments:
        #   - adjust synapse permanences
        #   - grow new synapses
        #   - grow new segments
        for learning_segments in (learning_active_segments, learning_matching_segments):
            self.connections.adjustSynapses(
                learning_segments,
                anchor_input,
                self.permanence_increment,
                -self.permanence_decrement
            )

            # grow new synapses. calculate `max_new` number of synapses to grow per
            # segment. `max_new` might be a number or a list of numbers
            if self.sample_size == -1:
                max_new = len(anchor_input)
            else:
                max_new = self.sample_size - potential_overlaps[learning_segments]

            if self.max_synapses_per_segment != -1:
                synapse_counts = self.connections.mapSegmentsToSynapseCounts(
                    learning_segments
                )

                num_synapses_until_max = self.max_synapses_per_segment - synapse_counts

                max_new = np.where(
                    max_new <= num_synapses_until_max,
                    max_new,
                    num_synapses_until_max
                )

            self.connections.growSynapsesToSample(
                learning_segments,
                anchor_input,
                max_new,
                self.initial_permanence,
                self.nupic_rng
            )

        # remaining calls without a matching segment: grow one
        new_segment_cells = np.setdiff1d(remaining_cells, cells_for_candidate_segments)

        num_new_synapses = len(anchor_input)

        if self.sample_size != -1:
            num_new_synapses = min(num_new_synapses, self.sample_size)

        if self.max_synapses_per_segment != -1:
            num_new_synapses = min(num_new_synapses, self.max_synapses_per_segment)

        new_segments = self.connections.createSegments(new_segment_cells)

        self.connections.growSynapsesToSample(
            new_segments,
            anchor_input,
            num_new_synapses,
            self.initial_permanence,
            self.nupic_rng
        )

        self.active_segments = active_segments
        self.sensory_associated_cells = self.learning_cells

    def sensory_compute(self, anchor_input, anchor_growth_candidates, learn):
        if learn:
            self.sensory_compute_learning_mode(anchor_growth_candidates)
        else:
            self.sensory_compute_inference_mode(anchor_input)

    def get_active_cells(self):
        return self.active_cells

    def get_learnable_cells(self):
        return self.learning_cells

    def get_sensory_associated_cells(self):
        return self.sensory_associated_cells

    def get_num_of_cells(self):
        return self.cells_per_axis * self.cells_per_axis


class Simplified2DLocationModule():
    """
    a model of a location module. it's similar to a grid cell module, but it uses
    squares rather than triangles.

    the cells are arranged into an m*n rectangle which is tiled onto 2D space.
    each cell represents a small rectangle in each tile.

    +------+------+------++------+------+------+
    | Cell | Cell | Cell || Cell | Cell | Cell |
    |  #1  |  #2  |  #3  ||  #1  |  #2  |  #3  |
    |      |      |      ||      |      |      |
    +--------------------++--------------------+
    | Cell | Cell | Cell || Cell | Cell | Cell |
    |  #4  |  #5  |  #6  ||  #4  |  #5  |  #6  |
    |      |      |      ||      |      |      |
    +--------------------++--------------------+
    | Cell | Cell | Cell || Cell | Cell | Cell |
    |  #7  |  #8  |  #9  ||  #7  |  #8  |  #9  |
    |      |      |      ||      |      |      |
    +------+------+------++------+------+------+


    path integration works *somehow*. this model doesn't attempt to propose how
    "path integration" works. it rather shows how locations are anchored to sensory
    cues.

    the model receives a "delta location" vector and
    shifts the active cells accordingly. the model stores intermediate coordinates of
    active cells. whenever sensory cues activate a cell, the model adds this cell to
    the list of coordinates being shifted.

    orientation is specified in a counter-clockwise manner:
    when orientation is 0 degrees, displacement is [di, dj]: move di cells "down" and
    dj cells "right".
    when orientation is 90 degrees, the model is equivalent to a Cartesian system
    with the origin on the bottom left. displacement here is [dx, dy].

    usage:
        - whenever the sensor moves, call movement_compute()
        - whenever the sensor senses something, call sensory_compute()

    `anchor_input` is a feature-location pair SDR.

    specify how points are tracked via `anchoring_method`:
        - "corners": designed for noise-tolerance. will activate all cells that are
                     possible outcomes of path integration
        - "narrowing": designed to narrow down uncertainty of initial locations of
                       sensory stimuli
        - "discrete": network operates in a fully discrete space, where uncertainty
                      is impossible as long as movements are integers
    """
    def __init__(
        self,
        cells_per_axis,
        scale,
        orientation,
        anchor_input_size,
        cell_coordinate_offsets=(0.5,),
        activation_threshold=10,
        initial_permanence=0.21,
        connected_permanence=0.50,
        matching_threshold=10,
        sample_size=20,
        permanence_increment=0.1,
        permanence_decrement=0.0,
        max_synapses_per_segment=-1,
        anchoring_method="narrowing",
        rotation_matrix=None,
        seed=42
    ):
        """
        @param cells_per_axis (int)
        Determines the number of cells.
        Determines how space is divided between the cells.

        @param scale (float)
        Determines the amount of world space covered by all the cells combined.
        This defines the "scale" of the module.

        @param orientation (float)
        Rotation of the map, measured in radians. Specified in a counter-clockwise
        manner.

        @param anchor_input_size (int)
        Number of input bits in the anchor input.

        @param cell_coordinate_offsets (list of floats)
        - Each float much be between 0.0 and 1.0.
        - Every time a cell is activated by anchor input, this class adds a "phase"
          which is shifted in subsequent motions. By default, this phase is
          placed at the center of the cell.
        - This parameter allows you to control where the point is placed and whether
          multiple are placed.
        - e.g.: with value [0.2, 0.8] when cell [2, 3] is activated, it will place
          4 phases corresponding to the following points in cell coordinates:
          [2.2, 3.2], [2.2, 3.8], [2.8, 3.2], [2.8, 3.8]
        """

        self.cells_per_axis = cells_per_axis
        self.scale = scale
        self.anchoring_method = anchoring_method

        self.cell_dimensions = np.array(
            [self.cells_per_axis, self.cells_per_axis],
            dtype="int"
        )

        self.module_map_dimensions = np.array([self.scale, self.scale], dtype="float")
        self.phases_per_unit_distance = 1.0 / self.module_map_dimensions

        if rotation_matrix is None:
            self.orientation = orientation
            self.rotation_matrix = np.array([
                [np.cos(orientation), -np.sin(orientation)],
                [np.sin(orientation), np.cos(orientation)]
            ])

            if self.anchoring_method == "discrete":
                # need to convert matrix to have integer values
                nonzeros = self.rotation_matrix[
                    np.where(np.abs(self.rotation_matrix) > 0)
                ]

                smallest_value = np.amin(nonzeros)
                self.rotation_matrix = np.ceil(self.rotation_matrix / smallest_value)
        else:
            self.rotation_matrix = rotation_matrix

        self.cell_coordinate_offsets = cell_coordinate_offsets

        # spatial phase is measured as two numbers in the range [0.0, 1.0) (for x and y)
        self.active_phases = np.empty((2, 0), dtype="float")
        self.cells_for_active_phases = np.empty(0, dtype="int")
        self.phase_displacement = np.empty((0, 2), dtype="float")

        self.active_cells = np.empty(0, dtype="int")

        # cells that were activated by sensory input in an learning / inference timestep
        self.sensory_associated_cells = np.empty(0, dtype="int")

        self.active_segments = np.empty(0, dtype="uint32")

        self.connections = SparseMatrixConnections(
            np.prod(self.cell_dimensions),
            anchor_input_size
        )

        self.initial_permanence = initial_permanence
        self.connected_permanence = connected_permanence
        self.matching_threshold = matching_threshold
        self.sample_size = sample_size
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.activation_threshold = activation_threshold
        self.max_synapses_per_segment = max_synapses_per_segment

        self.nupic_rng = Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def reset(self):
        """
        clear the active cells.
        """

        self.active_phases = np.empty((2, 0), dtype="float")
        self.phase_displacement = np.empty((0, 2), dtype="float")
        self.cells_for_active_phases = np.empty(0, dtype="int")
        self.active_cells = np.empty(0, dtype="int")
        self.sensory_associated_cells = np.empty(0, dtype="int")

    def compute_active_cells(self):
        """
        compute which cells are active.
        """

        if self.active_phases.size:
            # round each coordinate to the nearest cell
            active_cell_coordinates = np.floor(
                self.active_phases * self.cell_dimensions
            ).astype("int")

            # convert coordinates to cell numbers
            self.cells_for_active_phases = np.ravel_multi_index(
                active_cell_coordinates.T,
                self.cell_dimensions
            )

            self.active_cells = np.unique(self.cells_for_active_phases)

    def activate_random_location(self):
        """
        set the location to a random point.
        """

        self.active_phases = np.array([self.np_rng.random(2)])

        if self.anchoring_method == "discrete":
            # need to place the phase in the middle of a cell
            self.active_phases = np.floor(
                self.active_phases * self.cell_dimensions
            ) / self.cell_dimensions

        self.compute_active_cells()

    def movement_compute(self, displacement, noise_factor=0):
        """
        shift the current active cells by a vector.

        @param displacement (pair of floats)
        A translation vector [di, dj].
        """

        if noise_factor != 0:
            displacement = deepcopy(displacement)
            axis1_noise = self.np_rng.normal(0, noise_factor)
            axis2_noise = self.np_rng.normal(0, noise_factor)

            displacement[0] += axis1_noise
            displacement[1] += axis2_noise

        # calculate delta in the module's coordinates
        phase_displacement = (
            np.matmul(self.rotation_matrix, displacement)
            * self.phases_per_unit_distance
        )

        # shift the active coordinates
        if self.active_phases.size:
            np.add(
                self.active_phases,
                phase_displacement,
                out=self.active_phases
            )
            np.round(self.active_phases, decimals=9, out=self.active_phases)
            np.mod(self.active_phases, 1.0, out=self.active_phases)

        self.compute_active_cells()
        self.phase_displacement = phase_displacement

    def sensory_compute_inference_mode(self, anchor_input):
        """
        infer the location from sensory input. activate any cells with enough active
        synapses to this sensory input. deactivate all other cells.

        @param anchor_input (np.array)
        A sensory input. This will often come from a feature-location pair layer.
        """

        if len(anchor_input) == 0:
            return

        overlaps = self.connections.computeActivity(
            anchor_input,
            self.connected_permanence
        )

        active_segments = np.where(overlaps >= self.activation_threshold)[0]

        sensory_supported_cells = np.unique(
            self.connections.mapSegmentsToCells(active_segments)
        )

        # remove previously active cells that don't have sensory support in the current
        # timestep
        inactivated = np.setdiff1d(self.active_cells, sensory_supported_cells)
        inactivated_indices = np.isin(
            self.cells_for_active_phases,
            inactivated
        ).nonzero()[0]

        if inactivated_indices.size > 0:
            self.active_phases = np.delete(
                self.active_phases,
                inactivated_indices,
                axis=0
            )

        # find activated cells in the current timestep
        activated = np.setdiff1d(sensory_supported_cells, self.get_active_cells)

        # find centers of point clouds
        if self.anchoring_method == "corners":
            activated_coords_base = np.transpose(
                np.unravel_index(sensory_supported_cells, self.cell_dimensions)
            ).astype("float")
        else:
            activated_coords_base = np.transpose(
                np.unravel_index(activated, self.cell_dimensions)
            ).astype("float")

        # generate points to add
        activated_coords = np.concatenate(
            [
                activated_coords_base + [i_offset, j_offset]
                for i_offset in self.cell_coordinate_offsets
                for j_offset in self.cell_coordinate_offsets
            ]
        )

        if self.anchoring_method == "corners":
            self.active_phases = activated_coords / self.cell_dimensions
        else:
            if activated_coords.size > 0:
                self.active_phases = np.append(
                    self.active_phases,
                    activated_coords / self.cell_dimensions,
                    axis=0
                )

        self.compute_active_cells()

        self.active_segments = active_segments
        self.sensory_associated_cells = sensory_supported_cells

    def sensory_compute_learning_mode(self, anchor_input):
        """
        associate this location with a sensory input. subsequently, anchor input will
        activate the current location during anchor().

        @param anchor_input (np.array)e
        A sensory inut. this will often come from a feature-location pair layer.
        """

        # cells with an active segment: reinforce the segment
        overlaps = self.connections.computeActivity(
            anchor_input,
            self.connected_permanence
        )
        active_segments = np.where(overlaps >= self.activation_threshold)[0]

        cells_for_active_segments = self.connections.mapSegmentsToCells(active_segments)
        learning_active_segments = active_segments[
            np.isin(cells_for_active_segments, self.active_cells)
        ]

        # remaining cells with a matching segment: reinforce the best matching segment
        potential_overlaps = self.connections.computeActivity(anchor_input)
        matching_segments = np.where(potential_overlaps >= self.matching_threshold)[0]

        remaining_cells = np.setdiff1d(self.active_cells, cells_for_active_segments)

        candidate_segments = self.connections.filterSegmentsByCell(
            matching_segments,
            remaining_cells
        )
        cells_for_candidate_segments = self.connections.mapSegmentsToCells(
            candidate_segments
        )
        candidate_segments = candidate_segments[
            np.isin(cells_for_candidate_segments, remaining_cells)
        ]

        learning_matching_segments = candidate_segments[
            argmax_multi(
                potential_overlaps[candidate_segments],
                cells_for_candidate_segments
            )
        ]

        # learn on both chosen active segments and matching segments:
        #   - adjust synapse permanences
        #   - grow new synapses
        #   - grow new segments
        for learning_segments in (learning_active_segments, learning_matching_segments):
            self.connections.adjustSynapses(
                learning_segments,
                anchor_input,
                self.permanence_increment,
                -self.permanence_decrement
            )

            # grow new synapses. calculate `max_new` number of synapses to grow per
            # segment. `max_new` might be a number or a list of numbers
            if self.sample_size == -1:
                max_new = len(anchor_input)
            else:
                max_new = self.sample_size - potential_overlaps[learning_segments]

            if self.max_synapses_per_segment != -1:
                synapse_counts = self.connections.mapSegmentsToSynapseCounts(
                    learning_segments
                )

                num_synapses_until_max = self.max_synapses_per_segment - synapse_counts

                max_new = np.where(
                    max_new <= num_synapses_until_max,
                    max_new,
                    num_synapses_until_max
                )

            self.connections.growSynapsesToSample(
                learning_segments,
                anchor_input,
                max_new,
                self.initial_permanence,
                self.nupic_rng
            )

        # remaining calls without a matching segment: grow one
        new_segment_cells = np.setdiff1d(remaining_cells, cells_for_candidate_segments)

        num_new_synapses = len(anchor_input)

        if self.sample_size != -1:
            num_new_synapses = min(num_new_synapses, self.sample_size)

        if self.max_synapses_per_segment != -1:
            num_new_synapses = min(num_new_synapses, self.max_synapses_per_segment)

        new_segments = self.connections.createSegments(new_segment_cells)

        self.connections.growSynapsesToSample(
            new_segments,
            anchor_input,
            num_new_synapses,
            self.initial_permanence,
            self.nupic_rng
        )

        self.active_segments = active_segments
        self.sensory_associated_cells = self.active_cells

    def sensory_compute(self, anchor_input, anchor_growth_candidates, learn):
        if learn:
            self.sensory_compute_learning_mode(anchor_growth_candidates)
        else:
            self.sensory_compute_inference_mode(anchor_input)

    def get_active_cells(self):
        return self.active_cells

    def get_learnable_cells(self):
        return self.active_cells

    def get_sensory_associated_cells(self):
        return self.sensory_associated_cells

    def get_num_of_cells(self):
        return np.prod(self.cell_dimensions)


def gaussian(sig, d):
    return np.exp(-np.power(d, 2.) / (2 * np.power(sig, 2.)))


def argmax_multi(values, groups):
    """
    gets indices of the max values of each group in `values` (np.array),
    grouping the elements by their corresponding value in `groups` (np.array).

    returns index (within `values`) of maximal element in each group.

    example:
        argmax_multi(values = [5, 4, 7, 2, 9, 8],    -->    [2, 4]
                     groups = [0, 0, 0, 1, 1, 1])
    """

    sorter = np.argsort(groups, kind="mergesort")

    values = values[sorter]

    _, indices, lengths = np.unique(groups, return_index=True, return_counts=True)

    max_values = np.maximum.reduceat(values, indices)
    all_max_indices = np.flatnonzero(np.repeat(max_values, lengths) == values)

    indices = all_max_indices[np.searchsorted(all_max_indices, indices)]

    return sorter[indices]


def choose_reliable_active_firing_rate(
    cells_per_axis,
    bump_sigma,
    minimum_active_diameter
):
    """
    when a cell is activated by sensory input, this implies that the phase is
    within a particular small patch of the rhombus. the path is roughly equivalent
    to a circle of diameter (1/cells_per_axis) * (2/sqrt(3)), centered on the cell.
    the (2/sqrt(3)) accounts for the fact that when circles are packed into
    hexagons, there are small uncovered spaces between the circles. the circles need
    to expand by a factor of (2/sqrt(3)) to cover this space.

    this sensory input will activate the phase at the center of this cell.
    to account for uncertainty of the actual phase that was used during learning,
    the bump of active cells needs to be sufficiently large for this cell to remain
    active until the bump has moved by the above diameter.
    so the diameter of the bump (and equivalently, the cell's firing field) needs
    to be at least 2 of the above diameters.

    @param minimum_active_diameter (float or None)
    If specified, this makes sure the bump of active cells is always above a certain
    size.

    @return
    Returns `active_firing_rate`
    """

    firing_field_diameter = 2 * (1. / cells_per_axis) * (2. / np.sqrt(3))

    if minimum_active_diameter:
        firing_field_diameter = max(firing_field_diameter, minimum_active_diameter)

    return gaussian(bump_sigma, firing_field_diameter / 2.)
