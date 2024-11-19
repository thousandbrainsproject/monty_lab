# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Used to be in frameworks/models/

import numpy as np
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.htm import L4, L6a_3d


class L4_and_L6a_3d_LM(LearningModule):  # noqa N801
    def __init__(
        self,
        tm_num_minicolumns,
        tm_num_cells_per_minicolumn,
        tm_proximal_w,
        tm_initial_permanence,
        tm_connected_permanence,
        tm_permanence_increment,
        tm_permanence_decrement,
        tm_seed,
        gc_num_modules_per_axis,
        gc_num_cells_per_axis_per_module,
        gc_cell_coordinate_offsets,
        gc_activation_threshold,
        gc_initial_permanence,
        gc_connected_permanence,
        gc_matching_threshold,
        gc_sample_size,
        gc_permanence_increment,
        gc_permanence_decrement,
        gc_anchoring_method,
        gc_random_location,
        gc_seed,
    ):

        self.l4 = L4(
            proximal_n=tm_num_minicolumns,
            proximal_w=tm_proximal_w,
            basal_n=(
                3
                * gc_num_modules_per_axis
                * gc_num_cells_per_axis_per_module
                * gc_num_cells_per_axis_per_module
            ),
            basal_w=(3 * gc_num_modules_per_axis),
            apical_n=0,
            apical_w=0,
            cells_per_column=tm_num_cells_per_minicolumn,
            activation_threshold=int(3 * gc_num_modules_per_axis * 0.5),
            reduced_basal_threshold=int(3 * gc_num_modules_per_axis * 0.5),
            initial_permanence=tm_initial_permanence,
            connected_permanence=tm_connected_permanence,
            matching_threshold=int(3 * gc_num_modules_per_axis * 0.5),
            sample_size=gc_num_modules_per_axis,
            permanence_increment=tm_permanence_increment,
            permanence_decrement=tm_permanence_decrement,
            seed=tm_seed,
        )

        self.l6a = L6a_3d(
            num_modules=gc_num_modules_per_axis,
            cells_per_axis=gc_num_cells_per_axis_per_module,
            anchor_input_size=(tm_num_minicolumns * tm_num_cells_per_minicolumn),
            cell_coordinate_offsets=gc_cell_coordinate_offsets,
            activation_threshold=gc_activation_threshold,
            initial_permanence=gc_initial_permanence,
            connected_permanence=gc_connected_permanence,
            matching_threshold=gc_matching_threshold,
            sample_size=gc_sample_size,
            permanence_increment=gc_permanence_increment,
            permanence_decrement=gc_permanence_decrement,
            anchoring_method=gc_anchoring_method,
            random_location=gc_random_location,
            seed=gc_seed,
        )

        # ensure variables are defined for the first time
        self.locations_are_unique = None
        self.location_representation_set = None
        self.per_object_data = None
        self.per_sensation_data = None

        self.is_done = False

    def matching_step(self, observation):
        """
        the inference phase.
        """

        i, (test_id, curve, coord) = observation

        # step 1: motor input to grid cell layer to do path integration and activate
        #         cells
        self.l6a.movement_compute(next_location=coord.numpy())

        # step 2, 3: sensory input fed to temporal memory layer, with basal input
        #            coming from grid cell layer
        self.l4.sensory_compute(
            active_columns=curve.squeeze().numpy(),
            basal_input=self.l6a.get_location_representation(),
            basal_growth_candidates=self.l6a.get_learnable_location_representation(),
            learn=False
        )

        # step 4: grid cell layer updated with new sensory representation
        self.l6a.sensory_compute(
            anchor_input=self.l4.get_active_cells(),
            anchor_growth_candidates=self.l4.get_winner_cells(),
            learn=False
        )

        # use sensory activated grid cells to determine if L4's active cell
        # representations is recognizable by the location layer
        inference_location_representation = \
            self.l6a.get_sensory_associated_location_representation()

        target_location_representation = set(
            np.concatenate(self.per_object_data[test_id.item()])
        )

        # inference rule:
        # a representation is correctly inferred if the inference location is
        # a *strict* subset of the location representations collecting during
        # training for this particular object.
        self.is_done = (
            len(inference_location_representation)
            and set(
                inference_location_representation
            ) <= target_location_representation
        )

        if self.is_done:
            print(
                "Converged upon a correct representation in {0} step(s) for "
                "object {1}".format(i + 1, test_id.item())
            )
            print()
        elif not self.is_done and (
            tuple(
                inference_location_representation
            ) in self.location_representation_set
        ):
            print("Converged upn an incorrect representation.")

            self.is_done = True

    def exploratory_step(self, observation):
        """
        the training phase.
        """

        (curve, coord) = observation

        # step 1: motor input to grid cell layer to do path integration and activate
        #         cells
        self.l6a.movement_compute(next_location=coord.numpy())

        # step 2, 3: sensory input fed to temporal memory layer, with basal input
        #            coming from grid cell layer
        self.l4.sensory_compute(
            active_columns=curve.squeeze().numpy(),
            basal_input=self.l6a.get_location_representation(),
            basal_growth_candidates=self.l6a.get_learnable_location_representation(),
            learn=True
        )

        # step 4: grid cell layer updated with new sensory representation
        self.l6a.sensory_compute(
            anchor_input=self.l4.get_active_cells(),
            anchor_growth_candidates=self.l4.get_winner_cells(),
            learn=True
        )

        # step 5: store the sensory activated grid cells corresponding to this sensation
        sensory_associated_location_representation = \
            self.l6a.get_sensory_associated_location_representation()

        self.locations_are_unique = (
            self.locations_are_unique
            and tuple(
                sensory_associated_location_representation
            ) not in self.location_representation_set
        )

        self.location_representation_set.add(
            tuple(sensory_associated_location_representation)
        )

        self.per_sensation_data.append(sensory_associated_location_representation)

    def receive_votes(self, votes):
        pass

    def send_out_vote(self):
        pass

    def propose_goal_state(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def set_experiment_mode(self, mode):
        pass

    def pre_epoch(self):
        if self.step_type == "exploratory_step":
            self.locations_are_unique = True
            self.location_representation_set = set()
            self.per_object_data = dict()
        elif self.step_type == "matching_step":
            pass
        else:
            raise ValueError("Unknown step type!")

    def post_epoch(self):
        pass

    def pre_episode(self):
        self.l4.reset()
        self.l6a.reset()

        # set a location in the object space to first look at
        # displacements will be calculated from this first point onwards
        self.l6a.set_first_location()

        self.is_done = False

        if self.step_type == "exploratory_step":
            # activate random location in grid cell space only for training phase
            self.l6a.activate_random_location()

            self.per_sensation_data = []
        elif self.step_type == "matching_step":
            pass
        else:
            raise ValueError("Unknown step type!")

    def post_episode(self, object_id):
        if self.step_type == "exploratory_step":
            self.per_object_data[object_id.item()] = self.per_sensation_data
        elif self.step_type == "matching_step":
            if not self.is_done:
                print(
                    "Unable to converge upon correct representation for {0}".format(
                        object_id.item()
                    )
                )
                print()
        else:
            raise ValueError("Unknown step type!")
