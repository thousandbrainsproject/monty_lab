# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import habitat_sim.utils as hab_utils
import numpy as np
from tbp.monty.frameworks.models.motor_policies import InformedPolicy


class TouchPolicy(InformedPolicy):
    """Policy class for a touch sensor agent. Includes additional functions
    for moving along an object based on its surface normal
    """
    def __init__(
        self,
        reverse_actions,
        min_perc_on_obj,
        good_view_percentage,
        desired_object_distance,
        alpha,
        **kwargs,
    ):
        """Initialize policy.

        :param reverse_actions: actions that reverse actions specified in
                self.action_space (eg: move_left-move_right)
        :param min_perc_on_obj: Minimum percentage of patch that needs to be
                on the object. If under this amount, reverse the previous action
                to get the patch back on the object.
        :param good_view_percentage: How much percent of the view finder perception
            should be filled with the object. (If less, move closer)
        """
        super().__init__(
            reverse_actions, min_perc_on_obj, good_view_percentage, **kwargs
        )
        self.desired_object_distance = desired_object_distance
        self.tangential_angle = 0
        self.alpha = alpha

    def pre_episode(self):
        self.tangential_angle = 0
        return super().pre_episode()

    def touch_object(self, raw_observation, view_sensor_id):
        """
        At beginning of episode move close enough to the object.

        Used the raw observations returned from the dataloader and not the
        extracted features from the sensor module.
        """
        self.action = self.agent_id + ".move_forward"
        depth_at_center = self.get_depth_at_center(raw_observation, view_sensor_id)
        self.amount = (
            depth_at_center
            - self.desired_object_distance
            - self.state["agent_id_0"]["sensors"]["view_finder.depth"]["position"][2]
        )
        return self.action, self.amount

    def get_depth_at_center(self, raw_observation, view_sensor_id):
        observation_shape = raw_observation[self.agent_id][view_sensor_id][
            "depth"
        ].shape
        depth_at_center = raw_observation[self.agent_id][view_sensor_id]["depth"][
            observation_shape[0] // 2, observation_shape[1] // 2
        ]
        assert (
            depth_at_center > 0
        ), "Object must be initialized such that the touch sensor can touch it by moving forward"  # noqa: E501
        return depth_at_center

    def __call__(self):
        """Return the next action and amount.

        This requires self.processed_observations to be updated at every step
        in the Monty class. self.processed_observations contains the features
        extracted by the sensor module for the guiding sensor (patch).
        """
        assert (
            self.processed_observations["features"]["object_coverage"] > 0
        ), "Action has taken the sensor completely off the object"
        action = self.get_next_action()
        amount = self.get_next_amount(action)
        self.post_action(action, amount)
        return action, amount

    def get_next_action(self):
        """Cycle through four actions

        First move forward to touch the object at the right distance
        Then orient toward the normal along direction 1
        Then orient toward the normal along direction 2
        Then move tangentially along the object surface
        Then start over

        """
        self._last_action, _ = self.last_action()
        if "move_forward" in self._last_action:
            return self.agent_id + ".orient_horizontal"
        elif "orient_horizontal" in self._last_action:
            return self.agent_id + ".orient_vertical"
        elif "orient_vertical" in self._last_action:
            return self.agent_id + ".move_tangentially"
        elif "move_tangentially" in self._last_action:
            # orient around object if it's not centered in view
            if self.processed_observations["features"]["on_object"] == 0:
                return self.agent_id + ".orient_horizontal"
            # move to the desired_object_distance if it is in view
            else:
                return self.agent_id + ".move_forward"

    def get_next_constraint(self, action, amount):
        """
        Set the 'constraint' of actuation to be a direction 0 - 2pi.
        This controls the move_tangential action
            - currently set to 0 (go up)
            - could also be random: np.random.rand()*2*np.pi
              to generate a random angle 0 - 2pi
        set the 'constraint' for orientation as the distance that the controller
        travels to compensate for the rotation
        TODO: replicate a version of the self.action_name_to_sample_fn
        process which is done with amounts but not constraints
        """
        amount_radians = np.radians(amount)
        depth = self.processed_observations["features"]["depth"]
        if "move_tangentially" in action:
            new_target_direction = (np.random.rand() - 0.5) * 2 * np.pi
            self.tangential_angle += new_target_direction * self.alpha
            return hab_utils.quat_rotate_vector(
                self.state["agent_id_0"]["rotation"],
                [np.sin(self.tangential_angle), np.cos(self.tangential_angle), 0]
            )
        if "orient_horizontal" in action:
            move_left_amount = np.tan(amount_radians) * depth
            move_forward_amount = (
                depth * (1 - np.cos(np.radians(amount))) / np.cos(np.radians(amount))
            )
            return move_left_amount, move_forward_amount
        if "orient_vertical" in action:
            move_down_amount = np.tan(amount_radians) * depth
            move_forward_amount = (
                depth * (1 - np.cos(np.radians(amount))) / np.cos(np.radians(amount))
            )
            return move_down_amount, move_forward_amount

    def get_next_amount(self, action):
        """
        TODO: integrate better with self.action_name_to_sample_fn
        """
        # during the initialization step, there is no amount needed yet
        if not hasattr(self, "processed_observations"):
            return None
        if "move_tangentially" in action:
            return self.action_name_to_sample_fn[action](
                **self.action_names_to_params[action]
            )
        if "move_forward" in action:
            return (
                self.processed_observations["features"]["depth"]
                - self.desired_object_distance
            )
        if "orient" in action:
            return self.orienting_angle_from_normal(action)

    def orienting_angle_from_normal(self, action):
        original_point_normal = self.processed_observations["features"]["point_normal"]

        inverse_magnum_rotation = hab_utils.common.quat_to_magnum(
            self.state["agent_id_0"]["rotation"]
        ).inverted()
        inverse_quaternion_rotation = hab_utils.common.quat_from_magnum(
            inverse_magnum_rotation
        )
        rotated_point_normal = hab_utils.quat_rotate_vector(
            inverse_quaternion_rotation, original_point_normal
        )
        x, y, z = rotated_point_normal

        if "right" in action:
            return -np.degrees(np.arctan(x / z))
        if "up" in action:
            return -np.degrees(np.arctan(y / z))
