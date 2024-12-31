# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentDataLoaderPerObject,
    InformedEnvironmentDataLoader,
)


class TouchEnvironmentDataLoader(InformedEnvironmentDataLoader):
    """
    Extension of the InofrmedEnvironmentDataLoader with initialization steps
    specifically for the touch agent (touch_object)

    In addition, the __next__ call now includes a get_constraint method
    This method adds an additional parameter that is needed for the
    move_tangentially, orient_vertical, and orient_horizontal actions
    """

    def __next__(self):
        if self._counter == 0:
            # Return first observation after 'reset' before any action is applied
            self._counter += 1
            return self._observation
        else:
            action_with_amount = self.motor_system()
            self._action, self._amount = action_with_amount
            self.update_habitat_sim_agent_actuation_constraint()
            self._observation, self.motor_system.state = self.dataset[
                action_with_amount
            ]
            self._counter += 1
            return self._observation

    def touch_object(self, view_sensor_id="view_finder"):
        """Policy to touch the object before an episode starts.

        :param view_sensor_id: The name of the touch sensor
        :param min_depth: How much percent of the view finder perception
                should be filled with the object. (If less, move closer)
        """
        action, amount = self.motor_system.touch_object(
            self._observation, view_sensor_id
        )

        self._observation, self.motor_system.state = self.dataset[(action, amount)]

    def update_habitat_sim_agent_actuation_constraint(self):
        """
        set the 'constraint' of actuation to be a direction 0 - 2pi.
        This controls the move_tangential action
        set the 'constraint' for orientation as the distance that
        the controller travels to compensate for the rotation
        """
        constraint = self.motor_system.get_next_constraint(self._action, self._amount)
        if "move_tangentially" in self._action or "orient" in self._action:
            self.dataset.env._env._sim.agents[0].agent_config.action_space[
                self._action
            ].actuation.constraint = constraint

    def pre_episode(self):
        EnvironmentDataLoaderPerObject.pre_episode(self)
        self.touch_object()
