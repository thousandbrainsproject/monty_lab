# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import dataclass

from habitat_sim.agent import ActionSpec, ActuationSpec

from tbp.monty.simulators.habitat.agents import MultiSensorAgent


@dataclass(frozen=True)
class TouchAgent(MultiSensorAgent):
    """
    Adds left and right movement to the MultiSensorAgent
    And remove the absolute action space option
    """

    def get_spec(self):
        spec = super().get_spec()

        spec.action_space = {
            # Body actions (move the agent and sensors)
            f"{self.agent_id}.move_tangentially": ActionSpec(
                "move_tangentially", ActuationSpec(amount=self.translation_step)
            ),
            f"{self.agent_id}.move_forward": ActionSpec(
                "move_forward", ActuationSpec(amount=self.translation_step)
            ),
            f"{self.agent_id}.orient_horizontal": ActionSpec(
                "orient_horizontal", ActuationSpec(amount=self.rotation_step)
            ),
            f"{self.agent_id}.orient_vertical": ActionSpec(
                "orient_vertical", ActuationSpec(amount=self.rotation_step)
            ),
        }

        return spec
