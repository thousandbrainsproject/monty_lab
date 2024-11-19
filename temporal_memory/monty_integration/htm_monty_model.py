# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Used to be in frameworks/models/

from tbp.monty.frameworks.models.abstract_monty_classes import Monty


class MontyHTM(Monty):
    def __init__(self, learning_modules):
        self.learning_modules = learning_modules

        self.step_type = None

    def _matching_step(self, observation):
        pass

    def _exploratory_step(self, observation):
        pass

    def step(self, observation):
        pass

    def aggregate_sensory_inputs(self, observation):
        pass

    def _step_learning_modules(self):
        pass

    def _vote(self):
        pass

    def _pass_goal_states(self):
        pass

    def _set_step_type_and_check_if_done(self):
        pass

    def _post_step(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def pre_episode(self):
        pass

    def post_episode(self):
        pass

    def set_experiment_mode(self, mode):
        pass

    def is_done(self):
        pass


class SingleLMMontyHTM(MontyHTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(self.learning_modules) == 1

    def step(self, observation):
        if self.step_type == "matching_step":
            self._matching_step(observation)
        elif self.step_type == "exploratory_step":
            self._exploratory_step(observation)
        else:
            raise ValueError("Unknown step type!")

    def matching_step(self, observation):
        self.learning_modules[0].matching_step(observation)

    def exploratory_step(self, observation):
        self.learning_modules[0].exploratory_step(observation)

    def set_step_type(self, step_type):
        assert step_type in ["matching_step", "exploratory_step"], "Unknown step type!"

        self.step_type = step_type
        self.learning_modules[0].step_type = step_type

    def pre_epoch(self):
        self.learning_modules[0].pre_epoch()

    def post_epoch(self):
        self.learning_modules[0].post_epoch()

    def pre_episode(self):
        self.learning_modules[0].pre_episode()

    def post_episode(self, object_id):
        """
        passing in `object_id` to each LM's post_episode() because we need to update
        a dictionary which contains data pertaining to each object.
        this detail is required by the L4_and_L6a_3d_LM LearningModule.
        """
        self.learning_modules[0].post_episode(object_id)

    def is_done(self):
        return self.learning_modules[0].is_done
