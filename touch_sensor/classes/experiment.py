# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import matplotlib.pyplot as plt

from tbp.monty.frameworks.experiments import MontyExperiment


class MontyTouchSensorExperiment(MontyExperiment):
    """Stripped down experiment, based on Viviane's code in custom_monty_experiments.py
    For now, I am taking out the learning and inference, and just plotting
    the agent's observations after getting a good view during the pre_episode
    """

    def run_episode(self):
        """Run one episode until model is_done."""
        self.pre_episode()
        for step, observation in enumerate(self.dataloader):
            if self.show_sensor_output:
                self.show_observation(observation, step)
            self.pass_observation_to_motor_system(observation)
            if step >= self.max_steps:
                break
        self.post_episode()

    def pre_episode(self):
        """Pre episode where we pass target object to the model for logging."""
        self.model.pre_episode()
        self.dataloader.pre_episode()
        self.max_steps = self.experiment_args.max_train_steps
        if self.show_sensor_output:
            self.initialize_online_plotting()

    def post_episode(self):
        self.dataloader.post_episode()

    def initialize_online_plotting(self):
        self.fig, self.ax = plt.subplots(
            1, 2, figsize=(9, 5), gridspec_kw={"width_ratios": [1, 1.1]}
        )
        self.colorbar = self.fig.colorbar(None, fraction=0.046, pad=0.04)
        [ax.set_axis_off() for ax in self.ax.ravel()]
        self.ax[0].set_title("Vision")
        self.ax[1].set_title("Touch")

    def pass_observation_to_motor_system(self, observation):
        sensor_module_outputs = self.model.aggregate_sensory_inputs(observation)
        self.model.motor_system.processed_observations = sensor_module_outputs[0]

    def show_observation(self, observation, step):
        self.fig.suptitle(
            f"Observation at step {step}"
            + ("" if step == 0 else f"\n{self.dataloader._action.split('.')[-1]}")
        )
        self.show_view_finder(observation)
        self.show_patch(observation)
        plt.pause(0.00001)

    def show_view_finder(self, observation):
        self.ax[0].clear()
        self.ax[0].imshow(observation["agent_id_0"]["view_finder"]["rgba"])
        # Show a square in the middle as a rough estimate of where the touch agent is
        image_shape = observation["agent_id_0"]["view_finder"]["rgba"].shape
        square = plt.Rectangle(
            (image_shape[1] * 4.5 // 10, image_shape[0] * 4.5 // 10),
            image_shape[1] / 10,
            image_shape[0] / 10,
            fc="none",
            ec="white",
        )
        self.ax[0].add_patch(square)

    def show_patch(self, observation):
        self.ax[1].clear()
        depth_image = self.ax[1].imshow(
            observation["agent_id_0"]["patch"]["depth"], cmap="viridis_r"
        )
        self.colorbar.update_normal(depth_image)
