# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Generate images from the viewfinder for use with the ViT-based models.
This module contains configs, a logger, and a motor policy for generating RGBD images
of objects taken from the viewfinder. The motor policy ensures that the whole
object fits within the view-finder's frame. It does this by moving forward until the
object enters a small buffer region around the viewfinder's frame. The logger saves the
images as .npy files and writes a jsonl file containing metadata about the object
and pose for each image.

To visualize the images, run the script
`monty_lab/dmc_config/scripts/render_view_finder_images.py`.
"""

import copy
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    MoveForward,
)
from tbp.monty.frameworks.config_utils.config_args import (
    EvalLoggingConfig,
    MontyArgs,
    MotorSystemConfigInformedNoTransStepS20,
    PatchAndViewMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_informed_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.loggers.monty_handlers import MontyHandler
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.motor_policies import (
    InformedPolicy,
    get_perc_on_obj_semantic,
)

from .common import PRETRAIN_DIR, RANDOM_ROTATIONS_5

"""
Basic setup
-----------
"""
# Specify parent directory where all experiment should live under.
output_dir = os.path.expanduser("~/tbp/results/monty/projects/view_finder_images")

# Where to find the pretrained model. Not really used, but necessary
# to set up an eval experiment.
model_name = PRE
model_path = os.path.expanduser(
    "~/tbp/results/dmc/pretrained_models/dist_agent_1lm/pretrained/model.pt"
)

train_rotations = get_cube_face_and_corner_views_rotations()


class ViewFinderRGBDHandler(MontyHandler):
    """Save RGBD from view finder and episode metadata at the end of each episode."""

    def __init__(self):
        self.report_count = 0
        self.save_dir = None
        self.view_finder_id = None

    @classmethod
    def log_level(cls):
        return "DETAILED"

    def initialize(self, data, output_dir, episode, mode, **kwargs):
        # Create output directory, putting existing directory into another
        # location if it exists.
        output_dir = Path(output_dir).expanduser()
        self.save_dir = output_dir / "view_finder_rgbd"
        if self.save_dir.exists():
            old_dir = output_dir / "view_finder_rgbd_old"
            if old_dir.exists():
                shutil.rmtree(old_dir)
            self.save_dir.rename(old_dir)
        self.save_dir.mkdir(parents=True)

        # Create arrays subdirectory.
        arrays_dir = self.save_dir / "arrays"
        arrays_dir.mkdir()

        # Determine which sensor module ID to use. Probably always 1.
        sm_ids = [k for k in data["DETAILED"][episode].keys() if k.startswith("SM_")]
        sm_nums = [int(name.split("_")[-1]) for name in sm_ids]
        self.view_finder_id = f"SM_{max(sm_nums)}"

    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
        """
        Changed name to report episode since we are currently running with
        reporting and flushing exactly once per episode
        """

        if self.report_count == 0:
            self.initialize(data, output_dir, episode, mode, **kwargs)

        output_data = dict()
        output_data["episode"] = episode
        if mode == "eval":
            target_info = data["BASIC"]["eval_targets"][episode]
            output_data["object"] = target_info["primary_target_object"]
            output_data["rotation"] = target_info["primary_target_rotation_euler"]

        # Combine RGB and depth into a single RGBD image.
        obs = data["DETAILED"][episode][self.view_finder_id]["raw_observations"][-1]
        rgba = obs["rgba"]
        depth = obs["depth"]

        rgbd = rgba / 255.0
        rgbd[:, :, 3] = depth

        # Save the image.
        arrays_dir = self.save_dir / "arrays"
        array_path = arrays_dir / f"{episode}.npy"
        np.save(array_path, rgbd)

        # Save the metadata.
        metadata_path = self.save_dir / "episodes.jsonl"
        with open(metadata_path, "a") as f:
            json.dump(output_data, f, cls=BufferEncoder)
            f.write(os.linesep)

        self.report_count += 1

    def close(self):
        pass


class FramedObjectPolicy(InformedPolicy):
    """Custom motor policy that helps keep the object in-frame"""

    def move_close_enough(
        self,
        raw_observation: Mapping,
        view_sensor_id: str,
        target_semantic_id: int,
        multiple_objects_present: bool,
    ):
        # ) -> Tuple[Union[Action, None], bool]:
        """At beginning of episode move close enough to the object.

        Used the raw observations returned from the dataloader and not the
        extracted features from the sensor module.

        Args:
            raw_observation: The raw observations from the dataloader
            view_sensor_id: The ID of the view sensor
            target_semantic_id: The semantic ID of the primary target object in the
                scene.
            multiple_objects_present: Whether there are multiple objects present in the
                scene. If so, we do additional checks to make sure we don't get too
                close to these when moving forward

        Returns:
            Tuple[Union[Action, None], bool]: The next action to take and whether the
                episode is done.

        Raises:
            ValueError: If the object is not visible
        """
        # Reconstruct 2D semantic map.
        depth_image = raw_observation[self.agent_id][view_sensor_id]["depth"]
        semantic_3d = raw_observation[self.agent_id][view_sensor_id]["semantic_3d"]
        semantic_image = semantic_3d[:, 3].reshape(depth_image.shape).astype(int)

        if not multiple_objects_present:
            semantic_image[semantic_image > 0] = target_semantic_id

        points_on_target_obj = semantic_image == target_semantic_id
        n_points_on_target_obj = points_on_target_obj.sum()

        # For multi-object experiments, handle the possibility that object is no
        # longer visible.
        if multiple_objects_present and n_points_on_target_obj == 0:
            logging.debug("Object not visible, cannot move closer")
            return None, True

        if n_points_on_target_obj > 0:
            closest_point_on_target_obj = np.min(depth_image[points_on_target_obj])
            logging.debug(
                "closest target object point: " + str(closest_point_on_target_obj)
            )
        else:
            raise ValueError(
                "May be initializing experiment with no visible target object"
            )

        perc_on_target_obj = get_perc_on_obj_semantic(
            semantic_image, semantic_id=target_semantic_id
        )
        logging.debug("% on target object: " + str(perc_on_target_obj))

        # If the object touches outer pixels, we are close enough.
        edge_buffer_pct = 5
        edge_buffer = int(edge_buffer_pct / 100 * semantic_image.shape[0])
        if semantic_image[:edge_buffer, :].sum() > 0:  # top side
            return None, True
        elif semantic_image[-edge_buffer:, :].sum() > 0:  # bottom side
            return None, True
        elif semantic_image[:, :edge_buffer].sum() > 0:  # left side
            return None, True
        elif semantic_image[:, -edge_buffer:].sum() > 0:  # right side
            return None, True

        # Also calculate closest point on *any* object so that we don't get too close
        # and clip into objects; NB that any object will have a semantic ID > 0
        points_on_any_obj = semantic_image > 0
        closest_point_on_any_obj = np.min(depth_image[points_on_any_obj])
        logging.debug("closest point on any object: " + str(closest_point_on_any_obj))

        if perc_on_target_obj < self.good_view_percentage:
            if closest_point_on_target_obj > self.desired_object_distance:
                if multiple_objects_present and (
                    closest_point_on_any_obj < self.desired_object_distance / 4
                ):
                    logging.debug(
                        "getting too close to other objects, not moving forward"
                    )
                    return None, True  # done
                else:
                    logging.debug("move forward")
                    return MoveForward(agent_id=self.agent_id, distance=0.005), False
            else:
                logging.debug("close enough")
                return None, True  # done
        else:
            logging.debug("Enough percent visible")
            return None, True  # done


"""
Configs
---------
"""

motor_system_config = MotorSystemConfigInformedNoTransStepS20(
    motor_system_class=FramedObjectPolicy,
    motor_system_args=make_informed_policy_config(
        action_space_type="distant_agent_no_translation",
        action_sampler_class=ConstantSampler,
        rotation_degrees=5.0,
        use_goal_state_driven_actions=False,
        switch_frequency=1.0,
        good_view_percentage=0.5,  # Make sure we define the required
        desired_object_distance=0.2,
    ),
)

# The config dictionary for the standard experiment with 14 standard training rotations.
view_finder_base = dict(
    # Set up experiment
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(PRETRAIN_DIR / "dist_agent_1lm/pretrained"),
        n_eval_epochs=len(train_rotations),
        max_eval_steps=1,
        max_total_steps=1,
    ),
    logging_config=EvalLoggingConfig(
        output_dir=output_dir,
        run_name="view_finder_base",
        monty_handlers=[ViewFinderRGBDHandler],
        wandb_handlers=[],
    ),
    # Set up monty, including LM, SM, and motor system.
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=1),
        motor_system_config=motor_system_config,
    ),
    # Set up environment/data
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    # dataset_args=dataset_args,
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(
            positions=[[0.0, 1.5, -0.2]], rotations=train_rotations
        ),
    ),
    # Doesn't get used, but currently needs to be set anyways.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
)
# Set viewfinder resolution to 224 x 224
dataset_args = view_finder_base["dataset_args"]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = [
    [64, 64],
    [224, 224],
]
dataset_args.__post_init__()

"""
14 Randomly Generated Rotations
-------------------------------
"""
view_finder_randrot_14 = copy.deepcopy(view_finder_base)
view_finder_randrot_14["experiment_args"].n_eval_epochs = 14
view_finder_randrot_14["logging_config"].run_name = "view_finder_randrot_14"
view_finder_randrot_14[
    "eval_dataloader_args"
].object_init_sampler = RandomRotationObjectInitializer()

"""
5 (Pre-defined) Random Rotations
--------------------------------
"""
view_finder_randrot_5 = copy.deepcopy(view_finder_base)
view_finder_randrot_5["experiment_args"].n_eval_epochs = 5
view_finder_randrot_5["logging_config"].run_name = "view_finder_randrot_5"
view_finder_randrot_5[
    "eval_dataloader_args"
].object_init_sampler = PredefinedObjectInitializer(
    positions=[[0.0, 1.5, -0.2]],
    rotations=RANDOM_ROTATIONS_5,
)

CONFIGS = {
    "view_finder_base": view_finder_base,
    "view_finder_randrot_14": view_finder_randrot_14,
    "view_finder_randrot_5": view_finder_randrot_5,
}
