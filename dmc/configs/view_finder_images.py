# Copyright 2025 Thousand Brains Project
# Copyright 2023 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Store images from the view-finder for use with the ViT-based models.
This module contains configs, a logger, and a motor policy for generating RGBD images
of objects taken from the viewfinder. The motor policy helps (but doesn't guarantee)
that the whole object fits within the view-finder's frame. It does this by moving
forward until the object enters a small buffer region around the viewfinder's frame.
The logger saves the images as .npy files and writes a jsonl file containing metadata
about the object and pose for each image.

Three configs are defined:
- view_finder_base: 14 standard training rotations
- view_finder_randrot_all: 14 randomly generated rotations
- view_finder_randrot: 5 pre-defined "random" rotations
- view_finder_32: 32 training rotations for rapid learning experiments

All use 77 objects.

To visualize the images, run the script
`monty_lab/dmc/scripts/render_view_finder_images.py`.
"""

import copy
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Mapping, Tuple, Union

import numpy as np
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
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
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMountHabitatDatasetArgs,
)

from .common import DMC_PRETRAIN_DIR, DMC_ROOT_DIR, RANDOM_ROTATIONS_5
from .fig6_rapid_learning import TRAIN_ROTATIONS as TRAIN_ROTATIONS_32

# All view-finder image experiments will be stored under 'view_finder_images',
# a directory at the same level as the results directory.
VIEW_FINDER_DIR = DMC_ROOT_DIR / "view_finder_images"


class ViewFinderRGBDHandler(MontyHandler):
    """Save RGBD from view finder and episode metadata at the end of each episode."""

    def __init__(self):
        self.initialized = False
        self.save_dir = None
        self.view_finder_id = None

    @classmethod
    def log_level(cls):
        return "DETAILED"

    def initialize(
        self,
        data: Mapping,
        output_dir: str,
        episode: int,
        mode: str,
        **kwargs,
    ) -> None:
        """Initialize the handler.

        Sets `self.save_dir`, the path to `view_finder_rgbd` under the experiment's
        output directory, and creates it along with child directory for numpy arrays.
        If the directory exists, it will be deleted.

        Also infers the view finder's sensor module ID and sets `self.view_finder_id`.

        Args:
            data (Mapping): The data from the model..
            output_dir (str): The experiment's output directory.
            episode (int): The current episode number.
            mode (str): The mode (train or eval).
        """
        # Create the output directory. An existing directory will be deleted.
        output_dir = Path(output_dir).expanduser()
        self.save_dir = output_dir / "view_finder_rgbd"
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True)

        # Create arrays subdirectory.
        arrays_dir = self.save_dir / "arrays"
        arrays_dir.mkdir()

        # Determine which sensor module ID to use. Probably always 1.
        sm_ids = [k for k in data["DETAILED"][episode].keys() if k.startswith("SM_")]
        sm_nums = [int(name.split("_")[-1]) for name in sm_ids]
        self.view_finder_id = f"SM_{max(sm_nums)}"
        self.initialized = True

    def report_episode(
        self,
        data: Mapping,
        output_dir: str,
        episode: int,
        mode: str = "train",
        **kwargs,
    ) -> None:
        """Store the RGBD image and episode metadata.

        Args:
            data (Mapping): The data from the model.
            output_dir (str): The experiment's output directory.
            episode (int): The current episode number.
            mode (str): The mode (train or eval).
        """

        if not self.initialized:
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

    def close(self):
        """Does nothing but must be implemented."""
        pass


class FramedObjectPolicy(InformedPolicy):
    """Custom motor policy that helps keep the object in-frame

    Reimplements `InformedPolicy.move_close_enough` to add an extra termination
    condition: if the object enters a small buffer region in the view-finder's
    image frame, the agent will not move forward again. This class also moves
    forward in smaller increments than the default policy.

    """

    def move_close_enough(
        self,
        raw_observation: Mapping,
        view_sensor_id: str,
        target_semantic_id: int,
        multiple_objects_present: bool,
    ) -> Tuple[Union[Action, None], bool]:
        """At beginning of episode move close enough to the object.

        Acts almost identically to the `InformedPolicy.move_close_enough` method
        but adds an extra condition that will halt the agent's advances. More
        specifically, the agent will not move forward if the object breaches a
        smaller buffer region around the image frame.

        Args:
            raw_observation: The raw observations from the dataloader
            view_sensor_id: The ID of the view sensor
            target_semantic_id: The semantic ID of the primary target object in the
                scene.
            multiple_objects_present: Whether there are multiple objects present in the
                scene. If so, we do additional checks to make sure we don't get too
                close to these when moving forward

        Returns:
            Tuple[Union[Action, None], bool]: The next action to take (may be `None`
            and whether the agent is close enough.

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

# Define the 14 standard training rotations used for 'view_finder_base'.
train_rotations = get_cube_face_and_corner_views_rotations()

# Define out motor system config that uses the custom policy and uses
# a shorter desired object distance than default.
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
        model_name_or_path=str(DMC_PRETRAIN_DIR / "dist_agent_1lm/pretrained"),
        n_eval_epochs=len(train_rotations),
        max_eval_steps=1,
        max_total_steps=1,
    ),
    logging_config=EvalLoggingConfig(
        output_dir=str(VIEW_FINDER_DIR),
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
    # Set up eval dataloader with objects placed closer to the agent than default.
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
# Set viewfinder resolution to 224 x 224.
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
view_finder_randrot_all = copy.deepcopy(view_finder_base)
view_finder_randrot_all["experiment_args"].n_eval_epochs = 14
view_finder_randrot_all["logging_config"].run_name = "view_finder_randrot_all"
view_finder_randrot_all[
    "eval_dataloader_args"
].object_init_sampler = RandomRotationObjectInitializer()

"""
5 (Pre-defined) Random Rotations
--------------------------------
"""
view_finder_randrot = copy.deepcopy(view_finder_base)
view_finder_randrot["experiment_args"].n_eval_epochs = 5
view_finder_randrot["logging_config"].run_name = "view_finder_randrot"
view_finder_randrot[
    "eval_dataloader_args"
].object_init_sampler = PredefinedObjectInitializer(
    positions=[[0.0, 1.5, -0.2]],
    rotations=RANDOM_ROTATIONS_5,
)
"""
32 Training Rotations for Rapid Learning Experiments
----------------------------------------------------
"""
view_finder_32 = copy.deepcopy(view_finder_base)
view_finder_32["experiment_args"].n_eval_epochs = 32
view_finder_32["logging_config"].run_name = "view_finder_32"
view_finder_32[
    "eval_dataloader_args"
].object_init_sampler = PredefinedObjectInitializer(
    positions=[[0.0, 1.5, -0.2]],
    rotations=TRAIN_ROTATIONS_32,
)

CONFIGS = {
    "view_finder_base": view_finder_base,
    "view_finder_randrot_all": view_finder_randrot_all,
    "view_finder_randrot": view_finder_randrot,
    "view_finder_32": view_finder_32,
}
