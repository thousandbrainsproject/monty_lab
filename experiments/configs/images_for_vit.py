"""
This module contains configs, a logger, and a motor policy for generating RGBD images
of objects taken from the viewfinder. The motor policy ensures that the whole
object fits within the view-finder's frame. It does this by moving forward until the
object enters a small buffer region around the viewfinder's frame. The logger saves the
images as .npy files and writes a jsonl file containing metadata about the object
and pose for each image.

The primary use case for this module is to generate object images used for training
and testing traditional models (e.g., vision transformers).

"""
import copy
import json
import logging
import os
import shutil
from pathlib import Path

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


class ViewFinderRGBDHandler(MontyHandler):
    """Save RGBD from view finder at the end of each episode."""

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

        # Create images subdirectory.
        images_dir = self.save_dir / "images"
        images_dir.mkdir()

        # Determine which sensor module ID to use.
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
        images_dir = self.save_dir / "images"
        image_path = images_dir / f"{episode}.npy"
        np.save(image_path, rgbd)

        # Save the metadata.
        metadata_path = self.save_dir / "episodes.json"
        with open(metadata_path, "a") as f:
            json.dump(output_data, f, cls=BufferEncoder)
            f.write(os.linesep)

        self.report_count += 1

    def close(self):
        pass


class FramedObjectPolicy(InformedPolicy):
    def move_close_enough(
        self, raw_observation, view_sensor_id, target_semantic_id, multi_objects_present
    ):  # -> Tuple[Union[Action, None], bool]:
        """At beginning of episode move close enough to the object.

        Used the raw observations returned from the dataloader and not the
        extracted features from the sensor module.

        :param target_semantic_id: The semantic ID of the primary target object in the
            scene.
        :param multi_objects_present: Whether there are multiple objects present in
            the scene. If so, we do additional checks to make sure we don't get too
            close to these when moving forward
        """

        view = raw_observation[self.agent_id][view_sensor_id]["semantic"]
        points_on_target_obj = (
            raw_observation[self.agent_id][view_sensor_id]["semantic"]
            == target_semantic_id
        )

        # For multi-object experiments, handle the possibility that object is no
        # longer visible
        if multi_objects_present and (
            len(
                raw_observation[self.agent_id][view_sensor_id]["depth"][
                    points_on_target_obj
                ]
            )
            == 0
        ):
            logging.debug("Object not visible, cannot move closer")
            return None, True

        if len(points_on_target_obj) > 0:
            closest_point_on_target_obj = np.min(
                raw_observation[self.agent_id][view_sensor_id]["depth"][
                    points_on_target_obj
                ]
            )
            logging.debug(
                "closest target object point: " + str(closest_point_on_target_obj)
            )
        else:
            raise ValueError(
                "May be initializing experiment with no visible target object"
            )

        perc_on_target_obj = get_perc_on_obj_semantic(
            view, sematic_id=target_semantic_id
        )
        logging.debug("% on target object: " + str(perc_on_target_obj))

        # If the object touches outer pixels, we are close enough.
        edge_buffer_pct = 5
        edge_buffer = int(edge_buffer_pct / 100 * view.shape[0])
        if view[:edge_buffer, :].sum() > 0:  # top side
            return None, True
        elif view[-edge_buffer:, :].sum() > 0:  # bottom side
            return None, True
        elif view[:, :edge_buffer].sum() > 0:  # left side
            return None, True
        elif view[:, -edge_buffer:].sum() > 0:  # right side
            return None, True

        # Also calculate closest point on *any* object so that we don't get too close
        # and clip into objects; NB that any object will have a semantic ID > 0
        points_on_any_obj = (
            raw_observation[self.agent_id][view_sensor_id]["semantic"] > 0
        )
        closest_point_on_any_obj = np.min(
            raw_observation[self.agent_id][view_sensor_id]["depth"][points_on_any_obj]
        )
        logging.debug("closest point on any object: " + str(closest_point_on_any_obj))

        #
        if perc_on_target_obj < self.good_view_percentage:
            if closest_point_on_target_obj > self.desired_object_distance:
                if multi_objects_present and (
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
Basic setup
-----------
"""
# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

# Where to find the pretrained model.
model_path = os.path.expanduser(
    "~/tbp/results/monty/pretrained_models/pretrained_ycb_dmc/dist_agent_1lm/pretrained/model.pt"
)

object_names = SHUFFLED_YCB_OBJECTS
test_rotations = get_cube_face_and_corner_views_rotations()

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

# The config dictionary for the evaluation experiment.
view_finder_base = dict(
    # Set up experiment
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,  # load the pre-trained models from this path
        n_eval_epochs=len(test_rotations),
        max_eval_steps=1,
        max_total_steps=1,
    ),
    logging_config=EvalLoggingConfig(
        output_dir=os.path.join(project_dir, "view_finder_base"),
        run_name="eval",
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
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(
            positions=[[0.0, 1.5, -0.2]], rotations=test_rotations
        ),
    ),
    # Doesn't get used, but currently needs to be set anyways.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)

# ----------------------------------------------------------------------------
# 224 x 224
view_finder_224 = copy.deepcopy(view_finder_base)
view_finder_224["logging_config"].output_dir = os.path.join(
    project_dir, "view_finder_224"
)
dataset_args = view_finder_224["dataset_args"]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = [
    [64, 64],
    [224, 224],
]
dataset_args.__post_init__()

# ----------------------------------------------------------------------------
# 256 x 256
view_finder_256 = copy.deepcopy(view_finder_base)
view_finder_256["logging_config"].output_dir = os.path.join(
    project_dir, "view_finder_256"
)
dataset_args = view_finder_256["dataset_args"]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = [
    [64, 64],
    [256, 256],
]
dataset_args.__post_init__()

# ----------------------------------------------------------------------------
# 384 x 384
view_finder_384 = copy.deepcopy(view_finder_base)
view_finder_384["logging_config"].output_dir = os.path.join(
    project_dir, "view_finder_384"
)
dataset_args = view_finder_384["dataset_args"]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = [
    [64, 64],
    [384, 384],
]
dataset_args.__post_init__()

"""
Random rotations
---------
"""
view_finder_base_randrot = copy.deepcopy(view_finder_base)
view_finder_base_randrot["experiment_args"].n_eval_epochs = 14
view_finder_base_randrot["logging_config"].output_dir = os.path.join(
    project_dir, "view_finder_base_randrot"
)
view_finder_base_randrot["eval_dataloader_args"].object_init_sampler = (
    RandomRotationObjectInitializer()
)

# ----------------------------------------------------------------------------
# 224 x 224
view_finder_224_randrot = copy.deepcopy(view_finder_base_randrot)
view_finder_224_randrot["logging_config"].output_dir = os.path.join(
    project_dir, "view_finder_224_randrot"
)
dataset_args = view_finder_224_randrot["dataset_args"]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = [
    [64, 64],
    [224, 224],
]
dataset_args.__post_init__()

# ----------------------------------------------------------------------------
# 256 x 256
view_finder_256_randrot = copy.deepcopy(view_finder_base_randrot)
view_finder_256_randrot["logging_config"].output_dir = os.path.join(
    project_dir, "view_finder_256_randrot"
)
dataset_args = view_finder_256_randrot["dataset_args"]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = [
    [64, 64],
    [256, 256],
]
dataset_args.__post_init__()

# ----------------------------------------------------------------------------
# 384 x 384
view_finder_384_randrot = copy.deepcopy(view_finder_base_randrot)
view_finder_384_randrot["logging_config"].output_dir = os.path.join(
    project_dir, "view_finder_384_randrot"
)
dataset_args = view_finder_384_randrot["dataset_args"]
dataset_args.env_init_args["agents"][0].agent_args["resolutions"] = [
    [64, 64],
    [384, 384],
]
dataset_args.__post_init__()


CONFIGS = {
    "view_finder_base": view_finder_base,
    "view_finder_224": view_finder_224,
    "view_finder_256": view_finder_256,
    "view_finder_384": view_finder_384,
    "view_finder_base_randrot": view_finder_base_randrot,
    "view_finder_224_randrot": view_finder_224_randrot,
    "view_finder_256_randrot": view_finder_256_randrot,
    "view_finder_384_randrot": view_finder_384_randrot,
}
