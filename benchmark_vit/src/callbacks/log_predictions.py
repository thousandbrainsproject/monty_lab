# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import wandb
import torch
import numpy as np
from lightning import Callback
from torchvision.utils import make_grid
from src.metrics.rotation_error import get_rotation_error
class LogPredictionsCallback(Callback):
    def __init__(self):
        super().__init__()

    def _log_predictions(self, trainer, pl_module, stage):
        device = pl_module.device
        # Get the appropriate dataloader
        dataloader = (trainer.datamodule.val_dataloader() if stage == "val" 
                      else trainer.datamodule.test_dataloader())
        label_encoder = trainer.datamodule.dataset.get_label_encoder()

        # Create a wandb Table
        columns = [
            "Epoch", "Object Image", "Object Name",
            "Predicted Object Type", "Ground Truth Quaternion", 
            "Predicted Quaternion", "Rotation Error (Radians)",
            "Rotation Error (Degrees)"
        ]
        table = wandb.Table(columns=columns)

        for batch in dataloader:
            rgbd_images, object_ids, unit_quaternions = batch

            # Get model predictions
            with torch.no_grad():
                pred_class, pred_unit_quaternions = pl_module(rgbd_images.to(device))
                pred_class = pred_class.argmax(dim=1)

            # Get object name of pred_class using label
            pred_object_names = [str(label_encoder.inverse_transform([pred_class[i].item()])[0]) for i in range(len(pred_class))] 
            gt_object_names = [str(label_encoder.inverse_transform([object_ids[i].item()])[0]) for i in range(len(object_ids))]

            # Add rows to the table
            for i in range(len(rgbd_images)):
                image = rgbd_images[i].cpu() # (4, 224, 224)
                rgb_image = image[:3, :, :]
                gt_object_name = gt_object_names[i]
                gt_quaternion = unit_quaternions[i].cpu().numpy()
                pred_object_name = pred_object_names[i]
                pred_quaternion = pred_unit_quaternions[i].cpu().numpy()
                rotation_error_radians = get_rotation_error(pred_quaternion, gt_quaternion).item()
                rotation_error_degrees = rotation_error_radians * 180 / np.pi

                # Convert image tensor to wandb Image
                image_wandb = wandb.Image(make_grid(rgb_image, normalize=True, scale_each=True))

                # Add row to the table
                table.add_data(
                    trainer.current_epoch, image_wandb, gt_object_name,
                    pred_object_name, str(gt_quaternion), str(pred_quaternion),
                    str(rotation_error_radians), str(rotation_error_degrees)
                )

        # Log the table to wandb
        trainer.logger.experiment.log({f"{stage}_predictions_table": table})

    def on_validation_end(self, trainer, pl_module):
        # Log predictions only at the last epoch
        if trainer.current_epoch == (trainer.max_epochs - 1):
            self._log_predictions(trainer, pl_module, "val")

    def on_test_end(self, trainer, pl_module):
        self._log_predictions(trainer, pl_module, "test")
