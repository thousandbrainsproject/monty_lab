# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import Accuracy, MeanMetric
from src.metrics.rotation_error import get_rotation_error
from src.losses.loss import quaternion_geodesic_loss

class ViTLitModule(LightningModule):
    """ViT model for 6DoF object pose estimation. 

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        rotation_weight: float
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to finetune.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.classification_loss = torch.nn.CrossEntropyLoss()
        self.quaternion_geodesic_loss = quaternion_geodesic_loss

        # Create metric objects
        self.train_metrics = self.create_metrics("train")
        self.val_metrics = self.create_metrics("val")
        self.test_metrics = self.create_metrics("test")

    def create_metrics(self, prefix):
        return {
            f"{prefix}/loss": MeanMetric(),
            f"{prefix}/classification_loss": MeanMetric(),
            f"{prefix}/quaternion_geodesic_loss": MeanMetric(),
            f"{prefix}/class_acc": Accuracy(task="multiclass", num_classes=77),
            f"{prefix}/rotation_error": MeanMetric()
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for metric in self.val_metrics.values():
            metric.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of object ids.
            - A tensor of rotations.
        """
        rgbd_image, object_id, unit_quaternion = batch
        pred_class, pred_quaternion = self.forward(rgbd_image)
        classification_loss = self.classification_loss(pred_class, object_id)
        quaternion_geodesic_loss = self.quaternion_geodesic_loss(pred_quaternion, unit_quaternion)
        # TODO: Add lambda to balance classification and rotation loss
        loss = classification_loss + self.hparams.rotation_weight * quaternion_geodesic_loss

        return loss, classification_loss, quaternion_geodesic_loss, pred_class, pred_quaternion, object_id, unit_quaternion

    def log_metrics(self, prefix: str, loss: torch.Tensor, classification_loss: torch.Tensor,
                    quaternion_geodesic_loss: torch.Tensor, pred_class: torch.Tensor,
                    pred_quaternion: torch.Tensor, object_id: torch.Tensor, unit_quaternion: torch.Tensor):
        """Log metrics for a given prefix (train, val, test)."""
        metrics = self.train_metrics if prefix == "train" else self.val_metrics if prefix == "val" else self.test_metrics
        
        rotation_error = get_rotation_error(pred_quaternion, unit_quaternion)
        
        # Move metrics to the correct device
        for metric in metrics.values():
            metric.to(self.device)

        # Update all metrics
        metrics[f"{prefix}/loss"](loss)
        metrics[f"{prefix}/classification_loss"](classification_loss)
        metrics[f"{prefix}/quaternion_geodesic_loss"](quaternion_geodesic_loss)
        metrics[f"{prefix}/class_acc"](pred_class, object_id)
        metrics[f"{prefix}/rotation_error"](rotation_error)
        
        # Log all metrics
        self.log_dict({name: metric.compute() for name, metric in metrics.items()},
                      on_step=False, on_epoch=True, prog_bar=True)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, classification_loss, quaternion_geodesic_loss, pred_class, pred_quaternion, object_id, unit_quaternion = self.model_step(batch)
        self.log_metrics("train", loss, classification_loss, quaternion_geodesic_loss, pred_class, pred_quaternion, object_id, unit_quaternion)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        val_loss, classification_loss, quaternion_geodesic_loss, pred_class, pred_quaternion, object_id, unit_quaternion = self.model_step(batch)
        self.log_metrics("val", val_loss, classification_loss, quaternion_geodesic_loss, pred_class, pred_quaternion, object_id, unit_quaternion)


    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, classification_loss, quaternion_geodesic_loss, pred_class, pred_quaternion, object_id, unit_quaternion = self.model_step(batch)  
        self.log_metrics("test", loss, classification_loss, quaternion_geodesic_loss, pred_class, pred_quaternion, object_id, unit_quaternion)


    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
