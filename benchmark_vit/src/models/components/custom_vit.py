# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import torch
import torch.nn.functional as F
from torch import nn

from transformers import ViTModel, AutoConfig
from einops.layers.torch import Rearrange

MODEL_DICT = {
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",
    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",

}

class CustomPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=32, p2=32)
        self.norm1 = nn.LayerNorm(4 * 32 * 32)
        self.projection = nn.Linear(4 * 32 * 32, config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x, interpolate_pos_encoding=False):
        x = self.rearrange(x)
        x = self.norm1(x)
        x = self.projection(x)
        x = self.norm2(x)
        return x

class ViT(nn.Module):
    # TODO: Test differently sized images (224, 256, 384)
    # TODO: Test with inverted Depth (0 in background, closer to 1 for object)
    def __init__(self, *, model_name, num_classes=77):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes  
        # Initialize model
        try:
            model_path = MODEL_DICT[self.model_name]
        except KeyError:
            raise ValueError(f"Model {model_name} not found in MODEL_DICT")

        # Load configuration and update num_channels
        config = AutoConfig.from_pretrained(model_path)
        config.num_channels = 4  # RGB-D input, 4 channels

        self.model = ViTModel.from_pretrained(
            model_path, 
            config=config, 
            ignore_mismatched_sizes=True
        )

        # Replace patch embeddings to handle 4 channels
        # self.model.embeddings.patch_embeddings.projection = nn.Conv2d(
        #     in_channels=4,  # RGBD has 4 channels
        #     out_channels=self.model.config.hidden_size,
        #     kernel_size=self.model.embeddings.patch_embeddings.projection.kernel_size,
        #     stride=self.model.embeddings.patch_embeddings.projection.stride
        # )
        self.model.embeddings.patch_embeddings = CustomPatchEmbeddings(self.model.config)
        # Adjust position embeddings for different patch sizes
        num_patches = (224 // 32) * (224 // 32)  # Adjust based on your input size
        self.model.embeddings.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, self.model.config.hidden_size)
        )

        # Reset the position embedding interpolation flag if it exists
        if hasattr(self.model.embeddings, 'num_patches'):
            self.model.embeddings.num_patches = num_patches

        # Classification and quaternion heads
        self.classification_head = nn.Linear(self.model.config.hidden_size, num_classes)
        self.quaternion_head = nn.Linear(self.model.config.hidden_size, 4)  # quaternion has 4 parameters

        self.freeze_pretrained_weights()
    
    def freeze_pretrained_weights(self):
        # Freeze all pretrained weights, except for the new patch_embeddings and heads
        for name, param in self.model.named_parameters():
            if 'patch_embeddings' not in name:
                param.requires_grad = False
        
        # Ensure the classification and rotation heads remain trainable
        for param in self.classification_head.parameters():
            param.requires_grad = True
        for param in self.quaternion_head.parameters():
            param.requires_grad = True

    def forward(self, img):
        # Embeds
        # embeds = self.to_patch_embedding(img)

        # Use the pretrained ViT model
        outputs = self.model(img)
        
        # Get the pooled output (CLS token)
        pooled_output = outputs.last_hidden_state[:, 0]

        # Predict class
        pred_class = self.classification_head(pooled_output)
        
        # Predict quaternion
        pred_quaternion = self.quaternion_head(pooled_output)

        # Normalize quaternion
        pred_quaternion = F.normalize(pred_quaternion, p=2, dim=1)

        return pred_class, pred_quaternion








