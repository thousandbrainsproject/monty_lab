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
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class SimpleNet(nn.Module):
    def __init__(self, num_classes=77, hidden_size=768):
        super().__init__()
        
        # Patch embedding-like input layer
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=32, p2=32)
        self.input_layer = nn.Sequential(
            nn.LayerNorm(4 * 32 * 32),
            nn.Linear(4 * 32 * 32, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Hidden layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Classification head
        self.classification_head = nn.Linear(hidden_size, num_classes)
        
        # Quaternion head for rotation
        self.quaternion_head = nn.Linear(hidden_size, 4)

    def forward(self, x):
        # Reshape input
        x = self.rearrange(x)
        
        # Input layer
        x = self.input_layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Hidden layer
        x = self.hidden_layer(x)
        
        # Classification head
        pred_class = self.classification_head(x)
        
        # Quaternion head
        pred_quaternion = self.quaternion_head(x)
        pred_quaternion = F.normalize(pred_quaternion, p=2, dim=1)
        
        return pred_class, pred_quaternion

