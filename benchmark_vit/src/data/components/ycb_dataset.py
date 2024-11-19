# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

class YCBDataset(Dataset):
    def __init__(self, data_dir: str):
        """YCB Dataset for ViewFinder iamges per episode with rotation information. 

        Args:
            data_dir (str): Path to the directory containing the data. Example: `../data/view_finder_224/eval/view_finder_rgbd`. 
                            It should have a images subdirectory and episodes.json file.
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"

        with open(self.data_dir / "episodes.json", "r") as f:
            self.episodes = [json.loads(line) for line in f]

        self.unique_object_names = self._get_unique_object_names()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.unique_object_names)

    @property
    def object_classes(self):
        return self.unique_object_names
    
    def get_label_encoder(self):
        return self.label_encoder
    
    def __len__(self):
        return len(os.listdir(self.image_dir))
    
    def _get_unique_object_names(self):
        """
        Get unique object names from episodes.json file.

        Each row in the episodes.json file is a dictionary and has "object" key, e.g.
        {"episode": 2456, "object": "mustard_bottle", "rotation": [90, 270, 270]}
        """
        unique_object_names = set()
        for episode in self.episodes:
            unique_object_names.add(episode["object"])
        
        return list(unique_object_names)
    
    def extract_object_id(self, idx: int) -> int:
        """Extract object id from episodes.json file.

        Args:
            idx (int): Index.

        Returns:
            int: Object id.
        """
        episode = self.episodes[idx]
        object_name = episode["object"]
        return self.label_encoder.transform([object_name])[0]

    def extract_rotation(self, idx: int) -> torch.Tensor:
        """Extract rotation from episodes.json file.

        Args:
            file_path (str): File path.

        Returns:
            torch.Tensor: Rotation.
        """
        episode = self.episodes[idx]
        rotation = episode["rotation"]

        return torch.tensor(rotation, dtype=torch.float32)
    
    def normalize_rotation_to_unit_quaternion(self, rotation: torch.Tensor) -> torch.Tensor:
        """Normalize rotation to unit quaternion.
        Args:
            rotation (torch.Tensor): Rotation in degrees.

        Returns:
            torch.Tensor: Unit quaternion.
        """
        # Convert degrees to radians
        rotation_rad = np.radians(rotation.numpy())
        
        # Create a rotation object from Euler angles (assuming XYZ order)
        r = R.from_euler('xyz', rotation_rad, degrees=False)
        
        # Convert to quaternion
        quaternion = r.as_quat()
        
        # Convert back to torch tensor
        return torch.tensor(quaternion, dtype=torch.float32)

    def __getitem__(self, idx):
        rgbd_image = np.load(self.image_dir / f"{idx}.npy")
        rgbd_image = torch.tensor(rgbd_image, dtype=torch.float32)
        rgbd_image = rgbd_image.permute(2, 0, 1)

        object_id = self.extract_object_id(idx)
        object_id = torch.tensor(object_id, dtype=torch.int64)
        euler_rotation = self.extract_rotation(idx)
        unit_quaternion = self.normalize_rotation_to_unit_quaternion(euler_rotation)

        return rgbd_image, object_id, unit_quaternion
