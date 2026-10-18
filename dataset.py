import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

N_UNIQUE = 32560
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.255], dtype=np.float32)

class FreiHAND(Dataset):
    def __init__(self, root, train=True, size=192, val_frac=0.1):
        self.root, self.size, self.train = Path(root), size, train
        xyz = np.array(json.load(open(self.root / "training_xyz.json")), dtype=np.float32)
        K = np.array(json.load(open(self.root / "training_K.json")), dtype=np.float32)
        uv = np.einsum("nij,nkj->nki", K, xyz)
        self.uv = uv[..., :2] / uv[..., 2:]

        n_val = int(N_UNIQUE * val_frac)
        ids = np.arange(N_UNIQUE * 4)
        is_val = (ids % N_UNIQUE) >= (N_UNIQUE - n_val)
        self.ids = ids[~is_val] if train else ids[is_val]

        self.aug = A.Compose(
            [
                A.Affine(rotate=(-45, 45), scale=(0.8, 1.2), translate_percent=0.05, p=0.9),
                A.ColorJitter(0.3, 0.3, 0.3, 0.05, p=0.8),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        ) if train else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = int(self.ids[i])
        img = cv2.imread(str(self.root / "training" / "rgb" / f"{idx:08d}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        kp = self.uv[idx % N_UNIQUE] * (self.size / 224.0)

        if self.aug:
            out = self.aug(image=img, keypoints=kp)
            img, kp = out["image"], np.array(out["keypoints"], dtype=np.float32)

        kp = np.clip(kp, 0, self.size - 1)
        img = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.from_numpy(kp.astype(np.float32))
