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
    def __init__(self, root, train=True, size=192, val_frac=0.1, bg_dir=None, bg_prob=0.7, neg_prob=0.15):
        self.root, self.size, self.train = Path(root), size, train
        xyz = np.array(json.load(open(self.root / "training_xyz.json")), dtype=np.float32)
        K = np.array(json.load(open(self.root / "training_K.json")), dtype=np.float32)
        uv = np.einsum("nij,nkj->nki", K, xyz)
        self.uv = uv[..., :2] / uv[..., 2:]

        n_val = int(N_UNIQUE * val_frac)
        ids = np.arange(N_UNIQUE * 4)
        is_val = (ids % N_UNIQUE) >= (N_UNIQUE - n_val)
        self.ids = ids[~is_val] if train else ids[is_val]
        self.bg_prob = bg_prob
        self.neg_prob = neg_prob
        self.bgs = []
        if bg_dir:
            exts = {".jpg", ".jpeg", ".png"}
            self.bgs = [p for p in Path(bg_dir).rglob("*") if p.suffix.lower() in exts]
            print(f"{len(self.bgs)} background images")
        self.aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-45, 45), scale=(0.6, 1.2), translate_percent=0.12, p=0.9),
                A.ColorJitter(0.4, 0.4, 0.4, 0.05, p=0.8),
                A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=7)], p=0.3),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        ) if train else None
        self.photo = A.Compose([
            A.ColorJitter(0.4, 0.4, 0.4, 0.05, p=0.8),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=7)], p=0.3),
        ]) if train else None

    def __len__(self):
        return len(self.ids)

    def _composite(self, img, idx, rng):
        h, w = img.shape[:2]
        mask = cv2.imread(str(self.root / "training" / "mask" / f"{idx % N_UNIQUE:08d}.jpg"), 0)
        bg = cv2.imread(str(self.bgs[rng.integers(len(self.bgs))]))
        if mask is None or bg is None:
            return img
        mask = cv2.resize(mask, (w, h))
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        bh, bw = bg.shape[:2]
        s = int(min(bh, bw) * rng.uniform(0.4, 1.0))
        y0, x0 = rng.integers(0, bh - s + 1), rng.integers(0, bw - s + 1)
        bg = cv2.resize(bg[y0:y0 + s, x0:x0 + s], (w, h))
        if rng.random() < 0.5:
            bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(0.5, 1.5))
        k = int(rng.choice([3, 5, 7]))
        m = cv2.erode((mask > 127).astype(np.float32), np.ones((k, k), np.uint8))
        a = cv2.GaussianBlur(m, (0, 0), rng.uniform(0.8, 2.0))[..., None]
        return (img * a + bg * (1 - a)).astype(np.uint8)

    def _bg_crop(self, rng, lo=0.2):
        bg = None
        for _ in range(5):
            bg = cv2.imread(str(self.bgs[rng.integers(len(self.bgs))]))
            if bg is not None:
                break
        if bg is None:
            return np.zeros((self.size, self. size, 3), np.uint8)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        bh, bw = bg.shape[:2]
        s = max(8, int(min(bh, bw) * rng.uniform(lo, 1.0)))
        y0, x0 = rng.integers(0, bh - s + 1), rng.integers(0, bw - s + 1)
        return cv2.resize(bg[y0:y0 + s, x0:x0 + s], (self.size, self.size))

    def _negative(self, rng):
        img = self._bg_crop(rng)
        mode = rng.random()
        if mode < 0.4:
            k = int(rng.integers(N_UNIQUE))
            mask = cv2.imread(str(self.root / "training" / "mask" / f"{k:08d}.jpg"), 0)
            if mask is not None:
                patch = self._bg_crop(rng)
                mask = cv2.resize(mask, (self.size, self.size))
                ks = int(rng.choice([3, 5, 7]))
                m = cv2.erode((mask > 127).astype(np.float32), np.ones((ks, ks), np.uint8))
                a = cv2.GaussianBlur(m, (0, 0), rng.uniform(0.8, 2.0))[..., None]
                img = (patch * a + img * (1 - a)).astype(np.uint8)
        elif mode < 0.6:
            c0 = c1 = rng.integers(0, 256, 3)
            t = np.linspace(0, 1, self.size, dtype=np.float32)
            t = t[None, :, None] if rng.random() < 0.5 else t[:, None, None]
            img = np.broadcast_to(c0 * (1 - t) + c1 * t, (self.size, self.size, 3))
            img = (img + rng.normal(0, rng.uniform(0, 6), img.shape)).clip(0, 255).astype(np.uint8)

        if rng.random() < 0.5:
            img = img[:, ::-1]
        img = np.ascontiguousarray(img)
        if self.photo:
            img = self.photo(image=img)["image"]
        img = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.zeros(21, 2), torch.tensor(0.0)

    def __getitem__(self, i):
        idx = int(self.ids[i])
        rng = np.random.default_rng() if self.train else np.random.default_rng(idx)
        if self.bgs and rng.random() < self.neg_prob:
            return self._negative(rng)
        img = cv2.imread(str(self.root / "training" / "rgb" / f"{idx:08d}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # type: ignore
        if self.bgs and rng.random() < self.bg_prob:
            img = self._composite(img, idx, rng)
        img = cv2.resize(img, (self.size, self.size))
        kp = self.uv[idx % N_UNIQUE] * (self.size / 224.0)

        if self.aug:
            for _ in range(5):
                out = self.aug(image=img, keypoints=kp)
                k2 = np.array(out["keypoints"], dtype=np.float32)
                if (k2 >= 0).all() and (k2 <= self.size - 1).all():
                    break
            img, kp = out["image"], k2

        kp = np.clip(kp, 0, self.size - 1)
        img = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.from_numpy(kp.astype(np.float32)), torch.tensor(1.0)
