import sys

import cv2
import numpy as np

from dataset import FreiHAND, MEAN, STD
from infer import FINGERS, COLORS

ds = FreiHAND(sys.argv[1], train=True, bg_dir=sys.argv[2], bg_prob=1.0)
tiles = []
for i in np.random.choice(len(ds), 16, replace=False):
    img, kp = ds[int(i)][:2]
    img = (img.numpy().transpose(1, 2, 0) * STD + MEAN) * 255
    img = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=2, fy=2)
    pts = (kp.numpy() * 2).astype(int)
    for chain, col in zip(FINGERS, COLORS):
        for a, b in zip(chain, chain[1:]):
            cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), col, 2)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)
    tiles.append(img)

grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
cv2.imwrite("viz_comp.png", grid)
