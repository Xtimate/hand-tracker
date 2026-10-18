import cv2
from mpmath import identify
import numpy as np
from dataset import FreiHAND, MEAN, STD

FINGERS = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
COLORS = [(0,0,255), (0,165,255), (0,255,0), (255,255,0), (255,0,255)]

ds = FreiHAND("/home/xtimate/projects/hand-tracker/data", train=True)
tiles = []
for i in np.random.choice(len(ds), 8, replace=False):
    img, kp = ds[int(i)]
    img = (img.numpy().transpose(1, 2, 0) * STD * MEAN) * 255
    img = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
    kp = kp.numpy().astype(int)
    for chain, col in zip(FINGERS, COLORS):
        for a, b in zip(chain, chain[1:]):
            cv2.line(img, tuple(kp[a]), tuple(kp[b]), col, 2)
        for x, y in kp:
            cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
        tiles.append(img)

cols = 8
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
cv2.imwrite("viz_check.png", grid)
