import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

SIZE, K = 192, 2
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FINGERS = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
COLORS = [(0,0,255), (0,165,255), (0,255,255), (255,255,0), (255,0,255)]

def crop(frame, cx, cy, side):
    s = SIZE / side
    M = np.array([[s, 0, -(cx - side / 2) * s], [0, s, -(cy - side / 2) * s]], np.float32)
    return cv2.warpAffine(frame, M, (SIZE, SIZE))

def preprocess(patch):
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]

def predict(sess, frame, box):
    cx, cy, side = box
    ox, oy = sess.run(None, {"image": preprocess(crop(frame, cx, cy, side))})
    kp = np.stack([ox[0].argmax(-1), oy[0].argmax(-1)], -1) / K
    return kp * (side / SIZE) + np.array([cx - side / 2, cy - side / 2])

def next_box(kp, frame_shape, margin=1.5, min_side=64):
    lo, hi = kp.min(0), kp.max(0)
    c = (lo + hi) / 2
    side = float(np.clip((hi - lo).max() * margin, min_side, min(frame_shape[:2]) * 1.5))
    return c[0], c[1], side

def center_box(shape):
    h, w = shape[:2]
    return w / 2, h / 2, 0.6 * min(h, w)

def draw(frame, kp, box):
    cx, cy, side = box
    p0 = (int(cx - side / 2), int(cy - side / 2))
    cv2.rectangle(frame, p0, (int(cx + side / 2), int(cy + side / 2)), (255, 255, 255), 1)
    pts = kp.astype(int)
    for chain, col in zip(FINGERS, COLORS):
        for a, b in zip(chain, chain[1:]):
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), col, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    sess = ort.InferenceSession("hand.onnx", providers=["CPUExecutionProvider"])
    box, ms = None, 0.0
    track, flip = True, False
    while True:
        ok, frame = cap.read()
        if flip:
            frame = cv2.flip(frame, 1)
        if not ok:
            break
        if box is None:
            box = center_box(frame.shape)
        t = time.perf_counter()
        kp = predict(sess, frame, box)
        ms = 0.9 * ms + 0.1 * (time.perf_counter() - t) * 1000
        draw(frame, kp, box)
        box = next_box(kp, frame.shape) if track else center_box(frame.shape)
        cv2.putText(frame, f"{ms:.1f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("hand", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            box = None
        if key == ord("t"):
            track = not track
        if key == ord("f"):
            flip = not flip
    cap.release()
    cv2.destroyAllWindows()

main()
