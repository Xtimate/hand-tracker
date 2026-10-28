import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

ACQUIRE = 0.5
KEEP = 0.3
SIZE, K = 192, 2
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FINGERS = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
COLORS = [(0,0,255), (0,165,255), (0,255,255), (255,255,0), (255,0,255)]


def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def decode_kp(ox, oy, box):
    cx, cy, side = box
    kp = np.stack([ox.argmax(-1), oy.argmax(-1)], -1) / K
    return kp * (side / SIZE) + np.array([cx - side / 2, cy - side / 2])

def crop(frame, cx, cy, side):
    s = SIZE / side
    M = np.array([[s, 0, -(cx - side / 2) * s], [0, s, -(cy - side / 2) * s]], np.float32)
    return cv2.warpAffine(frame, M, (SIZE, SIZE))

def preprocess(patch):
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]

def predict(sess, frame, box):
    ox, oy, pl = sess.run(None, {"image": preprocess(crop(frame, *box))})
    return decode_kp(ox[0], oy[0], box), sigmoid(float(pl[0, 0]))

def scan(sess, frame):
    h, w = frame.shape[:2]
    boxes = []
    for frac in (0.45, 0.7, 1.0):
        side = frac * min(h, w)
        xs = np.linspace(side / 2, w - side / 2, 3)
        ys = np.linspace(side / 2, h - side / 2, 3)
        for y in ys:
            for x in xs:
                b = (float(x), float(y), float(side))
                if b not in boxes:
                    boxes.append(b)
    batch = np.concatenate([preprocess(crop(frame, *b)) for b in boxes])
    ox, oy, pl = sess.run(None, {"image": batch})
    i = int(pl[:, 0].argmax())
    return boxes[i], decode_kp(ox[i], oy[i], boxes[i]), sigmoid(float(pl[i, 0]))
class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1 / (1 + (1 / (2 * np.pi * cutoff)) / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t,  1e-3)
        a = self._alpha(self.dc, dt)
        self.dx = a * (x - self.x) / dt + (1 - a) * self.dx
        a = self._alpha(self.mc + self.beta * np.abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x

    def reset(self):
        self.x = None

def refine(sess, frame, box, iters=2):
    kp, p = predict(sess, frame, box)
    for _ in range(iters):
        box = box_from_kp(kp, frame.shape)
        kp, p = predict(sess, frame, box)
    return kp, p, box

class Tracker:
    def __init__(self, sess):
        self.sess, self.filt = sess, OneEuro()
        self.box, self.lost = None, 0

    def reset(self):
        self.box, self.lost = None, 0
        self.filt.reset()

    def step(self, frame, now):
        kp, p = None, 0.0
        if self.box is not None:
            kp, p, self.box = refine(self.sess, frame, self.box)
        self.lost = self.lost + 1 if p < KEEP else 0
        if self.box is None or self.lost >= 3:
            b2, kp2, k2 = scan(self.sess, frame)
            if p2 >= ACQUIRE:
                kp, p, self.box = refine(self.sess, frame, b2)
                self.lost = 0
                self.filt.reset()
            else:
                kp, self.box = None, None
        if kp is None:
            return None, p, None
        used = self.box
        nb = box_from_kp(kp, frame.shape)
        self.box = (nb[0], nb[1], 0.8 * used[2] + 0.2 * nb[2])
        return self.filt(kp, now), p, used
def box_from_kp(kp, frame_shape, margin=2.6, min_frac=.3):
    lo, hi = kp.min(0), kp.max(0)
    c = (lo + hi) / 2
    short = min(frame_shape[:2])
    side = float(np.clip((hi - lo).max() * margin, min_frac * short, 1.5 * short))
    return float(c[0]), float(c[1]), side


def center_box(shape):
    h, w = shape[:2]
    return w / 2, h / 2, 0.9 * min(h, w)

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
    sess = ort.InferenceSession(sys.argv[2] if len(sys.argv) > 2 else "hand.onnx", providers=["CPUExecutionProvider"])
    tracker = Tracker(sess)
    track, flip = True, False
    while True:
        ok, frame = cap.read()
        if flip:
            frame = cv2.flip(frame, 1)
        if not ok:
            break
        now = time.perf_counter()
        if track:
            kp, p, bx = tracker.step(frame, now)
        else:
            bx = center_box(frame.shape)
            kp, p = predict(sess, frame, bx)
        ms = 0.9 * ms + 0.1 * (time.perf_counter() - now) * 1000
        if kp is not None:
            draw(frame, kp, bx)
            cv2.putText(frame, f"p={p:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "searching", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"{ms:.1f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("hand", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            box = None
            tracker.reset()
        if key == ord("t"):
            track = not track
        if key == ord("f"):
            flip = not flip
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
