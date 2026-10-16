"""Core logic for SafeFall AI.

No Streamlit code in this file, so it is easy to test.
Pipeline: frame -> MediaPipe pose (33 keypoints) -> 68 features -> small neural network
(numpy only, weights saved from the Keras model) -> activity + confidence.
"""
import cv2
import numpy as np

CLASS_NAMES = ["fall", "walking", "sitting", "standing", "normal"]
DISPLAY = {
    "fall": "Fall Detected",
    "walking": "Walking",
    "sitting": "Sitting",
    "standing": "Standing",
    "normal": "Normal Activity",
}

# If the training frames were saved at a fixed size, put (width, height) here so the app
# resizes every uploaded frame in the same way. None = keep the original shape.
FRAME_SIZE = None
MAX_SIDE = 960  # big images are scaled down to this size for speed

CONN = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32)]


def load_weights(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def make_features(xs, ys):
    """Same 68 features as in training: centred on the hips, scaled by torso length,
    plus torso angle and body width/height ratio."""
    hip = np.array([(xs[23] + xs[24]) / 2, (ys[23] + ys[24]) / 2])
    sh = np.array([(xs[11] + xs[12]) / 2, (ys[11] + ys[12]) / 2])
    torso = max(np.linalg.norm(sh - hip), 0.02)
    nx = (xs - hip[0]) / torso
    ny = (ys - hip[1]) / torso
    dx, dy = abs(sh[0] - hip[0]), abs(sh[1] - hip[1])
    torso_angle = np.degrees(np.arctan2(dx, dy))
    aspect = (xs.max() - xs.min()) / (ys.max() - ys.min() + 1e-6)
    return np.concatenate([nx, ny, [torso_angle, aspect]])


def forward(feat, W):
    """Neural network forward pass with numpy (no TensorFlow needed)."""
    z = (feat - W["mean"]) / W["scale"]
    h1 = np.maximum(z @ W["W1"] + W["b1"], 0)
    h2 = np.maximum(h1 @ W["W2"] + W["b2"], 0)
    logits = h2 @ W["W3"] + W["b3"]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def prepare_frame(bgr):
    if FRAME_SIZE is not None:
        return cv2.resize(bgr, FRAME_SIZE)
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m > MAX_SIDE:
        s = MAX_SIDE / m
        return cv2.resize(bgr, (int(w * s), int(h * s)))
    return bgr


def analyse_frame(bgr, landmarker, W):
    """Return a dict with the prediction, or None if no person was found."""
    import mediapipe as mp

    bgr = prepare_frame(bgr)
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks[0]
    xs = np.array([p.x for p in lm], dtype=float)
    ys = np.array([p.y for p in lm], dtype=float)
    feat = make_features(xs, ys)
    probs = forward(feat, W)
    k = int(np.argmax(probs))
    torso_angle = float(feat[-2])
    aspect_ratio = float(feat[-1])
    return {"label": CLASS_NAMES[k], "conf": float(probs[k]), "probs": probs,
            "fall_prob": float(probs[0]), "xs": xs, "ys": ys, "bgr": bgr,
            "torso_angle": torso_angle, "aspect_ratio": aspect_ratio}


def draw_skeleton(bgr, xs, ys, is_fall=False):
    out = bgr.copy()
    h, w = out.shape[:2]
    line_col = (0, 0, 255) if is_fall else (0, 200, 0)  # BGR
    pts = [(int(x * w), int(y * h)) for x, y in zip(xs, ys)]
    for a, b in CONN:
        cv2.line(out, pts[a], pts[b], line_col, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(out, p, 4, (255, 120, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p, 2, (255, 255, 255), -1, cv2.LINE_AA)
    return out


def is_fall_result(r, thr):
    return r is not None and r["label"] == "fall" and r["conf"] >= thr


def find_alert(items, thr, n_consecutive):
    """Index (in items) where the first alert run starts, or None.
    An alert needs `n_consecutive` fall predictions in a row. Frames with no person
    found are skipped (they neither count nor break the run)."""
    run, start = 0, None
    for i, it in enumerate(items):
        r = it["result"]
        if r is None:
            continue
        if is_fall_result(r, thr):
            if run == 0:
                start = i
            run += 1
            if run >= n_consecutive:
                return start
        else:
            run, start = 0, None
    return None


def process_video(path, landmarker, W, step=10, max_frames=120, progress_cb=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    expected = max_frames if total <= 0 else min(max_frames, -(-total // step))
    items, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            items.append({"frame": idx, "time": idx / fps,
                          "result": analyse_frame(frame, landmarker, W)})
            if progress_cb:
                progress_cb(min(len(items) / max(expected, 1), 1.0))
            if len(items) >= max_frames:
                break
        idx += 1
    cap.release()
    return items, fps


def summarise(items, thr):
    counts = {n: 0 for n in CLASS_NAMES}
    confs, falls = [], 0
    for it in items:
        r = it["result"]
        if r is None:
            continue
        counts[r["label"]] += 1
        confs.append(r["conf"])
        falls += int(is_fall_result(r, thr))
    detected = sum(counts.values())
    return {"analysed": len(items), "detected": detected, "counts": counts,
            "falls": falls, "normal": detected - falls,
            "mean_conf": float(np.mean(confs)) if confs else 0.0}
