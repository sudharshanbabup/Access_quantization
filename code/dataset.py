"""
SynthShapes-32: a controlled synthetic image-classification benchmark with
ground-truth object masks. Each 32x32 RGB image contains exactly one shape
drawn from six classes on a textured, noisy background. Because the object
support (mask) is known by construction, this benchmark permits *direct*
measurement of explanation localization quality against ground truth, in
addition to the model-vs-model fidelity study that is our primary object.

The design deliberately mirrors the statistical regime of small natural-image
benchmarks (e.g. CIFAR-10): low resolution, coloured textures, class-dependent
shape priors, and additive sensor-like noise.
"""
import numpy as np

CLASSES = ["disk", "square", "triangle", "cross", "ring", "bar"]
NUM_CLASSES = len(CLASSES)


def _coords(H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return yy, xx


def _draw_shape(canvas, mask, cls, rng, H, W):
    """Render shape `cls` onto canvas (HxWx3) and mask (HxW) in place."""
    yy, xx = _coords(H, W)
    cy = rng.uniform(0.30 * H, 0.70 * H)
    cx = rng.uniform(0.30 * W, 0.70 * W)
    r = rng.uniform(0.16 * H, 0.26 * H)
    # class-conditioned colour prior (with overlap so the task is non-trivial)
    base = np.array([
        [0.90, 0.30, 0.30],  # disk    - reddish
        [0.30, 0.55, 0.90],  # square  - blue
        [0.35, 0.80, 0.40],  # triangle- green
        [0.90, 0.75, 0.25],  # cross   - amber
        [0.70, 0.40, 0.85],  # ring    - purple
        [0.85, 0.55, 0.30],  # bar     - orange
    ], dtype=np.float32)[cls]
    colour = np.clip(base + rng.normal(0, 0.12, size=3), 0.05, 0.95).astype(np.float32)

    if cls == 0:      # disk
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    elif cls == 1:    # square
        m = (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
    elif cls == 2:    # triangle (downward)
        m = (yy - (cy - r) >= 0) & \
            ((xx - cx) <= (r - (yy - (cy - r)))) & \
            (-(xx - cx) <= (r - (yy - (cy - r)))) & \
            (yy <= cy + r)
    elif cls == 3:    # cross
        arm = 0.38 * r
        m = ((np.abs(yy - cy) <= arm) & (np.abs(xx - cx) <= r)) | \
            ((np.abs(xx - cx) <= arm) & (np.abs(yy - cy) <= r))
    elif cls == 4:    # ring (annulus)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        m = (d2 <= r ** 2) & (d2 >= (0.55 * r) ** 2)
    else:             # bar (thick rotated-ish rectangle)
        m = (np.abs(yy - cy) <= 0.35 * r) & (np.abs(xx - cx) <= r)

    m = m.astype(np.float32)
    for c in range(3):
        canvas[..., c] = canvas[..., c] * (1 - m) + colour[c] * m
    mask[:] = np.maximum(mask, m)


def make_dataset(n, seed=0, H=32, W=32, noise=0.12):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, H, W, 3), dtype=np.float32)
    Y = np.zeros((n,), dtype=np.int64)
    M = np.zeros((n, H, W), dtype=np.float32)
    for i in range(n):
        cls = int(rng.integers(0, NUM_CLASSES))
        # textured low-frequency background
        bg_c = rng.uniform(0.15, 0.55, size=3).astype(np.float32)
        canvas = np.ones((H, W, 3), dtype=np.float32) * bg_c
        fx, fy = rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0)
        yy, xx = _coords(H, W)
        tex = 0.10 * np.sin(2 * np.pi * (fx * xx / W + fy * yy / H) +
                            rng.uniform(0, 2 * np.pi))
        canvas += tex[..., None]
        mask = np.zeros((H, W), dtype=np.float32)
        _draw_shape(canvas, mask, cls, rng, H, W)
        canvas += rng.normal(0, noise, size=canvas.shape).astype(np.float32)
        X[i] = np.clip(canvas, 0.0, 1.0)
        Y[i] = cls
        M[i] = mask
    # channel-wise normalisation statistics returned for reproducibility
    return X, Y, M


def normalise(X, mean=None, std=None):
    if mean is None:
        mean = X.reshape(-1, 3).mean(0)
        std = X.reshape(-1, 3).std(0) + 1e-6
    Xn = (X - mean) / std
    return Xn.astype(np.float32), mean, std


if __name__ == "__main__":
    X, Y, M = make_dataset(12, seed=1)
    print("X", X.shape, "Y", Y.shape, "M", M.shape,
          "class balance", np.bincount(Y))
    print("mask coverage mean", M.mean())
