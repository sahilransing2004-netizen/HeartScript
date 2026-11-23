"""Photographic degradations that turn a clean render into something phone-photo-like."""
import cv2
import numpy as np

SEVERITY_NAMES = {0: "none", 1: "mild", 2: "moderate", 3: "severe"}

PARAMS = {
    1: dict(rot=4, persp=0.03, blur=1.0, motion=0, noise=4, illum=0.15, cast=0.04,
            jpeg=(60, 95), scale=(0.90, 1.00), down=(0.80, 1.0), gray_p=0.2),
    2: dict(rot=10, persp=0.07, blur=1.8, motion=5, noise=8, illum=0.35, cast=0.08,
            jpeg=(40, 85), scale=(0.80, 1.00), down=(0.60, 1.0), gray_p=0.4),
    3: dict(rot=20, persp=0.12, blur=2.8, motion=9, noise=14, illum=0.55, cast=0.12,
            jpeg=(25, 75), scale=(0.70, 0.95), down=(0.45, 0.9), gray_p=0.5),
}


def _warp(img, p, rng):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    s = rng.uniform(*p["scale"])
    c = np.float32([w / 2, h / 2])
    ang = np.deg2rad(rng.uniform(-p["rot"], p["rot"]))
    R = np.float32([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = ((src - c) * s) @ R.T + c
    shift = np.float32([(1 - s) * w * 0.5 + 0.02 * w, (1 - s) * h * 0.5 + 0.02 * h])
    dst = dst + rng.uniform(-1, 1, 2).astype(np.float32) * shift
    dst = dst + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * np.float32([w, h]) * p["persp"]
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    bg = tuple(int(x) for x in rng.integers(20, 230, 3))
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def _illumination(img, p, rng):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xx / w - 0.5, yy / h - 0.5
    th = rng.uniform(0, 2 * np.pi)
    grad = np.cos(th) * xn + np.sin(th) * yn
    field = 1 + p["illum"] * rng.uniform(0.3, 1.0) * grad * 1.4 * rng.choice([-1.0, 1.0])
    if rng.random() < 0.5:
        cx, cy = rng.uniform(-0.4, 0.4, 2)
        sg = rng.uniform(0.15, 0.4)
        blob = np.exp(-((xn - cx) ** 2 + (yn - cy) ** 2) / (2 * sg ** 2))
        field = field * (1 - p["illum"] * rng.uniform(0.3, 1.0) * blob)
    field = np.clip(field, 0.25, 1.4).astype(np.float32)
    return img.astype(np.float32) * field[..., None]


def _motion_blur(img, k, rng):
    k = int(k) | 1
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    kern /= max(kern.sum(), 1e-6)
    return cv2.filter2D(img, -1, kern)


def degrade(img, rng, severity=2, out_size=None, allow_rot90=False):
    """img: BGR uint8. severity: 0 (none) .. 3 (severe). out_size = (width, height).
    Deterministic given rng."""
    severity = int(severity)
    if severity <= 0:
        out = img
    else:
        p = PARAMS[severity]
        out = _warp(img, p, rng)
        if allow_rot90 and rng.random() < 0.25:
            out = np.ascontiguousarray(np.rot90(out, int(rng.integers(1, 4))))
        f = _illumination(out, p, rng)
        gains = 1 + rng.uniform(-p["cast"], p["cast"], 3).astype(np.float32)
        f = np.clip(f * gains, 0, 255).astype(np.uint8)
        sig = rng.uniform(0, p["blur"])
        if sig > 0.3:
            f = cv2.GaussianBlur(f, (0, 0), sig)
        if p["motion"] and rng.random() < 0.35:
            f = _motion_blur(f, rng.integers(3, p["motion"] + 1), rng)
        d = rng.uniform(*p["down"])
        if d < 0.98:
            h, w = f.shape[:2]
            small = cv2.resize(f, (max(8, int(w * d)), max(8, int(h * d))), interpolation=cv2.INTER_AREA)
            f = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        n = rng.uniform(0, p["noise"])
        if n > 0.5:
            f = np.clip(f.astype(np.float32) + rng.normal(0, n, f.shape).astype(np.float32), 0, 255).astype(np.uint8)
        a, b = rng.uniform(0.75, 1.2), rng.uniform(-20, 20)
        f = np.clip(f.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if rng.random() < p["gray_p"]:
            f = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        q = int(rng.integers(p["jpeg"][0], p["jpeg"][1] + 1))
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else f
    if out_size is not None and (out.shape[1], out.shape[0]) != tuple(out_size):
        out = cv2.resize(out, tuple(out_size), interpolation=cv2.INTER_AREA)
    return out
