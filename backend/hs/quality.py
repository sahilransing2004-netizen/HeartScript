"""Input-quality gate: rejects blurred, blank/over-/under-exposed, and non-ECG-looking photos
BEFORE the classifier runs. These are simple image heuristics, not a learned OOD detector:
fit the thresholds on validation images (QualityGate.fit) and report how many real
negatives (selfies, documents, plain grid paper) it rejects."""
from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULTS = dict(
    blur_min=15.0,        # variance of Laplacian on a 640-px-wide gray image
    mean_min=35.0, mean_max=245.0,
    std_min=12.0,         # contrast
    orient_min=0.30,      # share of gradient energy in two orthogonal orientations (grid)
    dark_frac_min=0.002, dark_frac_max=0.20,   # share of trace-dark pixels
    trace_extent_min=0.35,  # longest dark connected component / longer image side
)


@dataclass
class QualityReport:
    ok: bool
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _gray(img, width=640):
    h, w = img.shape[:2]
    if w != width:
        img = cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _orientation_score(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    hist, _ = np.histogram(ang, bins=36, range=(0, 180), weights=mag)
    total = hist.sum() + 1e-9
    best = 0.0
    for i in range(36):
        j = (i + 18) % 36
        e = sum(hist[(i + d) % 36] + hist[(j + d) % 36] for d in (-1, 0, 1))
        best = max(best, e)
    return float(best / total)


def _trace_stats(gray):
    """Dark-trace fraction and extent of the longest dark connected component.
    A pixel is 'trace-dark' if it is much darker than its large-scale local average
    (grid lines are only slightly darker, so they do not qualify)."""
    local = cv2.GaussianBlur(gray, (0, 0), 25)
    dark = ((local.astype(np.int16) - gray.astype(np.int16)) > 60).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    frac = float(dark.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return frac, 0.0
    longest = max(max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, n))
    return frac, float(longest / max(gray.shape))


def analyze(img):
    g = _gray(img)
    frac, extent = _trace_stats(g)
    return dict(
        blur=float(cv2.Laplacian(g, cv2.CV_64F).var()),
        mean=float(g.mean()), std=float(g.std()),
        orient=_orientation_score(g),
        dark_frac=frac, trace_extent=extent,
    )


class QualityGate:
    def __init__(self, thresholds=None):
        self.t = dict(DEFAULTS)
        if thresholds:
            self.t.update(thresholds)

    def check(self, img):
        m = analyze(img)
        t, why = self.t, []
        if m["blur"] < t["blur_min"]:
            why.append("image is too blurry")
        if m["mean"] < t["mean_min"] or m["mean"] > t["mean_max"] or m["std"] < t["std_min"]:
            why.append("image is too dark, too bright or has no contrast")
        if m["orient"] < t["orient_min"]:
            why.append("no ECG grid pattern found")
        if not (t["dark_frac_min"] <= m["dark_frac"] <= t["dark_frac_max"]) or m["trace_extent"] < t["trace_extent_min"]:
            why.append("no ECG trace found")
        return QualityReport(ok=not why, reasons=why, metrics=m)

    def fit(self, images, q=1.0, safety=0.8):
        """Set thresholds from images that SHOULD pass (e.g. degraded validation renders):
        lower bounds at the q-th percentile times `safety`."""
        ms = [analyze(im) for im in images]
        col = lambda k: np.array([m[k] for m in ms])
        # floors stop a few very poor training images from making the gate uselessly lenient
        self.t["blur_min"] = max(float(np.percentile(col("blur"), q) * safety), 10.0)
        self.t["orient_min"] = max(float(np.percentile(col("orient"), q) * safety), 0.20)
        self.t["trace_extent_min"] = max(float(np.percentile(col("trace_extent"), q) * safety), 0.30)
        self.t["std_min"] = max(float(np.percentile(col("std"), q) * safety), 10.0)
        self.t["dark_frac_min"] = float(np.percentile(col("dark_frac"), q) * safety)
        self.t["dark_frac_max"] = float(np.percentile(col("dark_frac"), 100 - q) / safety)
        return self

    def to_dict(self):
        return dict(self.t)

    @classmethod
    def from_dict(cls, d):
        return cls(d)


def make_negatives(rng, n=60, size=(576, 768)):
    """Synthetic 'not an ECG' images: blank, noise, smooth scenes, plain grid paper, ruled lines.
    Use real negatives (selfies, documents, random photos) in the real evaluation."""
    h, w = size
    out = []
    for i in range(n):
        kind = ["blank", "noise", "scene", "grid_only", "lines", "blocks"][i % 6]
        if kind == "blank":
            img = np.full((h, w, 3), int(rng.integers(0, 255)), np.uint8)
        elif kind == "noise":
            img = rng.integers(0, 255, (h, w, 3)).astype(np.uint8)
        elif kind == "scene":
            small = rng.integers(0, 255, (6, 8, 3)).astype(np.uint8)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        elif kind == "grid_only":
            img = np.full((h, w, 3), 250, np.uint8)
            step = int(rng.integers(6, 14))
            for k in range(0, w, step):
                cv2.line(img, (k, 0), (k, h), (200, 200, 255) if k % (5 * step) else (140, 140, 250), 1)
            for k in range(0, h, step):
                cv2.line(img, (0, k), (w, k), (200, 200, 255) if k % (5 * step) else (140, 140, 250), 1)
        elif kind == "lines":
            img = np.full((h, w, 3), 245, np.uint8)
            for k in range(20, h, int(rng.integers(18, 40))):
                cv2.line(img, (20, k), (w - 20, k), (40, 40, 40), 1)
        else:
            img = np.full((h, w, 3), 220, np.uint8)
            for _ in range(8):
                x0, y0 = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
                cv2.rectangle(img, (x0, y0), (x0 + int(rng.integers(30, 250)), y0 + int(rng.integers(30, 200))),
                              tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        out.append((kind, img))
    return out
