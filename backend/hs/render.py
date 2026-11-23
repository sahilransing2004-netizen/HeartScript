"""Render a 12-lead signal as a paper-style ECG image (grid + 3x4 layout + rhythm strip)."""
import cv2
import numpy as np

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# PTB-XL lead order is I, II, III, aVR, aVL, aVF, V1..V6 (indices 0..11)
LAYOUT = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]  # rows x columns

# (background, minor grid, major grid) in BGR
GRID_STYLES = [
    ((245, 248, 255), (215, 215, 250), (170, 170, 240)),
    ((250, 250, 255), (200, 200, 255), (140, 140, 250)),
    ((240, 250, 240), (205, 235, 205), (150, 200, 150)),
    ((255, 255, 255), (225, 225, 225), (180, 180, 180)),
    ((235, 245, 255), (190, 200, 245), (120, 130, 235)),
]
TRACE_COLORS = [(0, 0, 0), (25, 25, 25), (50, 20, 20), (20, 20, 60)]


def _jit(color, rng, amt=8):
    return tuple(int(np.clip(c + rng.integers(-amt, amt + 1), 0, 255)) for c in color)


def render_ecg(signal, rng=None, width=768, height=576, fs=100, gain_mm_mv=None):
    """signal: (T, 12) in mV. Returns BGR uint8 image of size (height, width).
    Standard speed 25 mm/s; gain is 10 mm/mV unless randomised."""
    rng = rng or np.random.default_rng()
    sig = np.nan_to_num(np.asarray(signal, np.float32))
    if sig.ndim != 2 or sig.shape[1] != 12:
        raise ValueError(f"expected (T, 12) signal, got {sig.shape}")
    T = sig.shape[0]

    bg, minor, major = GRID_STYLES[int(rng.integers(len(GRID_STYLES)))]
    bg, minor, major = _jit(bg, rng, 4), _jit(minor, rng), _jit(major, rng)
    trace = TRACE_COLORS[int(rng.integers(len(TRACE_COLORS)))]
    img = np.empty((height, width, 3), np.uint8)
    img[:] = bg

    margin = int(round(width * 0.03))
    seconds = T / float(fs)
    ppm = (width - 2 * margin) / (25.0 * seconds)  # pixels per mm at 25 mm/s
    major_th = max(1, int(round(ppm / 3.5)))
    minor_th = max(1, int(round(ppm / 8.0)))

    # millimetre grid, major line every 5 mm
    for k in range(-int(margin // ppm) - 1, int((width - margin) // ppm) + 2):
        x = int(round(margin + k * ppm))
        if 0 <= x < width:
            is_major = k % 5 == 0
            cv2.line(img, (x, 0), (x, height - 1), major if is_major else minor,
                     major_th if is_major else minor_th)
    for k in range(-int(margin // ppm) - 1, int((height - margin) // ppm) + 2):
        y = int(round(margin + k * ppm))
        if 0 <= y < height:
            is_major = k % 5 == 0
            cv2.line(img, (0, y), (width - 1, y), major if is_major else minor,
                     major_th if is_major else minor_th)

    gain = gain_mm_mv or float(rng.choice([10, 10, 10, 10, 5, 20]))
    mv_px = gain * ppm
    px_per_sample = 25.0 * ppm / fs
    top_px = margin + 8.0 * ppm
    bottom_px = height - margin - 3.0 * ppm
    row_h = (bottom_px - top_px) / 4.0
    tr_th = max(1, int(round(ppm * rng.uniform(0.3, 0.7))))
    txt_scale = 0.38 * width / 768.0
    med = np.median(sig, axis=0)

    def draw(seg, x0, yc, name):
        y = yc - seg * mv_px
        half = row_h * 0.5 * 0.98
        y = np.clip(y, yc - half, yc + half)
        x = x0 + np.arange(len(seg)) * px_per_sample
        pts = np.round(np.stack([x, y], 1) * 16).astype(np.int32)
        cv2.polylines(img, [pts], False, trace, tr_th, cv2.LINE_AA, 4)
        cv2.putText(img, name, (int(x0 + 3 * ppm), int(yc - half + 10 * txt_scale + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, txt_scale, trace, 1, cv2.LINE_AA)

    seg_len = T // 4
    for r in range(3):
        yc = top_px + (r + 0.5) * row_h
        for c in range(4):
            li = LAYOUT[r][c]
            a, b = c * seg_len, (c + 1) * seg_len
            x0 = margin + a * px_per_sample
            draw(sig[a:b, li] - med[li], x0, yc, LEAD_NAMES[li])
            if c > 0:
                cv2.line(img, (int(x0), int(yc - 1.5 * ppm)), (int(x0), int(yc + 1.5 * ppm)), trace, 1)
    yc = top_px + 3.5 * row_h
    draw(sig[:, 1] - med[1], margin, yc, "II")

    if rng.random() < 0.8:
        cv2.putText(img, f"25 mm/s   {gain:g} mm/mV   {fs} Hz", (margin, int(margin + 4 * ppm)),
                    cv2.FONT_HERSHEY_SIMPLEX, txt_scale * 1.1, trace, 1, cv2.LINE_AA)
    return img
