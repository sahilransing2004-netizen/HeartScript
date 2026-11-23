"""Numpy/scikit-learn metrics: AUROC, calibration, selective prediction, bootstrap CIs."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fit_temperature(logits, y):
    """Single temperature T minimising BCE of sigmoid(logits / T) on validation data."""
    from scipy.optimize import minimize_scalar

    logits, y = np.asarray(logits, np.float64), np.asarray(y, np.float64)

    def nll(log_t):
        z = logits / np.exp(log_t)
        return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))

    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def auroc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def per_class_auroc(Y, P):
    return np.array([auroc(Y[:, k], P[:, k]) for k in range(Y.shape[1])])


def macro_auroc(Y, P):
    v = per_class_auroc(Y, P)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def abnormal_label(Y):
    """Columns are NORM, MI, STTC, CD, HYP -> abnormal = any of the last four."""
    return Y[:, 1:].max(axis=1)


def abnormal_score(P):
    """Screening score = 1 - P(NORM)."""
    return 1.0 - P[:, 0]


def youden_threshold(y, s):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    return t if np.isfinite(t) else 0.5


def sens_spec(y, s, t):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(s) >= t
    tp, fn = np.sum(pred & y), np.sum(~pred & y)
    tn, fp = np.sum(~pred & ~y), np.sum(pred & ~y)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def best_f1_thresholds(Y, P, grid=None):
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    out = []
    for k in range(Y.shape[1]):
        best, bt = -1.0, 0.5
        for t in grid:
            f = _f1(Y[:, k], P[:, k] >= t)
            if f > best:
                best, bt = f, float(t)
        out.append(bt)
    return np.array(out)


def _f1(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, fp, fn = np.sum(pred & y), np.sum(pred & ~y), np.sum(~pred & y)
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0


def macro_f1(Y, P, thr):
    return float(np.mean([_f1(Y[:, k], P[:, k] >= thr[k]) for k in range(Y.shape[1])]))


def reliability_bins(y, p, n_bins=15):
    """Returns (bin_conf, bin_acc, bin_count) for equal-width bins of predicted probability."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        cnt.append(int(m.sum()))
        conf.append(float(p[m].mean()) if m.any() else float("nan"))
        acc.append(float(y[m].mean()) if m.any() else float("nan"))
    return np.array(conf), np.array(acc), np.array(cnt)


def ece_binary(y, p, n_bins=15):
    conf, acc, cnt = reliability_bins(y, p, n_bins)
    m = cnt > 0
    return float(np.sum(cnt[m] / cnt.sum() * np.abs(acc[m] - conf[m])))


def ece_macro(Y, P, n_bins=15):
    return float(np.mean([ece_binary(Y[:, k], P[:, k], n_bins) for k in range(Y.shape[1])]))


def risk_coverage(correct, conf):
    """Returns coverage array, risk (error rate among retained) array, and AURC (mean risk)."""
    correct, conf = np.asarray(correct, float), np.asarray(conf, float)
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - correct[order]
    k = np.arange(1, len(err) + 1)
    risk = np.cumsum(err) / k
    return k / len(err), risk, float(risk.mean())


def accuracy_at_coverage(correct, conf, coverage):
    correct, conf = np.asarray(correct, float), np.asarray(conf, float)
    k = max(1, int(round(coverage * len(correct))))
    idx = np.argsort(-conf, kind="stable")[:k]
    return float(correct[idx].mean())


def bootstrap_ci(fn, arrays, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
