import numpy as np


def per_type(y, yhat, T):
    """Recall, precision and F1 per class index (NaN where undefined)."""
    rec, prec, f1, n = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan), np.zeros(T, int)
    for t in range(T):
        m, p = y == t, yhat == t
        n[t] = m.sum()
        if m.sum():
            rec[t] = (yhat[m] == t).mean()
        if p.sum():
            prec[t] = (y[p] == t).mean()
        elif m.sum():
            prec[t] = 0.0
        if m.sum():
            r, q = rec[t], prec[t]
            f1[t] = 0.0 if (r + q) == 0 else 2 * r * q / (r + q)
    return rec, prec, f1, n


def auroc(scores, labels):
    """AUROC with labels 1 = positive; ties averaged."""
    from scipy.stats import rankdata
    labels = np.asarray(labels).astype(bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(scores)
    return (r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
