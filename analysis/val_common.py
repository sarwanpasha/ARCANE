"""Shared helpers for nb_validate.py and lupus_robust.py (loading, scoring, donor-level statistics).
Only numpy / scipy / pandas (+ anndata for the h5ad) are required: OLS (HC3), logistic regression and the
likelihood-ratio test, AUROC, Mantel-Haenszel, Fleiss kappa and ICC are implemented here."""
import json
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

LOGF = [None]
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time() - T0:7.1f}s] {s}", flush=True)
    if LOGF[0]:
        LOGF[0].write(s + "\n")
        LOGF[0].flush()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def fmt(x, d=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "NA" if x is None else str(x)
    if not np.isfinite(x):
        return "NA"
    if x != 0 and (abs(x) < 10 ** (-d) or abs(x) >= 1e4):
        return f"{x:.1e}"
    return f"{x:.{d}f}"


def eci(e, lo, hi, d=3):
    return f"{fmt(e, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def find_col(cols, exact=(), patterns=(), exclude=(r"ontology_term_id",)):
    cols = list(cols)
    low = {c.lower(): c for c in cols}
    for e in exact:
        if e in cols:
            return e
        if e.lower() in low:
            return low[e.lower()]
    for p in patterns:
        for c in cols:
            if re.search(p, c, re.I) and not any(re.search(x, c, re.I) for x in exclude):
                return c
    return None


def is_sle(s):
    return bool(re.search(r"lupus|\bsle\b", str(s), re.I))


def collapse_levels(s, min_n=3, na="unknown"):
    s = pd.Series(s).astype(object)
    s = s.where(s.notna(), na).astype(str).replace({"nan": na, "None": na, "": na, "na": na})
    vc = s.value_counts()
    return s.where(~s.isin(vc[vc < min_n].index), "other")


def parse_age(s):
    if s is None:
        return np.nan
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*year", str(s))
    if m:
        return float(m.group(1))
    try:
        return float(str(s))
    except ValueError:
        return np.nan


def load_genesets(root, path):
    cands = [path] if os.path.isabs(path) else [os.path.join(root, path), os.path.join(root, "analysis", path),
                                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), path)]
    for p in cands:
        if os.path.exists(p):
            gs = json.load(open(p))
            log("genesets from", p)
            return {str(k): [str(g) for g in v] for k, v in gs.items() if isinstance(v, (list, tuple))}
    raise FileNotFoundError(f"genesets not found: {cands}")


# ------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------
def hl_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.median(np.subtract.outer(a, b)))


def strata_cells(g, strata):
    g, strata = np.asarray(g, int), np.asarray(strata).astype(str)
    return [np.where((strata == s) & (g == k))[0] for s in np.unique(strata) for k in (0, 1)]


def boot_idx(cells, rng):
    return np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells if len(c)])


def hl_boot(a, b, nboot, seed=3):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    est = hl_diff(a, b)
    if len(a) < 3 or len(b) < 3:
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = [hl_diff(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]) for _ in range(nboot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def strat_diff(y, g, strata):
    """sum_s w_s (mean_1 - mean_0) / sum_s w_s, w_s = n1 n0 / n, over strata containing both groups."""
    y, g, strata = np.asarray(y, float), np.asarray(g, int), np.asarray(strata).astype(str)
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        n1, n0 = int(g[m].sum()), int((1 - g[m]).sum())
        if n1 < 1 or n0 < 1:
            continue
        w = n1 * n0 / (n1 + n0)
        num += w * (y[m & (g == 1)].mean() - y[m & (g == 0)].mean())
        den += w
    return num / den if den > 0 else np.nan


def strat_perm(y, g, strata, nperm, seed=0):
    """Within-stratum permutation of the group labels; statistic = strat_diff. Two-sided and one-sided p."""
    y, g, strata = np.asarray(y, float), np.asarray(g, int), np.asarray(strata).astype(str)
    ok = np.isfinite(y)
    y, g, strata = y[ok], g[ok], strata[ok]
    rng = np.random.default_rng(seed)
    obs, perm, den, used = 0.0, np.zeros(nperm), 0.0, []
    for s in np.unique(strata):
        m = np.where(strata == s)[0]
        gg, yy = g[m], y[m]
        n1, n0 = int(gg.sum()), int(len(gg) - gg.sum())
        if n1 < 1 or n0 < 1:
            continue
        w = n1 * n0 / len(gg)
        tot = yy.sum()
        s1o = yy[gg == 1].sum()
        obs += w * (s1o / n1 - (tot - s1o) / n0)
        P = np.argsort(rng.random((nperm, len(gg))), axis=1)[:, :n1]
        s1 = yy[P].sum(1)
        perm += w * (s1 / n1 - (tot - s1) / n0)
        den += w
        used.append(dict(stratum=s, n1=n1, n0=n0))
    if not used:
        return dict(stat=np.nan, p=np.nan, p_greater=np.nan, p_less=np.nan, strata=[])
    eps = 1e-12 * max(1.0, abs(obs))
    return dict(stat=float(obs / den), p=float((1 + np.sum(np.abs(perm) >= abs(obs) - eps)) / (nperm + 1)),
                p_greater=float((1 + np.sum(perm >= obs - eps)) / (nperm + 1)),
                p_less=float((1 + np.sum(perm <= obs + eps)) / (nperm + 1)), strata=used)


def group_compare(y, g, strata, nboot, nperm, seed=0):
    """Donor-level SLE (g=1) vs healthy (g=0): plain mean difference, age-stratified difference with a
    stratified donor bootstrap CI, within-stratum permutation p; HL shift; per-stratum results."""
    y, g, strata = np.asarray(y, float), np.asarray(g, int), np.asarray(strata).astype(str)
    ok = np.isfinite(y)
    y, g, strata = y[ok], g[ok], strata[ok]
    out = dict(n_sle=int(g.sum()), n_healthy=int((1 - g).sum()))
    if g.sum() < 2 or (1 - g).sum() < 2:
        return out
    out["mean_sle"], out["mean_healthy"] = float(y[g == 1].mean()), float(y[g == 0].mean())
    out["diff"] = out["mean_sle"] - out["mean_healthy"]
    rng = np.random.default_rng(seed + 11)
    cells = strata_cells(g, strata)
    cells_all = strata_cells(g, np.zeros(len(g)))
    bs, bp = [], []
    for _ in range(nboot):
        i = boot_idx(cells, rng)
        bs.append(strat_diff(y[i], g[i], strata[i]))
        j = boot_idx(cells_all, rng)
        bp.append(y[j][g[j] == 1].mean() - y[j][g[j] == 0].mean())
    out["diff_lo"], out["diff_hi"] = [float(v) for v in np.nanpercentile(bp, [2.5, 97.5])]
    pr = strat_perm(y, g, strata, nperm, seed)
    out["strat_diff"] = pr["stat"]
    out["strat_lo"], out["strat_hi"] = [float(v) for v in np.nanpercentile(bs, [2.5, 97.5])]
    out["p_perm"], out["p_greater"], out["p_less"] = pr["p"], pr["p_greater"], pr["p_less"]
    out["strata_used"] = pr["strata"]
    out["hl"], out["hl_lo"], out["hl_hi"] = hl_boot(y[g == 1], y[g == 0], nboot, seed + 3)
    per = {}
    for s in np.unique(strata):
        m = strata == s
        if g[m].sum() >= 2 and (1 - g[m]).sum() >= 2:
            ys, gs = y[m], g[m]
            r = strat_perm(ys, gs, np.zeros(len(ys)), nperm, seed + 5)
            c = strata_cells(gs, np.zeros(len(gs)))
            b = []
            for _ in range(nboot):
                i = boot_idx(c, rng)
                b.append(ys[i][gs[i] == 1].mean() - ys[i][gs[i] == 0].mean())
            per[s] = dict(n_sle=int(gs.sum()), n_healthy=int((1 - gs).sum()), diff=r["stat"],
                          lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), p=r["p"],
                          p_greater=r["p_greater"], p_less=r["p_less"])
    out["by_stratum"] = per
    return out


def spearman_ci(x, y, nboot, seed=1):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 6 or np.std(x) == 0 or np.std(y) == 0:
        return dict(rho=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=int(len(x)))
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            bs.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return dict(rho=float(rho), lo=float(lo), hi=float(hi), p=float(p), n=int(len(x)))


def ols_hc3(y, X, names):
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    u = e / np.maximum(1 - h, 1e-4)
    V = XtXi @ (X.T * u ** 2) @ X @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    df = max(n - np.linalg.matrix_rank(X), 1)
    tq = stats.t.ppf(0.975, df)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = b / se
    res = {}
    for i, nm in enumerate(names):
        res[nm] = dict(coef=float(b[i]), se=float(se[i]), lo=float(b[i] - tq * se[i]), hi=float(b[i] + tq * se[i]),
                       p=float(2 * stats.t.sf(abs(t[i]), df)), p_greater=float(stats.t.sf(t[i], df)),
                       p_less=float(stats.t.cdf(t[i], df)))
    res["_n"], res["_k"], res["_rank"] = n, k, int(np.linalg.matrix_rank(X))
    return res


def design(df, nums, cats, min_n=3):
    cols, names = [np.ones(len(df))], ["Intercept"]
    for c in nums:
        cols.append(df[c].values.astype(float))
        names.append(c)
    for c in cats:
        lv = collapse_levels(df[c].values, min_n)
        levels = lv.value_counts().index.tolist()
        for l in levels[1:]:
            cols.append((lv.values == l).astype(float))
            names.append(f"{c}[{l}]")
    X = np.column_stack(cols)
    # drop constant / duplicated columns (except the intercept)
    keep = [0] + [j for j in range(1, X.shape[1]) if np.std(X[:, j]) > 0]
    return X[:, keep], [names[j] for j in keep]


def ols_effect(D, ycol, term, nums, cats, nboot, seed=7, strat_col=None):
    """OLS (HC3) of D[ycol] on term + nums + cats; donor bootstrap CI for `term` (resampling within
    strat_col x group cells when given). Rows with missing values are dropped."""
    use = [ycol, term] + list(nums)
    D = D.dropna(subset=use).copy()
    if len(D) < len(use) + 3 or D[term].nunique() < 2:
        return dict(n=int(len(D)), error="too few donors")
    X, nm = design(D, [term] + list(nums), cats)
    r = ols_hc3(D[ycol].values, X, nm)
    out = dict(r.get(term, {}))
    out.update(n=int(len(D)), covariates=nm[1:])
    if term not in r:
        return out
    j = nm.index(term)
    rng = np.random.default_rng(seed)
    g = (D[term].values > np.median(D[term].values)).astype(int) if D[term].nunique() > 2 else D[term].values.astype(int)
    st = D[strat_col].astype(str).values if strat_col else np.zeros(len(D))
    cells = strata_cells(g, st)
    y = D[ycol].values.astype(float)
    bs = []
    for _ in range(nboot):
        i = boot_idx(cells, rng)
        Xi = X[i]
        if np.linalg.matrix_rank(Xi) < X.shape[1]:
            continue
        bs.append(np.linalg.lstsq(Xi, y[i], rcond=None)[0][j])
    if len(bs) >= 50:
        out["boot_lo"], out["boot_hi"] = [float(v) for v in np.percentile(bs, [2.5, 97.5])]
        out["boot_n"] = len(bs)
    return out


def logit_fit(y, X, ridge=1e-6, maxit=200):
    y, X = np.asarray(y, float), np.asarray(X, float)
    b = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], ridge)
    pen[0] = 0
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        grad = X.T @ (y - p) - pen * b
        H = (X.T * W) @ X + np.diag(pen) + 1e-10 * np.eye(len(b))
        step = np.linalg.solve(H, grad)
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(X @ b, -30, 30)
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(H)), 0))
    return b, se, ll, p


def logistic_lrt(D, ycol, base_nums, add, cats=()):
    """LRT for adding `add` to logit(ycol) ~ base_nums + cats. Numeric predictors are z-scored."""
    D = D.dropna(subset=[ycol, add] + list(base_nums)).copy()
    for c in list(base_nums) + [add]:
        sd = D[c].std()
        D[c] = (D[c] - D[c].mean()) / (sd if sd > 0 else 1)
    if len(D) < 8 or D[ycol].nunique() < 2:
        return dict(n=int(len(D)), error="too few donors")
    X0, n0 = design(D, base_nums, cats)
    X1, n1 = design(D, list(base_nums) + [add], cats)
    y = D[ycol].values.astype(float)
    b0, _, ll0, p0 = logit_fit(y, X0)
    b1, se1, ll1, p1 = logit_fit(y, X1)
    chi = max(2 * (ll1 - ll0), 0.0)
    j = n1.index(add)
    sep = bool(np.max(np.abs(b1[1:])) > 15) if len(b1) > 1 else False
    return dict(n=int(len(D)), chi2=float(chi), df=int(X1.shape[1] - X0.shape[1]),
                p=float(stats.chi2.sf(chi, max(X1.shape[1] - X0.shape[1], 1))), coef_per_sd=float(b1[j]),
                se=float(se1[j]), auc_base=auroc(p0, y), auc_full=auroc(p1, y), separation_warning=sep,
                base_terms=n0[1:])


def auroc(score, y):
    s, y = np.asarray(score, float), np.asarray(y).astype(int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_test(score, y, nboot, seed=4):
    s, y = np.asarray(score, float), np.asarray(y).astype(int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    a = auroc(s, y)
    if y.sum() < 2 or (1 - y).sum() < 2:
        return dict(auroc=a, n=int(len(s)))
    rng = np.random.default_rng(seed)
    cells = strata_cells(y, np.zeros(len(y)))
    bs = [auroc(s[i], y[i]) for i in (boot_idx(cells, rng) for _ in range(nboot))]
    p = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided").pvalue
    return dict(auroc=a, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p_mwu=float(p),
                n_sle=int(y.sum()), n_healthy=int((1 - y).sum()))


def mantel_haenszel(tabs):
    T = np.array([t for t in tabs if (t[0] + t[1]) > 0 and (t[2] + t[3]) > 0], float).reshape(-1, 4)
    if len(T) == 0:
        return dict(OR=np.nan, lo=np.nan, hi=np.nan, n_strata=0)
    a, b, c, d = T.T
    n = T.sum(1)
    Rr, S = a * d / n, b * c / n
    P, Q = (a + d) / n, (b + c) / n
    sR, sS = Rr.sum(), S.sum()
    if sR <= 0 or sS <= 0:
        return dict(OR=(np.inf if sS <= 0 < sR else (0.0 if sR <= 0 < sS else np.nan)), lo=np.nan, hi=np.nan,
                    n_strata=int(len(T)))
    orr = sR / sS
    var = (P * Rr).sum() / (2 * sR ** 2) + (P * S + Q * Rr).sum() / (2 * sR * sS) + (Q * S).sum() / (2 * sS ** 2)
    se = np.sqrt(var)
    return dict(OR=float(orr), lo=float(np.exp(np.log(orr) - 1.96 * se)), hi=float(np.exp(np.log(orr) + 1.96 * se)),
                n_strata=int(len(T)))


def logit_frac(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    return np.log((k + 0.5) / (n - k + 0.5))


def fleiss_kappa_binary(F):
    """F: n_items x k_raters 0/1 (no NaN)."""
    F = np.asarray(F, float)
    k = F.shape[1]
    n1 = F.sum(1)
    n0 = k - n1
    Pi = (n1 ** 2 + n0 ** 2 - k) / (k * (k - 1))
    p1 = n1.sum() / (len(F) * k)
    Pe = p1 ** 2 + (1 - p1) ** 2
    return float((Pi.mean() - Pe) / (1 - Pe)) if Pe < 1 else np.nan


def icc(Y):
    """Y: n_targets x k_raters. Returns ICC(2,1) (absolute agreement) and ICC(3,1) (consistency)."""
    Y = np.asarray(Y, float)
    Y = Y[np.isfinite(Y).all(1)]
    n, k = Y.shape
    if n < 3 or k < 2:
        return dict(icc2_1=np.nan, icc3_1=np.nan, n=int(n))
    g = Y.mean()
    ssr = k * ((Y.mean(1) - g) ** 2).sum()
    ssc = n * ((Y.mean(0) - g) ** 2).sum()
    sse = ((Y - g) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return dict(icc2_1=float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)),
                icc3_1=float((msr - mse) / (msr + (k - 1) * mse)), n=int(n))


# ------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------
def load_cells(root, res_dir, tag, seeds, pcols, alpha):
    """Per seed flags (p <= alpha; NaN p -> NaN) aligned to the names of the first seed.
    Returns dict(names, batch, cond, label_arcane (seed-majority), flags={pcol: DataFrame cells x seeds})."""
    out = dict(flags={c: [] for c in pcols}, labels=[], seeds=[])
    names0 = None
    for s in seeds:
        f = os.path.join(root, res_dir, f"{tag}_seed{s}_cells.npz")
        if not os.path.exists(f):
            log("MISSING", f)
            continue
        C = np.load(f, allow_pickle=True)
        nm = pd.Index(np.asarray(C["names"]).astype(str))
        dup = nm.duplicated()
        if dup.any():
            log(f"WARNING seed {s}: {int(dup.sum())} duplicated names; keeping first")
        if names0 is None:
            names0 = nm[~dup]
            out["batch"] = pd.Series(np.asarray(C["batch"]).astype(str), index=nm)[~dup].values
            out["cond"] = pd.Series(np.asarray(C["cond"]).astype(str), index=nm)[~dup].values
        elif not nm[~dup].isin(names0).all() or len(nm[~dup]) != len(names0):
            log(f"WARNING seed {s}: cell names differ from first seed (matched by name; missing -> NaN)")
        for c in pcols:
            if c in C.files:
                pv = np.asarray(C[c], float)
                fl = np.where(np.isfinite(pv), (pv <= alpha).astype(float), np.nan)
                out["flags"][c].append(pd.Series(fl, index=nm)[~dup].reindex(names0).rename(s))
            else:
                log(f"seed {s}: column {c} missing")
        lab = np.asarray(C["label_arcane"]).astype(str) if "label_arcane" in C.files else np.full(len(nm), "na")
        out["labels"].append(pd.Series(lab, index=nm)[~dup].reindex(names0).fillna("na").values)
        out["seeds"].append(s)
        log(f"seed {s}: {len(nm)} cells; " + ", ".join(
            f"{c} rate {np.nanmean(out['flags'][c][-1]):.4f}" for c in pcols if out["flags"][c]))
    if names0 is None:
        raise FileNotFoundError(f"no {res_dir}/{tag}_seed*_cells.npz")
    out["names"] = names0
    out["flags"] = {c: pd.concat(v, axis=1) for c, v in out["flags"].items() if v}
    L = np.column_stack(out["labels"])
    agree = np.column_stack([(L == L[:, [j]]).sum(1) for j in range(L.shape[1])])
    out["label_arcane"] = L[np.arange(len(L)), np.argmax(agree, 1)]
    out["label_seed_agreement"] = float((agree.max(1) == L.shape[1]).mean())
    return out


def consensus(F):
    """Cell flagged when flagged in more than half of the seeds with a finite flag (>= 2 of 3)."""
    F = np.asarray(F, float)
    nav = np.isfinite(F).sum(1)
    k = np.nansum(F, 1)
    return np.where(nav > 0, (k > nav / 2).astype(float), np.nan)


def symbols(var_names, var_df):
    var = pd.Index(np.asarray(var_names).astype(str))
    if np.mean(var.str.startswith("ENS")) > 0.5 and var_df is not None:
        for c in ["feature_name", "gene_symbols", "gene_symbol", "symbol", "gene_name"]:
            if c in var_df.columns:
                return pd.Index(var_df[c].astype(str).values)
    return var


def load_expression(h5, genes, chunk=50000, obs_want=None):
    """Backed read; rows in chunks; library size from all genes; keeps only `genes` as log1p CP10k (float32).
    Returns obs (selected columns as str, index = obs names), present gene list, E."""
    import anndata as ad
    log("reading", h5)
    try:
        A = ad.read_h5ad(h5, backed="r")
        backed = True
    except Exception as e:  # noqa: BLE001
        log(f"backed read failed ({e!r}); loading fully")
        A = ad.read_h5ad(h5)
        backed = False
    var = symbols(A.var_names, A.var)
    first = {}
    for i, g in enumerate(var):
        first.setdefault(g, i)
    present = [g for g in dict.fromkeys(genes) if g in first]
    gidx = np.array([first[g] for g in present], int)
    log(f"genes present {len(present)}/{len(set(genes))}; missing: {sorted(set(genes) - set(present))}")
    cols = list(A.obs.columns)
    O = pd.DataFrame(index=pd.Index(np.asarray(A.obs_names).astype(str), name="cell"))
    found = {}
    for k, (ex, pats) in (obs_want or {}).items():
        c = find_col(cols, ex, pats)
        found[k] = c
        O[k] = A.obs[c].astype(str).values if c else "na"
    log("obs columns:", found)
    n = A.n_obs if hasattr(A, "n_obs") else A.X.shape[0]
    E = np.zeros((n, len(gidx)), np.float32)
    nonint = False
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        Xi = A.X[lo:hi]
        if sp.issparse(Xi):
            Xi = sp.csr_matrix(Xi, dtype=np.float32)
            samp = Xi.data[:100000]
            tot = np.asarray(Xi.sum(1)).ravel()
            sub = Xi[:, gidx].toarray()
        else:
            Xi = np.asarray(Xi, dtype=np.float32)
            samp = Xi[:50].ravel()
            tot = Xi.sum(1)
            sub = Xi[:, gidx]
        if len(samp) and not np.allclose(samp, np.round(samp)):
            nonint = True
            E[lo:hi] = sub
        else:
            E[lo:hi] = np.log1p(sub * (1e4 / np.maximum(tot, 1))[:, None])
        if (lo // chunk) % 5 == 0:
            log(f"  rows {hi}/{n}")
    if nonint:
        log("WARNING: X not integer counts; assumed already log-normalized")
    try:
        if backed:
            A.file.close()
    except Exception:  # noqa: BLE001
        pass
    return O, present, E, found


# ------------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------------
def z_by_label(E, labs, isref, sd_floor=0.1, zclip=10.0, min_ref=50):
    """z of every cell against the reference cells of the same label (cells of labels with < min_ref
    reference cells -> NaN)."""
    Z = np.full(E.shape, np.nan, np.float32)
    nref = {}
    for l in np.unique(labs):
        m = labs == l
        r = m & isref
        nref[l] = int(r.sum())
        if r.sum() < min_ref:
            continue
        mu = E[r].mean(0)
        sd = np.maximum(E[r].std(0), sd_floor)
        Z[m] = np.clip((E[m] - mu) / sd, -zclip, zclip)
    return Z, nref


def gene_score(Z, gi, up, dn=()):
    u = [gi[g] for g in up if g in gi]
    d = [gi[g] for g in dn if g in gi]
    if len(u) < 2 or (dn and len(d) < 2):
        return None, [g for g in up if g in gi], [g for g in dn if g in gi]
    with np.errstate(invalid="ignore"):
        s = np.nanmean(Z[:, u], 1)
        if d:
            s = s - np.nanmean(Z[:, d], 1)
    return s.astype(np.float32), [g for g in up if g in gi], [g for g in dn if g in gi]


def healthy_weights(lab, donor, healthy):
    d = pd.DataFrame(dict(lab=lab[healthy], donor=donor[healthy]))
    comp = d.groupby("donor").lab.value_counts(normalize=True).unstack(fill_value=0)
    return comp.mean(0)


def std_rate(values, lab, donor, w):
    """Composition-standardized donor mean of `values` (per label donor means weighted by w, renormalized
    over the labels the donor has). Also raw mean and n."""
    d = pd.DataFrame(dict(v=np.asarray(values, float), lab=lab, donor=donor)).dropna(subset=["v"])
    g = d.groupby(["donor", "lab"]).v.mean().unstack()
    W = w.reindex(g.columns).fillna(0.0)
    num = (g.fillna(0) * W).sum(1)
    den = (g.notna() * W).sum(1)
    out = pd.DataFrame(dict(std=num / den.replace(0, np.nan), raw=d.groupby("donor").v.mean(),
                            n=d.groupby("donor").size()))
    return out


def match_labels(all_labels, rx, excl=r"prolif|cycling|progen|doublet|unknown|^nan$|^na$"):
    return [l for l in all_labels if re.search(rx, l, re.I) and not re.search(excl, l, re.I)]


NCM_RX = r"^ncM$|non.?classical|CD16\+? ?mono"
CM_RX = r"^cM$|^classical mono|CD14\+? ?mono"
C1Q = ["C1QA", "C1QB", "C1QC", "FCGR1A", "CPVL"]
IFNG_UP = ["GBP1", "GBP5", "CXCL9", "CXCL10", "FCGR1A"]


def state_specs(gs, all_ref_labels):
    ncm = match_labels(all_ref_labels, NCM_RX)
    cm = match_labels(all_ref_labels, CM_RX)
    return [dict(name="C1Q_ncM", up=C1Q, down=[], labels=ncm),
            dict(name="IFNg_mono", up=IFNG_UP, down=list(gs.get("ISG_typeI_core_specific", [])), labels=cm + ncm)]


def threshold_ref(score, lab, isref, labels, q):
    """Positive call for non-reference cells: score > q-th percentile of the reference cells of that label."""
    pos = np.full(len(score), np.nan)
    thr = {}
    for l in labels:
        r = (lab == l) & isref & np.isfinite(score)
        m = (lab == l) & ~isref & np.isfinite(score)
        if r.sum() < 20:
            continue
        t = np.percentile(score[r], q)
        thr[l] = float(t)
        pos[m] = (score[m] > t).astype(float)
    return pos, thr


def threshold_healthy(score, lab, isquery, healthy, donor, labels, q, min_cells=50):
    """Threshold = q-th percentile of the healthy query cells of the label; healthy donors are scored with the
    threshold computed without their own cells (leave one donor out)."""
    pos = np.full(len(score), np.nan)
    thr = {}
    for l in labels:
        h = (lab == l) & isquery & healthy & np.isfinite(score)
        m = (lab == l) & isquery & np.isfinite(score)
        if h.sum() < min_cells:
            continue
        t = np.percentile(score[h], q)
        thr[l] = float(t)
        s = m & ~healthy
        pos[s] = (score[s] > t).astype(float)
        for d in np.unique(donor[h]):
            tr = h & (donor != d)
            te = m & healthy & (donor == d)
            if tr.sum() < min_cells:
                continue
            pos[te] = (score[te] > np.percentile(score[tr], q)).astype(float)
    return pos, thr


def isg_topn(isg, flag, lab, cellmask, labels):
    """ISG-only detector: per label flag the top-n cells (within cellmask) by ISG, n = ARCANE flags there."""
    out = np.full(len(isg), np.nan)
    for l in labels:
        m = np.where(cellmask & (lab == l) & np.isfinite(flag))[0]
        if len(m) == 0:
            continue
        nfl = int(np.nansum(flag[m]))
        f = np.zeros(len(m))
        if nfl > 0:
            f[np.argsort(-np.nan_to_num(isg[m], nan=-np.inf), kind="stable")[:nfl]] = 1
        out[m] = f
    return out


def mh_by_donor(pos, det, donor, mask):
    m = mask & np.isfinite(pos) & np.isfinite(det)
    d = pd.DataFrame(dict(p=pos[m] == 1, f=det[m] == 1, donor=donor[m]))
    tabs = [(int((x.p & x.f).sum()), int((x.p & ~x.f).sum()), int((~x.p & x.f).sum()), int((~x.p & ~x.f).sum()))
            for _, x in d.groupby("donor")]
    return mantel_haenszel(tabs)
