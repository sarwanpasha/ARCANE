#!/usr/bin/env python
"""SLE patients with a healthy-range interferon score who still carry many reference-unmatched (ARCANE-flagged) cells.

Run from ~/arcane:
    python isglow_patients.py [--root .] [--tag lupusfull] [--seeds 0,1,2] [--out summary/isglow]

Flags (per seed, then seed majority):  ctrl   = p_unknown_ctrl <= alpha
                                       strict = ctrl and p_unknown_bonf <= alpha
Cell states for pseudobulk: kept (not ctrl), ctrl-only, strict.  "Flagged" = ctrl (states 1+2) or strict (state 2);
the kept comparator is always state 0 (not flagged by ctrl).

Steps
  1  donor (= sample unit, obs `batch`; individual = donor_id) type-I ISG: per cell mean log1p-CP10k of ISG genes,
     z-scored against reference-healthy cells (role=='ref') of the same fine label; donor score = label-mix-standardized
     mean (weights = healthy-query label mix).  Donor flag fraction standardized to the same label mix.
     Thresholds = healthy-query q90 (and q95) of ISG and of flag fraction.  SLE groups: ISGhigh, ISGlow_flagHigh,
     ISGlow_flagLow.  ISG-low SLE vs healthy-query flag fraction: within-cohort (Processing_Cohort) weighted mean
     difference, permutation within cohort, stratified donor bootstrap CI; exceedance of the healthy q90 (threshold
     recomputed in every permutation); sensitivities: ISG-matched healthy, one unit per individual, OLS + ISG + cohort.
  2  ISG-low SLE: per-label flag rates vs healthy (diff, bootstrap CI, within-cohort permutation p, BH, share of
     standardized excess); donor-paired pseudobulk flagged-vs-kept log2FC per label (ISG-low SLE vs healthy);
     gene-set scores of flagged cells (level vs healthy kept cells, and paired flagged-minus-kept delta).
  3  clinical: disease_state, age, sex, ancestry, cohort, all-cell ISG/GC/IEG of ISGlow_flagHigh vs ISGhigh
     (Monte-Carlo exact chi-square / Fisher / KW / MWU).
  4  stability of ISGlow_flagHigh membership across seeds and under healthy-threshold bootstrap.
  5  leave-one-cohort-out robustness of step 1.
Outputs: <out>/RESULTS.md (<8 KB), results.json, csv tables, PNG figures, log.txt.  Sections never crash the run.
Memory: one streaming pass over the h5ad rows (backed mode, row windows, per-chunk CP10k); the pseudobulk accumulator
is (units x labels x 3) x genes float64 (~3.5 GB for 205 x 23 x 3 x 30k).
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

EXTRA_SETS = {
    "C1Q": ["C1QA", "C1QB", "C1QC"],
    "cDC1": ["CLEC9A", "DNASE1L3", "WDFY4", "XCR1"],
    "cycling": ["MKI67", "TYMS", "PCLAF", "TK1", "MCM7", "STMN1"],
}
# focus set name -> regex patterns over genesets.json keys (first match used)
FOCUS_FROM_GS = {
    "ISG_typeI": [r"^ISG_typeI$", r"^ISG"],
    "platelet": [r"^platelet$", r"platelet"],
    "HLA_classII": [r"^HLA_classII$", r"HLA"],
    "plasmablast": [r"^plasmablast$", r"plasma"],
    "DN2": [r"^DN2_ABC_up$", r"DN2"],
    "glucocorticoid": [r"^glucocorticoid_core$", r"glucocorticoid"],
    "IEG": [r"^dissociation_IEG$", r"IEG_core", r"IEG|dissoc"],
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tag", default="lupusfull")
    ap.add_argument("--h5ad", default=None)
    ap.add_argument("--res_dir", default="res2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--genesets", default="genesets.json")
    ap.add_argument("--isg_key", default="ISG_typeI")
    ap.add_argument("--out", default="summary/isglow")
    ap.add_argument("--chunk", type=int, default=20000, help="h5ad row window")
    ap.add_argument("--max_ref_cells", type=int, default=300000, help="reference cells read for ISG z (0 = all)")
    ap.add_argument("--min_ref_label", type=int, default=50, help="min ref cells of a label, else pooled ref")
    ap.add_argument("--min_cells", type=int, default=20, help="min cells of a unit x label for rates/scores")
    ap.add_argument("--min_flag", type=int, default=5, help="min flagged cells of a unit x label for profiles")
    ap.add_argument("--min_flag_sets", type=int, default=3, help="min flagged cells of a unit x label for set scores")
    ap.add_argument("--min_kept", type=int, default=20)
    ap.add_argument("--min_donors", type=int, default=4, help="min donors per group for a per-label comparison")
    ap.add_argument("--min_mean", type=float, default=0.2, help="min mean CP10k of a gene in a label")
    ap.add_argument("--pc", type=float, default=0.1, help="pseudo-count (CP10k) for log2")
    ap.add_argument("--sd_floor", type=float, default=0.25, help="floor of healthy between-donor log2 SD")
    ap.add_argument("--qs", default="0.90,0.95", help="healthy quantiles for thresholds (first = main)")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--top_genes", type=int, default=40, help="genes per label written to the LFC table")
    return ap.parse_args(argv)


ARGS = None
OUT = None
LOGF = None
R = {"errors": {}, "timing_s": {}}
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGF:
        LOGF.write(s + "\n")
        LOGF.flush()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Series):
        return clean(o.to_dict())
    return o


def save_results():
    try:
        with open(os.path.join(OUT, "results.json"), "w") as f:
            json.dump(clean(R), f, indent=1, default=str)
    except Exception as e:  # noqa: BLE001
        log("save_results failed", repr(e))


def section(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            log(f"\n=== [{time.time() - T0:7.0f}s] {name}")
            try:
                out = fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                R["errors"][name] = repr(e)
                log(f"!!! section {name} failed: {e!r}")
                log(traceback.format_exc())
                out = None
            R["timing_s"][name] = round(time.time() - t, 1)
            save_results()
            return out
        return run
    return deco


plt.rcParams.update({"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 160})


def savefig(fig, name):
    try:
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    finally:
        plt.close(fig)


def wcsv(df, name, index=False):
    df.to_csv(os.path.join(OUT, name), index=index)


# ----------------------------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------------------------
def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    m = np.isfinite(p)
    if m.sum() == 0:
        return out
    pv = p[m]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    q = np.empty_like(pv)
    q[o] = np.minimum(r, 1)
    out[m] = q
    return out


def fmt(x, d=2):
    try:
        if x is None or not np.isfinite(float(x)):
            return "NA"
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x != 0 and (abs(x) < 10 ** (-d) or abs(x) >= 1e4):
        return f"{x:.1e}"
    return f"{x:.{d}f}"


def eci(e, k="est", d=3, mult=1.0):
    if not e:
        return "NA"
    f = lambda v: fmt(v * mult if v is not None and np.isfinite(v) else np.nan, d)  # noqa: E731
    return f"{f(e.get(k))} [{f(e.get('lo'))}, {f(e.get('hi'))}]"


def mwu_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided")[1])
    except ValueError:
        return 1.0


def signed_rank(L):
    """Vectorized Wilcoxon signed-rank (normal approx.) per column of L (n x G); NaN-free."""
    L = np.asarray(L, float)
    n = L.shape[0]
    if n < 2:
        return np.full(L.shape[1], np.nan), np.full(L.shape[1], np.nan)
    r = stats.rankdata(np.abs(L), axis=0)
    pos, zer = L > 0, L == 0
    wp = (r * pos).sum(0) + 0.5 * (r * zer).sum(0)
    mu = n * (n + 1) / 4.0
    t0 = zer.sum(0).astype(float)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - (t0 ** 3 - t0) / 48.0
    z = np.where(var > 0, (wp - mu) / np.sqrt(np.maximum(var, 1e-12)), 0.0)
    p = 2 * stats.norm.sf(np.abs(z))
    p[t0 == n] = 1.0
    return z, p


def strat_stat(Y, g, strata):
    """Within-stratum mean difference (g=1 minus g=0), pooled over strata with weights n1*n0/(n1+n0).
    Y: n x K (NaN = missing), g: n bool, strata: n labels.  Returns K-vector."""
    Y = np.atleast_2d(np.asarray(Y, float).T).T
    V = np.isfinite(Y)
    Y0 = np.where(V, Y, 0.0)
    g = np.asarray(g, bool)
    num = np.zeros(Y.shape[1])
    den = np.zeros(Y.shape[1])
    for s in np.unique(strata):
        m = strata == s
        a, b = m & g, m & ~g
        n1, n0 = V[a].sum(0), V[b].sum(0)
        ok = (n1 >= 1) & (n0 >= 1)
        d = np.where(ok, Y0[a].sum(0) / np.maximum(n1, 1) - Y0[b].sum(0) / np.maximum(n0, 1), 0.0)
        w = np.where(ok, n1 * n0 / np.maximum(n1 + n0, 1), 0.0)
        num += w * d
        den += w
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _clusters(g, strata, clusters):
    """Cluster structure for permutation / bootstrap.  A cluster = individual x group (so g is constant within a
    cluster); its block = its most frequent stratum.  clusters=None -> every unit is its own cluster."""
    g = np.asarray(g, bool)
    strata = np.asarray(strata).astype(str)
    n = len(g)
    if clusters is None:
        cid = np.arange(n).astype(str)
    else:
        cid = np.asarray(clusters).astype(str)
    cid = np.char.add(np.char.add(cid, "|"), g.astype(int).astype(str))
    codes, uniq = pd.factorize(cid)
    K = len(uniq)
    gc = np.zeros(K, bool)
    gc[codes] = g
    blk = pd.Series(strata).groupby(codes).agg(lambda s: s.value_counts().index[0]).reindex(range(K)).values
    members = pd.Series(np.arange(n)).groupby(codes).apply(lambda s: s.values).reindex(range(K)).values
    return codes, gc, np.asarray(blk).astype(str), members


def perm_matrix(g, strata, clusters, nperm, rng):
    """(nperm+1) x n group indicator; row 0 observed, others permute cluster labels within cluster blocks."""
    codes, gc, blk, _ = _clusters(g, strata, clusters)
    P = nperm + 1
    Gc = np.empty((P, len(gc)), bool)
    Gc[0] = gc
    bidx = [np.where(blk == b)[0] for b in np.unique(blk)]
    for i in range(1, P):
        row = gc.copy()
        for ix in bidx:
            if len(ix) > 1:
                row[ix] = gc[ix][rng.permutation(len(ix))]
        Gc[i] = row
    return Gc[:, codes]


def strat_perm_multi(Y, g, strata, nperm=None, seed=0, clusters=None):
    """Permutation p (two-sided) of strat_stat, permuting g within strata (at the cluster = individual level when
    clusters are given, so repeated samples of one person move together); vectorized over columns."""
    nperm = nperm or ARGS.nperm
    Y = np.atleast_2d(np.asarray(Y, float).T).T
    g = np.asarray(g, bool)
    strata = np.asarray(strata).astype(str)
    V = np.isfinite(Y).astype(float)
    Y0 = np.where(V > 0, Y, 0.0)
    rng = np.random.default_rng(seed)
    P = nperm + 1
    GP = perm_matrix(g, strata, clusters, nperm, rng).astype(float)
    num = np.zeros((P, Y.shape[1]))
    den = np.zeros((P, Y.shape[1]))
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        Gp = GP[:, idx]
        if (Gp == 1).all() or (Gp == 0).all():
            continue
        Bp = 1.0 - Gp
        n1, n0 = Gp @ V[idx], Bp @ V[idx]
        s1, s0 = Gp @ Y0[idx], Bp @ Y0[idx]
        ok = (n1 >= 1) & (n0 >= 1)
        d = np.where(ok, s1 / np.maximum(n1, 1) - s0 / np.maximum(n0, 1), 0.0)
        w = np.where(ok, n1 * n0 / np.maximum(n1 + n0, 1), 0.0)
        num += w * d
        den += w
    st = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    obs = st[0]
    ge = (np.abs(st[1:]) >= np.abs(obs)[None, :] - 1e-12).sum(0)
    p = np.where(np.isfinite(obs), (ge + 1) / (nperm + 1), np.nan)
    return obs, p


def strat_boot(Y, g, strata, nboot=None, seed=1, clusters=None):
    """Bootstrap CI of strat_stat, resampling clusters (individuals; units if None) within each block x group."""
    nboot = nboot or ARGS.nboot
    Y = np.atleast_2d(np.asarray(Y, float).T).T
    g = np.asarray(g, bool)
    strata = np.asarray(strata).astype(str)
    rng = np.random.default_rng(seed)
    _, gc, blk, members = _clusters(g, strata, clusters)
    cells = [np.where((blk == s) & (gc == v))[0] for s in np.unique(blk) for v in (True, False)]
    cells = [c for c in cells if len(c)]
    B = np.full((nboot, Y.shape[1]), np.nan)
    for b in range(nboot):
        ks = np.concatenate([rng.choice(c, len(c)) for c in cells])
        idx = np.concatenate([members[k] for k in ks])
        B[b] = strat_stat(Y[idx], g[idx], strata[idx])
    lo, hi = np.nanpercentile(B, [2.5, 97.5], axis=0)
    return lo, hi


def strat_compare(y, g, strata, nboot=None, nperm=None, seed=0, clusters=None):
    """Scalar within-stratum comparison: estimate, CI, permutation p, n per group in usable strata."""
    y = np.asarray(y, float)
    g = np.asarray(g, bool)
    strata = np.asarray(strata).astype(str)
    use = np.zeros(len(y), bool)
    for s in np.unique(strata):
        m = (strata == s) & np.isfinite(y)
        if (m & g).sum() >= 1 and (m & ~g).sum() >= 1:
            use |= m
    if (use & g).sum() < 3 or (use & ~g).sum() < 3:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n1=int((use & g).sum()), n0=int((use & ~g).sum()))
    cl = None if clusters is None else np.asarray(clusters).astype(str)[use]
    est, p = strat_perm_multi(y[use], g[use], strata[use], nperm, seed, clusters=cl)
    lo, hi = strat_boot(y[use], g[use], strata[use], nboot, seed + 1, clusters=cl)
    return dict(est=float(est[0]), lo=float(lo[0]), hi=float(hi[0]), p=float(p[0]), n1=int((use & g).sum()),
                n0=int((use & ~g).sum()), n1_individuals=int(len(set(cl[g[use]]))) if cl is not None else None,
                strata=sorted(set(strata[use])))


def exceed_test(y, g, strata, q, nperm=None, seed=3, clusters=None):
    """Fraction of g=1 units above the q-quantile of g=0 units (expected 1-q); permutation within strata (cluster
    level when clusters given) recomputing the threshold from the permuted g=0 group."""
    nperm = nperm or ARGS.nperm
    y = np.asarray(y, float)
    g = np.asarray(g, bool)
    strata = np.asarray(strata).astype(str)
    ok = np.isfinite(y)
    use = np.zeros(len(y), bool)
    for s in np.unique(strata):
        m = (strata == s) & ok
        if (m & g).sum() and (m & ~g).sum():
            use |= m
    cl = None if clusters is None else np.asarray(clusters).astype(str)[use]
    y, g, strata = y[use], g[use], strata[use]
    if g.sum() < 3 or (~g).sum() < 3:
        return dict(frac=np.nan, p=np.nan)

    def st(gg):
        if gg.all() or (~gg).all():
            return np.nan
        return float(np.mean(y[gg] > np.quantile(y[~gg], q)))
    s0 = st(g)
    rng = np.random.default_rng(seed)
    GP = perm_matrix(g, strata, cl, nperm, rng)
    vals = np.array([st(GP[i]) for i in range(1, nperm + 1)])
    vals = vals[np.isfinite(vals)]
    cnt = int((vals >= s0 - 1e-12).sum())
    return dict(frac=s0, expected=1 - q, p=(cnt + 1) / (len(vals) + 1), n1=int(g.sum()), n0=int((~g).sum()))


def boot_diff(a, b, nboot=None, seed=4):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(est=np.nan, lo=np.nan, hi=np.nan)
    rng = np.random.default_rng(seed)
    nb = nboot or ARGS.nboot
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(nb)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(est=float(a.mean() - b.mean()), lo=float(lo), hi=float(hi))


def perm_chi2(x, grp, nperm=5000, seed=5):
    """Monte-Carlo exact test of independence for a categorical x across groups (chi-square statistic)."""
    x = pd.Series(np.asarray(x).astype(str))
    grp = pd.Series(np.asarray(grp).astype(str))
    if x.nunique() < 2 or grp.nunique() < 2:
        return np.nan

    xc = pd.factorize(x.values)[0]
    gc = pd.factorize(grp.values)[0]
    nx, ngr = xc.max() + 1, gc.max() + 1

    def chi(gv):
        T = np.bincount(xc * ngr + gv, minlength=nx * ngr).reshape(nx, ngr).astype(float)
        E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
        return float(((T - E) ** 2 / np.maximum(E, 1e-12)).sum())
    c0 = chi(gc)
    rng = np.random.default_rng(seed)
    cnt = sum(chi(rng.permutation(gc)) >= c0 - 1e-9 for _ in range(nperm))
    return (cnt + 1) / (nperm + 1)


def ols_hc3(y, X, names, clusters=None):
    """OLS with HC3 SEs, or CR1 cluster-robust SEs (df = n_clusters - 1) when clusters are given."""
    X = np.column_stack([np.ones(len(y))] + list(X))
    names = ["Intercept"] + list(names)
    m = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[m], y[m]
    if clusters is not None:
        clusters = np.asarray(clusters).astype(str)[m]
    keep = []
    for j in range(X.shape[1]):
        if np.linalg.matrix_rank(X[:, keep + [j]]) == len(keep) + 1:
            keep.append(j)
    X = X[:, keep]
    names = [names[j] for j in keep]
    n, p = X.shape
    if n <= p + 2:
        return None
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    if clusters is not None and len(set(clusters)) < n:
        codes, uq = pd.factorize(clusters)
        Gn = len(uq)
        Sg = np.zeros((Gn, p))
        np.add.at(Sg, codes, X * e[:, None])
        V = XtXi @ (Sg.T @ Sg) @ XtXi * (Gn / max(Gn - 1, 1)) * ((n - 1) / max(n - p, 1))
        df = max(Gn - 1, 1)
    else:
        h = np.einsum("ij,jk,ik->i", X, XtXi, X)
        w = e ** 2 / np.clip(1 - h, 1e-6, None) ** 2
        V = XtXi @ ((X * w[:, None]).T @ X) @ XtXi
        df = n - p
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    tq = stats.t.ppf(0.975, df)
    pv = 2 * stats.t.sf(np.abs(b / np.where(se > 0, se, np.nan)), df)
    return pd.DataFrame(dict(term=names, coef=b, lo=b - tq * se, hi=b + tq * se, p=pv)).set_index("term")


# ----------------------------------------------------------------------------------------------
# metadata helpers
# ----------------------------------------------------------------------------------------------
def find_col(cols, exact=(), patterns=(), exclude=(r"ontology_term_id",)):
    lowmap = {c.lower(): c for c in cols}
    for e in exact:
        if e in cols:
            return e
        if e.lower() in lowmap:
            return lowmap[e.lower()]
    for p in patterns:
        for c in cols:
            if re.search(p, c, re.I) and not any(re.search(x, c, re.I) for x in exclude):
                return c
    return None


def parse_age(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*year", str(s))
    if m:
        return float(m.group(1))
    try:
        return float(str(s))
    except ValueError:
        return np.nan


def is_sle_str(s):
    return bool(re.search(r"lupus|\bsle\b", str(s), re.I))


def collapse_levels(s, min_n=3, na="unknown"):
    s = pd.Series(s).astype(object)
    s = s.where(s.notna(), na).astype(str).replace({"nan": na, "None": na, "": na})
    vc = s.value_counts()
    return s.where(~s.isin(vc[vc < min_n].index), "other")


def std_rate(num, n, w, min_cells):
    """num, n: units x labels DataFrames; direct standardization to label weights w over labels with n>=min_cells."""
    Rt = num / n.where(n >= min_cells)
    W = pd.Series(w).reindex(Rt.columns).fillna(0).values[None, :]
    valid = Rt.notna().values
    numv = np.nansum(Rt.values * W, 1)
    den = (W * valid).sum(1)
    return pd.Series(np.where(den > 0, numv / np.where(den > 0, den, 1), np.nan), index=Rt.index)


def healthy_ref_mask(meta):
    """Healthy units that define thresholds: healthy-query units from cohorts that also contain SLE units
    (healthy-only cohorts have different flag base rates and are never compared within cohort)."""
    h = meta.is_sle == 0
    coh = set(meta.cohort[meta.is_sle == 1].astype(str))
    hm = h & meta.cohort.astype(str).isin(coh)
    return hm if hm.sum() >= 10 else h


def collapse_rows(A, ids):
    """Average rows of A (n x K) sharing an individual id -> (n_ind x K); NaN-aware."""
    ids = np.asarray(ids).astype(str)
    if len(set(ids)) == len(ids):
        return A
    return pd.DataFrame(A).groupby(ids).mean().values


# ----------------------------------------------------------------------------------------------
# 0. load
# ----------------------------------------------------------------------------------------------
@section("0_load_flags")
def load_flags():
    seeds = [int(s) for s in ARGS.seeds.split(",") if s.strip()]
    per = {}
    for s in seeds:
        f = os.path.join(ARGS.root, ARGS.res_dir, f"{ARGS.tag}_seed{s}_cells.npz")
        if not os.path.exists(f):
            log("missing", f)
            continue
        C = np.load(f, allow_pickle=True)
        classes = np.asarray(C["classes"]).astype(str) if "classes" in C.files else None
        y = np.asarray(C["y"])
        if y.dtype.kind in "iu" and classes is not None:
            y = np.where(y >= 0, classes[np.clip(y, 0, len(classes) - 1)], "none")
        df = pd.DataFrame(index=pd.Index(np.asarray(C["names"]).astype(str), name="cell"))
        df["y"] = np.asarray(y).astype(str)
        df["cond_raw"] = np.asarray(C["cond"]).astype(str)
        df["donor"] = np.asarray(C["batch"]).astype(str)
        pc = np.asarray(C["p_unknown_ctrl"], float)
        pb = np.asarray(C["p_unknown_bonf"], float) if "p_unknown_bonf" in C.files else np.ones_like(pc)
        if "p_unknown_bonf" not in C.files:
            log(f"seed {s}: p_unknown_bonf missing -> strict flag empty")
        df["ctrl"] = pc <= ARGS.alpha
        df["strict"] = df["ctrl"].values & (pb <= ARGS.alpha)
        per[s] = df
        log(f"seed {s}: {len(df)} cells, ctrl {df.ctrl.mean():.4f}, strict {df.strict.mean():.4f}")
    if not per:
        raise FileNotFoundError("no cells npz found")
    ss = sorted(per)
    if len(ss) % 2 == 0:
        log(f"WARNING: {len(ss)} seeds (even): majority flag requires > half, ties count as NOT flagged")
    if len(ss) < len(seeds):
        log(f"WARNING: only seeds {ss} of {seeds} found")
    D = per[ss[0]][["y", "cond_raw", "donor"]].copy()
    common = D.index
    for df in per.values():
        common = common.intersection(df.index)
    if not D.index.is_unique:
        log("WARNING: non-unique cell names; using seed-0 order (all seeds must have the same order)")
        common = D.index
    D = D.loc[common] if D.index.is_unique else D
    for k in ["ctrl", "strict"]:
        M = np.vstack([(per[s].loc[common, k].values if D.index.is_unique else per[s][k].values) for s in ss])
        D[k] = M.mean(0) > 0.5
        for i, s in enumerate(ss):
            D[f"{k}_s{s}"] = M[i]
    D["state"] = np.where(D.strict.values, 2, np.where(D.ctrl.values, 1, 0)).astype(np.int8)
    D["is_sle"] = D.cond_raw.map(is_sle_str).astype(int)
    R["seeds"] = ss
    R["overview"] = dict(n_cells=int(len(D)), n_units=int(D.donor.nunique()),
                         ctrl_rate={k: float(v) for k, v in D.groupby("is_sle").ctrl.mean().items()},
                         strict_rate={k: float(v) for k, v in D.groupby("is_sle").strict.mean().items()})
    log(R["overview"])
    return D


def _symbols(var_names, var_df):
    var = pd.Index(np.asarray(var_names).astype(str))
    if np.mean(var.str.startswith("ENS")) > 0.5 and var_df is not None:
        for c in ["feature_name", "gene_symbols", "gene_symbol", "symbol", "gene_name"]:
            if c in var_df.columns:
                log(f"using var['{c}'] as gene symbols")
                return pd.Index(var_df[c].astype(str).values)
    return var


@section("0_stream_expression")
def stream(D, isg_genes):
    """One pass over h5ad rows (query + subsampled ref).  Returns dict with pseudobulk sums, ISG raw scores, obs."""
    import anndata as ad
    h5 = ARGS.h5ad or os.path.join(ARGS.root, "data", "std", f"{ARGS.tag}.h5ad")
    log("reading", h5)
    try:
        A = ad.read_h5ad(h5, backed="r")
        backed = True
    except Exception as e:  # noqa: BLE001
        log(f"backed read failed ({e!r}); loading fully")
        A = ad.read_h5ad(h5)
        backed = False
    obs_all = A.obs
    names = pd.Index(np.asarray(A.obs_names).astype(str))
    cols = list(obs_all.columns)
    rc, lc = find_col(cols, ["role"]), find_col(cols, ["label"])
    role = obs_all[rc].astype(str).values if rc else np.array(["query"] * len(names))
    qpos = None
    if names.is_unique and D.index.is_unique:
        qpos = names.get_indexer(D.index)
        if (qpos < 0).any():
            log(f"WARNING: {(qpos < 0).sum()} npz cell names not in h5ad obs_names; trying positional alignment")
            qpos = None
    if qpos is None:
        qpos = np.where(role == "query")[0]
        if len(qpos) != len(D):
            raise ValueError(f"cannot align npz ({len(D)}) to h5ad query cells ({len(qpos)}) by name or position")
        R["join_mode"] = "position"
        log("WARNING: aligning npz to h5ad query cells by position (validated by donor id below)")
    else:
        R["join_mode"] = "name"
    if len(np.unique(qpos)) != len(qpos):
        raise ValueError("npz cells map to duplicated h5ad rows")
    bc = find_col(cols, ["batch"])
    if bc:
        mis = obs_all[bc].astype(str).values[qpos] != D.donor.values
        R["join_donor_mismatch"] = int(mis.sum())
        if mis.mean() > 0.001:
            raise ValueError(f"npz/h5ad join broken: {mis.sum()} cells with different donor id")
    else:
        log("WARNING: no batch column in h5ad; join not validated by donor")
    # label check: npz y must be the same fine label as h5ad obs label (ISG z uses ref cells of the h5ad label)
    if lc:
        hl = obs_all[lc].astype(str).values[qpos]
        agree = float(np.mean(hl == D.y.values))
        R["join_label_agreement"] = agree
        if agree < 0.99:
            log(f"WARNING: npz y agrees with h5ad '{lc}' for only {agree:.3f} of cells (int-coded y / different "
                f"label space?) -> using h5ad labels for query cells")
            D["y"] = hl
    rng = np.random.default_rng(0)
    rpos = np.where(role == "ref")[0]
    if ARGS.max_ref_cells and len(rpos) > ARGS.max_ref_cells:
        rpos = np.sort(rng.choice(rpos, ARGS.max_ref_cells, replace=False))
    ref_label = obs_all[lc].astype(str).values[rpos] if lc else np.array(["all"] * len(rpos))
    log(f"query cells {len(qpos)}, reference cells used {len(rpos)}")
    keep_cols = [c for c in [find_col(cols, ["Processing_Cohort", "processing_cohort", "cohort"], [r"cohort"]),
                             find_col(cols, ["self_reported_ethnicity", "pop_cov", "ancestry", "ethnicity"],
                                      [r"ethnic", r"ancestr"]),
                             find_col(cols, ["sex", "sex_cov", "gender"], [r"^sex"]),
                             find_col(cols, ["Age", "age", "age_cov"], [r"^age"]),
                             find_col(cols, ["development_stage"]),
                             find_col(cols, ["disease_state", "Status", "status"], [r"disease_state", r"flare"]),
                             find_col(cols, ["donor_id", "ind_cov", "individual"], [r"^donor_id$", r"ind_cov$"])]
                 if c]
    keep_cols = list(dict.fromkeys(keep_cols))
    obs = obs_all.iloc[qpos][keep_cols].copy()
    for c in keep_cols:
        obs[c] = obs[c].astype(str).values
    obs.index = D.index
    R["obs_columns"] = keep_cols
    var = _symbols(A.var_names, A.var)
    vser = pd.Series(np.arange(len(var)), index=var)
    vser = vser[~vser.index.duplicated()]
    isg_idx = np.array(sorted(vser.reindex([g for g in isg_genes if g in vser.index]).values.tolist()), int)
    R["n_isg_genes"] = int(len(isg_idx))
    log(f"ISG genes found: {len(isg_idx)}")
    if len(isg_idx) < 3:
        raise ValueError("fewer than 3 ISG genes in var")
    # group codes for query cells
    units = sorted(D.donor.unique())
    labels = sorted(D.y.unique())
    nU, nT, G = len(units), len(labels), len(var)
    code_q = (pd.Index(units).get_indexer(D.donor.values) * nT + pd.Index(labels).get_indexer(D.y.values)) * 3 \
        + D.state.values.astype(int)
    ng = nU * nT * 3
    log(f"pseudobulk accumulator {ng} x {G} float64 = {ng * G * 8 / 1e9:.2f} GB")
    S = np.zeros((ng, G), np.float64)
    N = np.bincount(code_q, minlength=ng).astype(float)
    isg_q = np.full(len(qpos), np.nan)
    isg_r = np.full(len(rpos), np.nan)
    allpos = np.concatenate([qpos, rpos])
    kind = np.concatenate([np.zeros(len(qpos), np.int8), np.ones(len(rpos), np.int8)])
    loc = np.concatenate([np.arange(len(qpos)), np.arange(len(rpos))])
    order = np.argsort(allpos, kind="stable")
    sp_pos = allpos[order]
    try:
        is_sparse = sp.issparse(A.X[0:2])
    except Exception:  # noqa: BLE001
        is_sparse = True
    win = ARGS.chunk if is_sparse else max(1000, int(2.5e8 / max(G, 1)))
    wid = sp_pos // win
    bounds = np.flatnonzero(np.diff(wid)) + 1
    done = 0
    nonint = False
    for oi in np.split(np.arange(len(sp_pos)), bounds):
        if not len(oi):
            continue
        rows = sp_pos[oi]
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        try:
            Xi = A.X[lo:hi][rows - lo] if backed else A.X[rows]
        except Exception as e:  # noqa: BLE001
            if not backed:
                raise
            log(f"backed row read failed ({e!r}); loading h5ad into memory")
            A = ad.read_h5ad(h5)
            backed = False
            if sp.issparse(A.X) and not sp.isspmatrix_csr(A.X):
                A.X = sp.csr_matrix(A.X)
            Xi = A.X[rows]
        Xi = sp.csr_matrix(Xi, dtype=np.float32)
        if not nonint and len(Xi.data) and not np.allclose(Xi.data[:50000], np.round(Xi.data[:50000])):
            nonint = True
            log("WARNING: X not integer counts; treated as counts anyway")
        tot = np.asarray(Xi.sum(1)).ravel()
        Xi.data *= np.repeat((1e4 / np.maximum(tot, 1)).astype(np.float32), np.diff(Xi.indptr))  # CP10k
        sc = np.asarray(np.log1p(Xi[:, isg_idx].toarray()).mean(1)).ravel()
        k, l_ = kind[order[oi]], loc[order[oi]]
        isg_q[l_[k == 0]] = sc[k == 0]
        isg_r[l_[k == 1]] = sc[k == 1]
        qi = np.where(k == 0)[0]
        if len(qi):
            cq = code_q[l_[qi]]
            u, inv = np.unique(cq, return_inverse=True)
            Ind = sp.csr_matrix((np.ones(len(qi), np.float32), (inv, np.arange(len(qi)))), shape=(len(u), len(qi)))
            S[u] += np.asarray((Ind @ Xi[qi]).toarray(), dtype=np.float64)
        done += len(oi)
        if done % (10 * win) < len(oi) or done == len(sp_pos):
            log(f"  rows {done}/{len(sp_pos)}  [{time.time() - T0:.0f}s]")
    del A
    # collapse duplicated symbols, drop all-zero genes
    if not var.is_unique:
        u, inv = np.unique(np.asarray(var), return_inverse=True)
        M = sp.csr_matrix((np.ones(len(var)), (np.arange(len(var)), inv)), shape=(len(var), len(u)))
        S = np.asarray(M.T @ S.T).T
        var = pd.Index(u)
        log("summed duplicated gene symbols ->", len(var))
    nz = S.sum(0) > 0
    S = S[:, nz]
    var = var[nz]
    log(f"genes with counts: {len(var)}")
    return dict(S=S.reshape(nU, nT, 3, -1), N=N.reshape(nU, nT, 3), units=units, labels=labels, var=var,
                isg_q=isg_q, isg_r=isg_r, ref_label=ref_label, obs=obs)


def load_genesets():
    p = ARGS.genesets if os.path.isabs(ARGS.genesets) else os.path.join(ARGS.root, ARGS.genesets)
    gs = json.load(open(p))
    gs = {str(k): [str(g) for g in v] for k, v in gs.items() if isinstance(v, (list, tuple))}
    focus = dict(EXTRA_SETS)
    used = {}
    for name, pats in FOCUS_FROM_GS.items():
        for pt in pats:
            hit = [k for k in gs if re.search(pt, k, re.I)]
            if hit:
                focus[name] = gs[hit[0]]
                used[name] = hit[0]
                break
    R["focus_set_source"] = used
    allsets = dict(gs)
    for k, v in focus.items():
        allsets.setdefault(k, v)
    log(f"gene sets {len(gs)} from {p}; focus sets {list(focus)}")
    return gs, focus, allsets


# ----------------------------------------------------------------------------------------------
# 1. donor table, groups, ISG-low vs healthy
# ----------------------------------------------------------------------------------------------
@section("1_donor_table")
def donor_table(D, E):
    obs = E["obs"]
    lab = D.y.values
    # ISG z of each query cell vs reference cells of the same label
    rl = pd.Series(E["isg_r"]).groupby(E["ref_label"]).agg(["mean", "std", "size"])
    gmu, gsd = np.nanmean(E["isg_r"]), np.nanstd(E["isg_r"])
    mu = pd.Series(lab).map(rl["mean"].where(rl["size"] >= ARGS.min_ref_label)).fillna(gmu).values
    sd = pd.Series(lab).map(rl["std"].where(rl["size"] >= ARGS.min_ref_label)).fillna(gsd).values
    D["isg_z"] = (E["isg_q"] - mu) / np.maximum(sd, 1e-3)
    R["ref_labels_pooled"] = sorted(set(lab) - set(rl.index[rl["size"] >= ARGS.min_ref_label]))
    cols = R.get("obs_columns", [])
    meta = pd.DataFrame(index=pd.Index(E["units"], name="unit"))
    keys = dict(cohort=find_col(cols, ["Processing_Cohort", "processing_cohort", "cohort"], [r"cohort"]),
                ancestry=find_col(cols, ["self_reported_ethnicity", "pop_cov", "ancestry", "ethnicity"],
                                  [r"ethnic", r"ancestr"]),
                sex=find_col(cols, ["sex", "sex_cov", "gender"], [r"^sex"]),
                age=find_col(cols, ["Age", "age", "age_cov"], [r"^age"]),
                dev=find_col(cols, ["development_stage"]),
                status=find_col(cols, ["disease_state", "Status", "status"], [r"disease_state", r"flare"]),
                individual=find_col(cols, ["donor_id", "ind_cov", "individual"], [r"^donor_id$", r"ind_cov$"]))
    R["meta_columns"] = keys
    o = obs.copy()
    o["_u"] = D.donor.values
    for k, c in keys.items():
        if c and c in o:
            meta[k] = o.groupby("_u")[c].agg(lambda s: s.value_counts().index[0]).reindex(meta.index)
    if keys.get("status") and keys["status"] in o:
        meta["status_mixed"] = (o.groupby("_u")[keys["status"]].nunique() > 1).reindex(meta.index)
    meta["age_num"] = meta["age"].map(parse_age) if "age" in meta else (
        meta["dev"].map(parse_age) if "dev" in meta else np.nan)
    if "individual" not in meta:
        meta["individual"] = meta.index.astype(str)
    ind = meta["individual"].astype(str)
    miss = ind.isin(["nan", "None", "", "unknown", "NA", "na"])
    meta["individual"] = np.where(miss, "unit:" + meta.index.astype(str), ind)
    if "cohort" not in meta:
        meta["cohort"] = "all"
    meta["cohort"] = meta["cohort"].astype(str)
    meta["is_sle"] = D.groupby("donor").is_sle.agg(lambda v: int(round(v.mean()))).reindex(meta.index)
    n = D.groupby(["donor", "y"]).size().unstack(fill_value=0).reindex(meta.index).fillna(0)
    healthy = meta.index[meta.is_sle == 0]
    prop = n.div(n.sum(1), axis=0)
    w = prop.loc[healthy].mean(0)
    w = w / w.sum()
    R["label_weights"] = w.round(4).to_dict()
    meta["n_cells"] = n.sum(1)
    zsum = D.groupby(["donor", "y"]).isg_z.sum().unstack(fill_value=0).reindex_like(n).fillna(0)
    meta["isg"] = std_rate(zsum, n, w, ARGS.min_cells)
    meta["isg_raw"] = D.groupby("donor").isg_z.mean().reindex(meta.index)
    flagcols = ["ctrl", "strict"] + [c for c in D.columns if re.match(r"(ctrl|strict)_s\d+$", c)]
    tabs = {}
    for c in flagcols:
        k = D.groupby(["donor", "y"])[c].sum().unstack(fill_value=0).reindex_like(n).fillna(0)
        tabs[c] = k
        meta[c] = std_rate(k, n, w, ARGS.min_cells)
        meta[c + "_raw"] = k.sum(1) / n.sum(1)
    wcsv(meta.reset_index(), "donor_table.csv")
    R["step1_overview"] = dict(n_sle=int(meta.is_sle.sum()), n_healthy=int((meta.is_sle == 0).sum()),
                               n_individuals=int(meta.individual.nunique()),
                               isg_median={"SLE": float(meta.isg[meta.is_sle == 1].median()),
                                           "healthy": float(meta.isg[meta.is_sle == 0].median())},
                               ctrl_median={"SLE": float(meta.ctrl[meta.is_sle == 1].median()),
                                            "healthy": float(meta.ctrl[meta.is_sle == 0].median())})
    log(R["step1_overview"])
    return meta, n, tabs, w


def thresholds(meta, flagcol, q, healthy_mask=None):
    h = healthy_ref_mask(meta) if healthy_mask is None else healthy_mask
    return float(np.nanquantile(meta.loc[h, "isg"], q)), float(np.nanquantile(meta.loc[h, flagcol], q))


def assign_groups(meta, flagcol, ti, tf):
    g = pd.Series("healthy", index=meta.index, dtype=object)
    s = meta.is_sle == 1
    isg_hi = meta.isg > ti
    fl_hi = meta[flagcol] > tf
    g[s & isg_hi] = "ISGhigh"
    g[s & ~isg_hi & fl_hi] = "ISGlow_flagHigh"
    g[s & ~isg_hi & ~fl_hi] = "ISGlow_flagLow"
    g[s & meta.isg.isna()] = "NA"
    return g


def compare_isglow(meta, flagcol, ti, tf, q, seed=0, healthy_isg_matched=False, extra_strata=None):
    """ISG-low SLE vs healthy units, within cohort; permutation / bootstrap at the individual level.
    healthy_isg_matched: healthy restricted to ISG <= ti AND strata = cohort x ISG tertile (pooled), so that the
    residual ISG difference inside the 'ISG-low' range cannot drive the flag difference.
    extra_strata: obs column (e.g. ancestry) crossed with cohort."""
    s = (meta.is_sle == 1) & (meta.isg <= ti)
    h = meta.is_sle == 0
    if healthy_isg_matched:
        h = h & (meta.isg <= ti)
    m = (s | h).values
    y = meta[flagcol].values[m]
    g = s.values[m]
    st = meta.cohort.astype(str).values[m]
    if extra_strata is not None and extra_strata in meta:
        st = np.char.add(np.char.add(st.astype(str), "|"),
                         collapse_levels(meta[extra_strata].values[m], 3).values.astype(str))
    if healthy_isg_matched:
        b = pd.qcut(meta.isg.values[m], 3, labels=False, duplicates="drop")
        b = np.where(np.isfinite(b), b, -1).astype(int).astype(str)
        st = np.char.add(np.char.add(st.astype(str), "|isg"), b)
    cl = meta.individual.astype(str).values[m] if "individual" in meta else None
    out = strat_compare(y, g, st, seed=seed, clusters=cl)
    out["exceed"] = exceed_test(y, g, st, q, seed=seed + 5, clusters=cl)
    out["median_isglow"] = float(np.nanmedian(y[g])) if g.any() else np.nan
    out["median_healthy"] = float(np.nanmedian(y[~g])) if (~g).any() else np.nan
    return out


@section("1_groups_and_comparison")
def step1(meta):
    qs = [float(x) for x in ARGS.qs.split(",")]
    res, rows = {}, []
    for fc in ["ctrl", "strict"]:
        for q in qs:
            ti, tf = thresholds(meta, fc, q)
            grp = assign_groups(meta, fc, ti, tf)
            if fc == "ctrl" and q == qs[0]:
                meta["group"] = grp
            meta[f"group_{fc}_q{int(round(q * 100))}"] = grp
            cmp_ = compare_isglow(meta, fc, ti, tf, q, seed=int(q * 100))
            cmp_m = compare_isglow(meta, fc, ti, tf, q, seed=int(q * 100) + 7, healthy_isg_matched=True)
            vc = grp[meta.is_sle == 1].value_counts().to_dict()
            key = f"{fc}_q{int(round(q * 100))}"
            res[key] = dict(thr_isg=ti, thr_flag=tf, groups=vc, isglow_vs_healthy=cmp_, vs_isg_matched_healthy=cmp_m)
            rows.append(dict(flag=fc, q=q, thr_isg=ti, thr_flag=tf, **{f"n_{k}": v for k, v in vc.items()},
                             diff=cmp_["est"], lo=cmp_["lo"], hi=cmp_["hi"], p_perm=cmp_["p"], n_isglow=cmp_["n1"],
                             n_healthy=cmp_["n0"], exceed_frac=cmp_["exceed"].get("frac"),
                             exceed_p=cmp_["exceed"].get("p"), diff_matched=cmp_m["est"], lo_m=cmp_m["lo"],
                             hi_m=cmp_m["hi"], p_matched=cmp_m["p"]))
            log(key, res[key])
    wcsv(pd.DataFrame(rows), "step1_isglow_vs_healthy.csv")
    # sensitivity: one unit per individual (unit with most cells)
    try:
        one = meta.sort_values("n_cells", ascending=False).drop_duplicates("individual")
        ti, tf = thresholds(one, "ctrl", qs[0])
        res["ctrl_one_unit_per_individual"] = compare_isglow(one, "ctrl", ti, tf, qs[0], seed=11)
    except Exception as e:  # noqa: BLE001
        res["ctrl_one_unit_per_individual"] = repr(e)
    # sensitivity: strata = cohort x ancestry (ancestry confounding)
    try:
        if "ancestry" in meta:
            ti, tf = thresholds(meta, "ctrl", qs[0])
            res["ctrl_cohort_x_ancestry"] = compare_isglow(meta, "ctrl", ti, tf, qs[0], seed=13,
                                                           extra_strata="ancestry")
    except Exception as e:  # noqa: BLE001
        res["ctrl_cohort_x_ancestry"] = repr(e)
    # OLS: flag ~ ISG-low SLE + ISG + cohort [+ ancestry + sex + age] (ISG-low SLE + healthy units),
    # cluster-robust SEs by individual
    try:
        ti, _ = thresholds(meta, "ctrl", qs[0])
        m = (meta.is_sle == 0) | ((meta.is_sle == 1) & (meta.isg <= ti))
        d = meta[m]
        X, nm = [d.is_sle.values.astype(float), d.isg.values], ["isglow_sle", "isg"]
        for cov in ["cohort", "ancestry", "sex"]:
            if cov in d:
                dm = pd.get_dummies(collapse_levels(d[cov], 3).values, drop_first=True).astype(float)
                X += [dm[c].values for c in dm.columns]
                nm += [f"{cov}_{c}" for c in dm.columns]
        if "age_num" in d and np.isfinite(d.age_num.astype(float)).mean() > 0.9:
            a = d.age_num.astype(float)
            X.append(a.fillna(a.median()).values)  # few missing ages -> median (logged)
            nm.append("age")
            R["ols_age_imputed"] = int(a.isna().sum())
        T = ols_hc3(d.ctrl.values.astype(float), X, nm, clusters=d.individual.values)
        if T is not None:
            wcsv(T.reset_index(), "step1_ols_isglow.csv")
            res["ols_isglow_adj_isg_cohort"] = T.loc["isglow_sle"].to_dict()
            res["ols_covariates"] = nm[1:]
    except Exception as e:  # noqa: BLE001
        res["ols_error"] = repr(e)
    R["step1"] = res
    # figure: ISG vs flag
    try:
        ti, tf = res[f"ctrl_q{int(round(qs[0] * 100))}"]["thr_isg"], res[f"ctrl_q{int(round(qs[0] * 100))}"]["thr_flag"]
        fig, ax = plt.subplots(figsize=(8, 6.5))
        col = {"healthy": "#4c72b0", "ISGhigh": "#c44e52", "ISGlow_flagHigh": "#dd8452", "ISGlow_flagLow": "#8c8c8c"}
        for gname, c in col.items():
            mm = meta.group == gname
            ax.scatter(meta.isg[mm], 100 * meta.ctrl[mm], s=40, c=c, alpha=.8, label=f"{gname} ({mm.sum()})",
                       edgecolors="none")
        ax.axvline(ti, ls="--", c="k", lw=1)
        ax.axhline(100 * tf, ls="--", c="k", lw=1)
        ax.set_xlabel("donor ISG z (vs reference)")
        ax.set_ylabel("flagged cells, std. (%)")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=12)
        savefig(fig, "fig1_isg_vs_flag.png")
    except Exception as e:  # noqa: BLE001
        log("fig1 failed", repr(e))
    return res


# ----------------------------------------------------------------------------------------------
# 2. which labels / states carry the excess
# ----------------------------------------------------------------------------------------------
@section("2a_label_rates")
def step2_rates(meta, n, tabs, w):
    qs = [float(x) for x in ARGS.qs.split(",")]
    ti, _ = thresholds(meta, "ctrl", qs[0])
    isglow = (meta.is_sle == 1) & (meta.isg <= ti)
    fh = meta.group == "ISGlow_flagHigh"
    h = meta.is_sle == 0
    out = {}
    for fc in ["ctrl", "strict"]:
        rate = (tabs[fc] / n.where(n >= ARGS.min_cells))
        rows = []
        for grp_name, gm in [("ISGlow", isglow), ("ISGlow_flagHigh", fh)]:
            m = (gm | h).values
            Y = rate.values[m]
            g = gm.values[m]
            st = meta.cohort.values[m]
            cl = meta.individual.astype(str).values[m]
            est, p = strat_perm_multi(Y, g, st, seed=21, clusters=cl)
            lo, hi = strat_boot(Y, g, st, seed=22, clusters=cl)
            for j, t in enumerate(rate.columns):
                a, b = rate.loc[gm, t].values, rate.loc[h, t].values
                na, nb = np.isfinite(a).sum(), np.isfinite(b).sum()
                # ISGlow_flagHigh is selected on a high flag fraction -> its flag-rate test is circular (descriptive)
                rows.append(dict(group=grp_name, selected_on_flag=grp_name == "ISGlow_flagHigh",
                                 label=t, weight=float(w.get(t, 0)), n_sle=int(na), n_h=int(nb),
                                 mean_sle=np.nanmean(a) if na else np.nan, mean_h=np.nanmean(b) if nb else np.nan,
                                 diff_raw=(np.nanmean(a) - np.nanmean(b)) if na and nb else np.nan,
                                 diff_within_cohort=est[j], lo=lo[j], hi=hi[j],
                                 p_perm=p[j] if na >= ARGS.min_donors and nb >= ARGS.min_donors else np.nan,
                                 p_mwu=mwu_p(a, b)))
        T = pd.DataFrame(rows)
        T["q"] = np.nan
        for gname in T.group.unique():
            mm = T.group == gname
            T.loc[mm, "q"] = bh(T.loc[mm, "p_perm"].values)
            contrib = T.loc[mm, "weight"] * T.loc[mm, "diff_raw"].fillna(0)
            T.loc[mm, "share_of_excess"] = contrib / contrib.sum() if contrib.sum() != 0 else np.nan
        T = T.sort_values(["group", "share_of_excess"], ascending=[True, False])
        wcsv(T, f"step2_label_rates_{fc}.csv")
        out[fc] = T
    R["step2_label_rates"] = {fc: T[T.group == "ISGlow"].head(8)[
        ["label", "mean_sle", "mean_h", "diff_within_cohort", "lo", "hi", "p_perm", "q", "share_of_excess"]].to_dict(
        orient="records") for fc, T in out.items()}
    try:
        T = out["ctrl"]
        T = T[(T.group == "ISGlow") & T.diff_within_cohort.notna()].sort_values("diff_within_cohort")
        fig, ax = plt.subplots(figsize=(8, max(5, 0.38 * len(T))))
        yy = np.arange(len(T))
        ax.errorbar(100 * T.diff_within_cohort, yy, xerr=[100 * (T.diff_within_cohort - T.lo),
                                                          100 * (T.hi - T.diff_within_cohort)],
                    fmt="o", c="#dd8452", ms=7, capsize=3)
        ax.axvline(0, c="k", lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{t}{' *' if q < 0.1 else ''}" for t, q in zip(T.label, T.q.fillna(1))])
        ax.set_xlabel("flag rate ISG-low SLE - healthy\n(within cohort, % points)")
        savefig(fig, "fig2_label_flag_excess.png")
    except Exception as e:  # noqa: BLE001
        log("fig2 failed", repr(e))
    return out


def label_mats(E, b):
    """Per label b: kept / flagged(ctrl) / flagged(strict) / all mean CP10k (units x G) and counts."""
    S, N = E["S"][:, b], E["N"][:, b]
    kept = S[:, 0]
    fc = S[:, 1] + S[:, 2]
    fs = S[:, 2]
    nk, nfc, nfs = N[:, 0], N[:, 1] + N[:, 2], N[:, 2]
    M = dict(kept=kept / np.maximum(nk, 1)[:, None], ctrl=fc / np.maximum(nfc, 1)[:, None],
             strict=fs / np.maximum(nfs, 1)[:, None], all=(S.sum(1)) / np.maximum(N.sum(1), 1)[:, None])
    return M, dict(kept=nk, ctrl=nfc, strict=nfs, all=N.sum(1))


@section("2b_lfc_profiles")
def step2_lfc(meta, E):
    qs = [float(x) for x in ARGS.qs.split(",")]
    ti, _ = thresholds(meta, "ctrl", qs[0])
    units = pd.Index(E["units"])
    m_ = meta.reindex(units)
    groups = [("ISGlow", ((m_.is_sle == 1) & (m_.isg <= ti)).values),
              ("ISGlow_flagHigh", (m_.group == "ISGlow_flagHigh").values)]
    h = (m_.is_sle == 0).values
    ids = m_.individual.astype(str).values
    var = np.asarray(E["var"])
    rows, summ = [], []
    for fc in ["ctrl", "strict"]:
        for b, t in enumerate(E["labels"]):
            M, Nn = label_mats(E, b)
            valid = (Nn[fc] >= ARGS.min_flag) & (Nn["kept"] >= ARGS.min_kept)
            h_ = valid & h
            for gname, gm in groups:
                a_ = valid & gm
                base = dict(group=gname, flag=fc, label=t, n_sle=int(a_.sum()), n_h=int(h_.sum()))
                if a_.sum() < ARGS.min_donors:
                    summ.append(dict(base, n_genes=0))
                    continue
                expr = M["all"][a_ | h_].mean(0) >= ARGS.min_mean
                gi = np.where(expr)[0]
                L_ = np.log2(M[fc][:, gi] + ARGS.pc) - np.log2(M["kept"][:, gi] + ARGS.pc)
                La = collapse_rows(L_[a_], ids[a_])  # one row per individual (no pseudoreplication)
                za, pa = signed_rank(La)
                hok = h_.sum() >= ARGS.min_donors
                if hok:
                    Lh = collapse_rows(L_[h_], ids[h_])
                    ph = stats.mannwhitneyu(La, Lh, axis=0, alternative="two-sided")[1]
                    dh, mh = La.mean(0) - Lh.mean(0), Lh.mean(0)
                else:
                    ph = dh = mh = np.full(len(gi), np.nan)
                T = pd.DataFrame(dict(group=gname, flag=fc, label=t, gene=var[gi], lfc=La.mean(0), z=za, p=pa,
                                      q=bh(pa), lfc_healthy=mh, diff_vs_healthy=dh, p_diff=ph, q_diff=bh(ph),
                                      n_sle=int(a_.sum()), n_h=int(h_.sum())))
                # group-specific up genes: up in flagged vs kept and more than in healthy flagged-vs-kept
                spec = T[(T.lfc > 0) & (T.q < 0.25)].copy()
                spec["rankscore"] = spec.z * (np.sign(spec.diff_vs_healthy).clip(lower=0) if hok else 1.0)
                top = spec.sort_values(["rankscore", "z"], ascending=False).head(ARGS.top_genes)
                dn = T[(T.lfc < 0) & (T.q < 0.25)].sort_values("z").head(ARGS.top_genes // 4)
                rows.append(pd.concat([top, dn]))
                summ.append(dict(base, n_genes=int(len(gi)), n_up_q10=int(((T.lfc > 0) & (T.q < .1)).sum()),
                                 n_specific_q10=int(((T.diff_vs_healthy > 0) & (T.q_diff < .1)).sum()),
                                 top_up=",".join((top[top.diff_vs_healthy > 0] if hok else top).gene.head(10)),
                                 top_specific_q10=",".join(T[(T.diff_vs_healthy > 0) & (T.q_diff < .1)].sort_values(
                                     "p_diff").gene.head(10))))
    if rows:
        wcsv(pd.concat(rows), "step2_lfc_top_genes.csv")
    Sm = pd.DataFrame(summ)
    wcsv(Sm, "step2_lfc_summary.csv")
    out = {}
    if len(Sm) and "n_up_q10" in Sm:
        for gname in ["ISGlow", "ISGlow_flagHigh"]:
            x = Sm[(Sm.flag == "ctrl") & (Sm.group == gname) & (Sm.n_genes > 0)]
            out[gname] = x.sort_values(["n_specific_q10", "n_up_q10"], ascending=False).head(8)[
                ["label", "n_sle", "n_h", "n_up_q10", "n_specific_q10", "top_specific_q10", "top_up"]].to_dict(
                orient="records")
    R["step2_lfc"] = out
    return Sm


@section("2c_geneset_scores")
def step2_sets(meta, E, allsets, focus, w):
    qs = [float(x) for x in ARGS.qs.split(",")]
    ti, _ = thresholds(meta, "ctrl", qs[0])
    units = pd.Index(E["units"])
    m_ = meta.reindex(units)
    isglow = ((m_.is_sle == 1) & (m_.isg <= ti)).values
    fh = (m_.group == "ISGlow_flagHigh").values
    h = (m_.is_sle == 0).values
    vi = pd.Index(E["var"])
    setnames = list(dict.fromkeys(list(focus) + list(allsets)))
    nU, nT = len(units), len(E["labels"])
    out = {}
    for fc in ["ctrl", "strict"]:
        LEV = np.full((nU, nT, len(setnames)), np.nan)
        DEL = np.full((nU, nT, len(setnames)), np.nan)
        ALL = np.full((nU, nT, len(setnames)), np.nan)
        NF = np.zeros((nU, nT))
        for b, t in enumerate(E["labels"]):
            M, Nn = label_mats(E, b)
            hk = h & (Nn["kept"] >= ARGS.min_kept)
            if hk.sum() < 3:
                continue
            expr = M["all"].mean(0) >= ARGS.min_mean * 0.5
            vf = Nn[fc] >= ARGS.min_flag_sets
            vd = vf & (Nn["kept"] >= ARGS.min_kept)
            NF[:, b] = np.where(vf, Nn[fc], 0)
            va = Nn["all"] >= ARGS.min_cells
            hka = h & va
            for k, sname in enumerate(setnames):
                gi = vi.get_indexer([g for g in allsets[sname] if g in vi])
                gi = gi[gi >= 0]
                gi = gi[expr[gi]]
                if len(gi) < 2:
                    continue
                # set-level score = log2 of the summed set expression of the pseudobulk (genes aggregated BEFORE
                # the log: per-gene log of sparse means from few flagged cells is biased downward, and SLE donors
                # have several-fold more flagged cells than healthy -> spurious "up" for lowly expressed sets)
                lk = np.log2(M["kept"][:, gi].sum(1) + ARGS.pc)
                lf = np.log2(M[fc][:, gi].sum(1) + ARGS.pc)
                la = np.log2(M["all"][:, gi].sum(1) + ARGS.pc)
                sd = max(float(np.std(lk[hk], ddof=1)), ARGS.sd_floor)
                LEV[vf, b, k] = (lf[vf] - lk[hk].mean()) / sd
                DEL[vd, b, k] = (lf[vd] - lk[vd]) / sd
                if hka.sum() >= 3:
                    sda = max(float(np.std(la[hka], ddof=1)), ARGS.sd_floor)
                    ALL[va, b, k] = (la[va] - la[hka].mean()) / sda
        W = pd.Series(w).reindex(E["labels"]).fillna(0).values

        def dstd(A, Wm):
            V = np.isfinite(A)
            num = np.nansum(np.where(V, A, 0) * Wm[:, :, None], 1)
            den = (V * Wm[:, :, None]).sum(1)
            return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        # all scores standardized to the healthy label mix over the labels valid for that donor (weighting by the
        # donor's own flagged-cell counts would let label composition of flagged cells, which differs by group by
        # design, drive the group difference)
        Wm = np.repeat(W[None, :], nU, 0)
        DL, DD = dstd(LEV, Wm), dstd(DEL, Wm)
        DA = dstd(ALL, Wm)
        R.setdefault("step2_median_flagged_cells", {})[fc] = dict(
            isglow=float(np.median(NF[isglow].sum(1))) if isglow.any() else None,
            healthy=float(np.median(NF[h].sum(1))) if h.any() else None)
        if fc == "ctrl":
            for k, sname in enumerate(setnames):
                if sname in focus:
                    meta[f"allcells_{sname}"] = pd.Series(DA[:, k], index=units).reindex(meta.index)
        rows = []
        for gname, gm in [("ISGlow", isglow), ("ISGlow_flagHigh", fh)]:
            m = gm | h
            cl = m_.individual.astype(str).values[m]
            for kind, A in [("flagged_level", DL), ("flagged_minus_kept", DD)]:
                est, p = strat_perm_multi(A[m], gm[m], m_.cohort.values[m], seed=31, clusters=cl)
                lo, hi = strat_boot(A[m], gm[m], m_.cohort.values[m], nboot=max(200, ARGS.nboot // 2), seed=32,
                                    clusters=cl)
                for k, sname in enumerate(setnames):
                    a, bb = A[gm, k], A[h, k]
                    rows.append(dict(group=gname, kind=kind, set=sname, focus=sname in focus,
                                     n_sle=int(np.isfinite(a).sum()), n_h=int(np.isfinite(bb).sum()),
                                     mean_sle=np.nanmean(a) if np.isfinite(a).any() else np.nan,
                                     mean_h=np.nanmean(bb) if np.isfinite(bb).any() else np.nan,
                                     diff=est[k], lo=lo[k], hi=hi[k], p_perm=p[k], p_mwu=mwu_p(a, bb)))
        T = pd.DataFrame(rows)
        T["q_focus"] = np.nan
        T["q_all"] = np.nan
        for (gname, kind), idx in T.groupby(["group", "kind"]).groups.items():
            T.loc[idx, "q_all"] = bh(T.loc[idx, "p_perm"].values)
            fi = [i for i in idx if T.loc[i, "focus"]]
            T.loc[fi, "q_focus"] = bh(T.loc[fi, "p_perm"].values)
        wcsv(T, f"step2_geneset_scores_{fc}.csv")
        # per-label focus-set comparison (flagged level), MWU + BH
        rows2 = []
        for (gname, gm), (b, t) in [(x, y) for x in [("ISGlow", isglow), ("ISGlow_flagHigh", fh)]
                                    for y in enumerate(E["labels"])]:
            for k, sname in enumerate(setnames):
                if sname not in focus:
                    continue
                ig, ih = m_.individual.astype(str).values[gm], m_.individual.astype(str).values[h]
                a = collapse_rows(LEV[gm, b, k][:, None], ig).ravel()
                bb = collapse_rows(LEV[h, b, k][:, None], ih).ravel()
                if np.isfinite(a).sum() < ARGS.min_donors or np.isfinite(bb).sum() < ARGS.min_donors:
                    continue
                a2 = collapse_rows(DEL[gm, b, k][:, None], ig).ravel()
                b2 = collapse_rows(DEL[h, b, k][:, None], ih).ravel()
                rows2.append(dict(group=gname, label=t, set=sname, n_sle=int(np.isfinite(a).sum()),
                                  n_h=int(np.isfinite(bb).sum()),
                                  level_sle=np.nanmean(a), level_h=np.nanmean(bb), p_level=mwu_p(a, bb),
                                  delta_sle=np.nanmean(a2) if np.isfinite(a2).any() else np.nan,
                                  delta_h=np.nanmean(b2) if np.isfinite(b2).any() else np.nan, p_delta=mwu_p(a2, b2)))
        T2 = pd.DataFrame(rows2)
        if len(T2):
            for gname in T2.group.unique():
                mm = T2.group == gname
                T2.loc[mm, "q_level"] = bh(T2.loc[mm, "p_level"].values)
                T2.loc[mm, "q_delta"] = bh(T2.loc[mm, "p_delta"].values)
            wcsv(T2.sort_values("p_level"), f"step2_geneset_by_label_{fc}.csv")
        out[fc] = (T, T2)
    T, T2 = out["ctrl"]
    R["step2_sets"] = T[(T.group == "ISGlow") & T.focus].sort_values(["kind", "p_perm"])[
        ["kind", "set", "n_sle", "n_h", "diff", "lo", "hi", "p_perm", "q_focus"]].to_dict(orient="records")
    if len(T2):
        R["step2_sets_by_label"] = T2[(T2.q_level < 0.1) | (T2.q_delta < 0.1)].sort_values("p_level").head(12)[
            ["group", "label", "set", "n_sle", "n_h", "level_sle", "level_h", "q_level", "delta_sle", "delta_h",
             "q_delta"]].to_dict(orient="records")
    try:
        F = T[(T.group == "ISGlow") & T.focus]
        fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
        order = F[F.kind == "flagged_level"].sort_values("diff").set.tolist()
        for ax, kind, lab in zip(axs, ["flagged_level", "flagged_minus_kept"],
                                 ["flagged cells vs healthy\n(z, ISG-low SLE - healthy)",
                                  "flagged - kept (paired)\n(z, ISG-low SLE - healthy)"]):
            G_ = F[F.kind == kind].set_index("set").reindex(order)
            yy = np.arange(len(G_))
            ax.errorbar(G_["diff"], yy, xerr=[G_["diff"] - G_["lo"], G_["hi"] - G_["diff"]], fmt="o",
                        c="#dd8452", ms=7, capsize=3)
            ax.axvline(0, c="k", lw=1)
            ax.set_yticks(yy)
            ax.set_yticklabels([f"{s}{' *' if q < 0.1 else ''}" for s, q in zip(G_.index, G_.q_focus.fillna(1))])
            ax.set_xlabel(lab)
        savefig(fig, "fig3_geneset_flagged.png")
    except Exception as e:  # noqa: BLE001
        log("fig3 failed", repr(e))
    return out


# ----------------------------------------------------------------------------------------------
# 3. clinical
# ----------------------------------------------------------------------------------------------
@section("3_clinical")
def step3(meta):
    # one unit per individual (largest) so repeated samples of a person are not counted as independent patients
    s = meta[meta.is_sle == 1].sort_values("n_cells", ascending=False).drop_duplicates("individual").copy()
    res = {"n_units_dropped_repeat_individual": int((meta.is_sle == 1).sum() - len(s))}
    for gcol in ["group", "group_strict_q90"]:
        if gcol not in s:
            continue
        A = s[s[gcol].isin(["ISGlow_flagHigh", "ISGhigh"])]
        r = dict(n=A[gcol].value_counts().to_dict())
        tabs = []
        for c in ["status", "sex", "ancestry", "cohort"]:
            if c not in A:
                continue
            x = collapse_levels(A[c], 1).values
            if c == "status" and "status_mixed" in A:
                x = np.where(A.status_mixed.astype(bool).values, "mixed", x)
            ct = pd.crosstab(x, A[gcol].values)
            tabs.append(ct.assign(variable=c).reset_index().rename(columns={"row_0": "level"}))
            lv = {}
            for L in ct.index:
                a = int(((x == L) & (A[gcol] == "ISGlow_flagHigh").values).sum())
                b = int(((x != L) & (A[gcol] == "ISGlow_flagHigh").values).sum())
                c_ = int(((x == L) & (A[gcol] == "ISGhigh").values).sum())
                d = int(((x != L) & (A[gcol] == "ISGhigh").values).sum())
                orr, pf = stats.fisher_exact([[a, b], [c_, d]])
                lv[str(L)] = dict(frac_isglow_flaghigh=a / max(a + b, 1), frac_isghigh=c_ / max(c_ + d, 1),
                                  OR=float(orr), p_fisher=float(pf))
            r[c] = dict(p_mc_chi2=perm_chi2(x, A[gcol].values), levels=lv)
        if tabs:
            wcsv(pd.concat(tabs), f"step3_clinical_tables_{gcol}.csv")
        # continuous: age, ISG, GC, IEG (all cells); KW over the 3 SLE groups + MWU two groups
        cont = {}
        for c in ["age_num", "isg", "ctrl", "allcells_glucocorticoid", "allcells_IEG", "allcells_glucocorticoid_core",
                  "allcells_dissociation_IEG", "n_cells"]:
            if c not in s or s[c].notna().sum() < 5:
                continue
            grps = [s.loc[s[gcol] == g, c].dropna().values for g in ["ISGlow_flagHigh", "ISGlow_flagLow", "ISGhigh"]]
            ok = [x for x in grps if len(x) >= 2]
            kw = stats.kruskal(*ok)[1] if len(ok) >= 2 else np.nan
            cont[c] = dict(median_isglow_flaghigh=float(np.median(grps[0])) if len(grps[0]) else np.nan,
                           median_isglow_flaglow=float(np.median(grps[1])) if len(grps[1]) else np.nan,
                           median_isghigh=float(np.median(grps[2])) if len(grps[2]) else np.nan,
                           p_kw=float(kw), p_mwu_fh_vs_isghigh=mwu_p(grps[0], grps[2]),
                           p_mwu_fh_vs_fl=mwu_p(grps[0], grps[1]))
        r["continuous"] = cont
        res[gcol] = r
    wcsv(meta.reset_index(), "donor_table.csv")
    R["step3"] = res
    return res


# ----------------------------------------------------------------------------------------------
# 4. stability
# ----------------------------------------------------------------------------------------------
@section("4_stability")
def step4(meta):
    qs = [float(x) for x in ARGS.qs.split(",")]
    q = qs[0]
    res = {}
    for fc in ["ctrl", "strict"]:
        main = meta[f"group_{fc}_q{int(round(q * 100))}"] if f"group_{fc}_q{int(round(q * 100))}" in meta else None
        if main is None:
            continue
        fh = main == "ISGlow_flagHigh"
        per = {}
        for s in R.get("seeds", []):
            c = f"{fc}_s{s}"
            if c not in meta:
                continue
            ti, tf = thresholds(meta, c, q)
            per[s] = assign_groups(meta, c, ti, tf)
        r = dict(n_main=int(fh.sum()))
        if per:
            P = pd.DataFrame(per)
            keep = (P.loc[fh] == "ISGlow_flagHigh").mean(1)
            sle = meta.is_sle == 1
            same_all = (P.loc[sle].eq(main[sle], axis=0)).mean(1)
            r.update(per_seed_n={str(k): int((v == "ISGlow_flagHigh").sum()) for k, v in per.items()},
                     mean_frac_seeds_keeping=float(keep.mean()) if len(keep) else np.nan,
                     n_in_all_seeds=int((keep == 1).sum()),
                     sle_group_agreement_all_groups=float(same_all.mean()))
            sets = [set(P.index[P[k] == "ISGlow_flagHigh"]) for k in P.columns]
            jac = [len(a & b) / len(a | b) if len(a | b) else np.nan for i, a in enumerate(sets) for b in sets[i + 1:]]
            r["pairwise_jaccard"] = jac
            meta[f"seedfrac_{fc}"] = (P == "ISGlow_flagHigh").mean(1)
        # healthy-threshold bootstrap
        rng = np.random.default_rng(41)
        hidx = np.where(healthy_ref_mask(meta).values)[0]
        cnt = np.zeros(len(meta))
        nb = max(200, ARGS.nboot // 2)
        for _ in range(nb):
            hb = meta.iloc[rng.choice(hidx, len(hidx))]
            ti, tf = float(np.nanquantile(hb.isg, q)), float(np.nanquantile(hb[fc], q))
            cnt += (assign_groups(meta, fc, ti, tf) == "ISGlow_flagHigh").values
        pb = pd.Series(cnt / nb, index=meta.index)
        meta[f"bootprob_{fc}"] = pb
        r["threshold_bootstrap"] = dict(mean_prob_main=float(pb[fh].mean()) if fh.any() else np.nan,
                                        n_prob_ge_0_8=int((pb >= 0.8).sum()),
                                        n_prob_ge_0_5=int((pb >= 0.5).sum()))
        res[fc] = r
        log(fc, r)
    keep_cols = [c for c in ["group", "group_strict_q90", "isg", "ctrl", "strict", "seedfrac_ctrl", "bootprob_ctrl",
                             "seedfrac_strict", "bootprob_strict", "cohort", "status"] if c in meta]
    wcsv(meta.loc[meta.is_sle == 1, keep_cols].reset_index(), "step4_group_stability.csv")
    R["step4"] = res
    return res


# ----------------------------------------------------------------------------------------------
# 5. leave one cohort out
# ----------------------------------------------------------------------------------------------
@section("5_loco")
def step5(meta):
    qs = [float(x) for x in ARGS.qs.split(",")]
    q = qs[0]
    rows = []
    for fc in ["ctrl", "strict"]:
        for c in ["none"] + sorted(meta.cohort.unique()):
            sub = meta if c == "none" else meta[meta.cohort != c]
            if (sub.is_sle == 0).sum() < 5 or (sub.is_sle == 1).sum() < 5:
                continue
            ti, tf = thresholds(sub, fc, q)
            grp = assign_groups(sub, fc, ti, tf)
            r = compare_isglow(sub, fc, ti, tf, q, seed=51)
            rows.append(dict(flag=fc, left_out=c, n_units=len(sub), thr_isg=ti, thr_flag=tf,
                             n_isglow_flaghigh=int((grp == "ISGlow_flagHigh").sum()),
                             n_isglow=int(grp.isin(["ISGlow_flagHigh", "ISGlow_flagLow"]).sum()),
                             diff=r["est"], lo=r["lo"], hi=r["hi"], p_perm=r["p"], n1=r["n1"], n0=r["n0"],
                             exceed_frac=r["exceed"].get("frac"), exceed_p=r["exceed"].get("p")))
        # single-cohort estimates
        for c in sorted(meta.cohort.unique()):
            sub = meta[meta.cohort == c]
            ti, tf = thresholds(meta, fc, q)
            a = sub.loc[(sub.is_sle == 1) & (sub.isg <= ti), fc].values
            b = sub.loc[sub.is_sle == 0, fc].values
            if len(a) >= 3 and len(b) >= 3:
                e = boot_diff(a, b, seed=53)
                rows.append(dict(flag=fc, left_out=f"only_{c}", n_units=len(sub), n1=len(a), n0=len(b),
                                 diff=e["est"], lo=e["lo"], hi=e["hi"], p_perm=mwu_p(a, b)))
    T = pd.DataFrame(rows)
    wcsv(T, "step5_loco.csv")
    R["step5"] = T.to_dict(orient="records")
    try:
        F = T[(T.flag == "ctrl")].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, max(4.5, 0.5 * len(F))))
        yy = np.arange(len(F))[::-1]
        ax.errorbar(100 * F["diff"], yy, xerr=[100 * (F["diff"] - F.lo), 100 * (F.hi - F["diff"])], fmt="o",
                    c="#dd8452", ms=7, capsize=3)
        ax.axvline(0, c="k", lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([("all cohorts" if x == "none" else (x.replace("only_", "only ") if x.startswith("only_")
                                                                else f"without {x}")) for x in F.left_out])
        ax.set_xlabel("flag fraction ISG-low SLE - healthy\n(% points, std.)")
        savefig(fig, "fig4_loco.png")
    except Exception as e:  # noqa: BLE001
        log("fig4 failed", repr(e))
    return T


# ----------------------------------------------------------------------------------------------
# RESULTS.md
# ----------------------------------------------------------------------------------------------
def write_md():
    L = []
    A = L.append
    ov = R.get("step1_overview", {})
    A("# ISG-low SLE patients with reference-unmatched cells\n")
    A(f"Units: {ov.get('n_sle')} SLE, {ov.get('n_healthy')} healthy-query ({ov.get('n_individuals')} individuals); "
      f"seeds {R.get('seeds')}; flag ctrl = p_unknown_ctrl<=0.05, strict = +p_unknown_bonf<=0.05 (seed majority). "
      f"ISG = type-I score z vs reference cells of same label, label-mix standardized; flag fraction standardized "
      f"to healthy label mix. Thresholds from healthy-query units of cohorts that contain SLE. Differences are "
      f"within Processing_Cohort (weighted); permutation and bootstrap resample individuals (donor_id), not "
      f"samples. Fractions in % points. ISGlow_flagHigh is selected on flag fraction: its flag-rate tests are "
      f"descriptive only.\n")
    s1 = R.get("step1", {})
    A("## 1. ISG-low SLE vs healthy-query: flag fraction")
    A("| flag | q | thr ISG / flag% | ISGhigh / ISGlow-flagHigh / ISGlow-flagLow | diff (pp) [CI] | p | "
      "frac > healthy q (exp.) p | vs ISG-matched healthy |")
    A("|---|---|---|---|---|---|---|---|")
    for k, v in s1.items():
        if not isinstance(v, dict) or "thr_isg" not in v:
            continue
        c, m, gg = v["isglow_vs_healthy"], v["vs_isg_matched_healthy"], v["groups"]
        ex = c.get("exceed", {})
        A(f"| {k.split('_')[0]} | {k.split('_q')[1]} | {fmt(v['thr_isg'])} / {fmt(100 * v['thr_flag'], 1)} | "
          f"{gg.get('ISGhigh', 0)} / {gg.get('ISGlow_flagHigh', 0)} / {gg.get('ISGlow_flagLow', 0)} | "
          f"{eci(c, mult=100, d=2)} (n {c.get('n1')}/{c.get('n0')}) | {fmt(c.get('p'), 4)} | "
          f"{fmt(ex.get('frac'))} ({fmt(ex.get('expected'))}) p {fmt(ex.get('p'), 4)} | "
          f"{eci(m, mult=100, d=2)} p {fmt(m.get('p'), 4)} |")
    o1 = s1.get("ctrl_one_unit_per_individual")
    if isinstance(o1, dict):
        A(f"\nOne unit per individual (ctrl, main q): diff {eci(o1, mult=100, d=2)} pp, p {fmt(o1.get('p'), 4)}.")
    oa = s1.get("ctrl_cohort_x_ancestry")
    if isinstance(oa, dict):
        A(f"Strata cohort x ancestry (ctrl, main q): diff {eci(oa, mult=100, d=2)} pp, p {fmt(oa.get('p'), 4)}, "
          f"n {oa.get('n1')}/{oa.get('n0')}.")
    ol = s1.get("ols_isglow_adj_isg_cohort")
    if ol:
        A(f"OLS (ISG-low SLE + healthy; adj. {'/'.join(s1.get('ols_covariates', []))}; cluster-robust by "
          f"individual): ISG-low SLE coef {fmt(100 * ol['coef'], 2)} pp "
          f"[{fmt(100 * ol['lo'], 2)}, {fmt(100 * ol['hi'], 2)}], p {fmt(ol['p'], 4)}.")
    A("\n## 2. Where the excess sits (ISG-low SLE, ctrl flag)")
    lr = R.get("step2_label_rates", {}).get("ctrl", [])
    if lr:
        A("| label | rate SLE / H | diff within cohort [CI] | q | share of std. excess |")
        A("|---|---|---|---|---|")
        for r in lr[:7]:
            A(f"| {r['label']} | {fmt(r['mean_sle'])} / {fmt(r['mean_h'])} | "
              f"{fmt(r['diff_within_cohort'], 3)} [{fmt(r['lo'], 3)}, {fmt(r['hi'], 3)}] | {fmt(r['q'], 3)} | "
              f"{fmt(r['share_of_excess'])} |")
    for gname, lf in R.get("step2_lfc", {}).items():
        A(f"\nFlagged-vs-kept pseudobulk log2FC, {gname} donors (n SLE/H; genes up q<.1; more than healthy q<.1; "
          f"top genes):")
        for r in lf[:6]:
            A(f"- {r['label']} ({r['n_sle']}/{r['n_h']}): {r['n_up_q10']}; {r['n_specific_q10']}: "
              f"{r['top_specific_q10'] or r['top_up'] or '-'}")
    st = R.get("step2_sets", [])
    if st:
        A("\nFocus gene-set scores of flagged cells, ISG-low SLE minus healthy (z units, within cohort):")
        for kind in ["flagged_level", "flagged_minus_kept"]:
            xs = [r for r in st if r["kind"] == kind][:6]
            A(f"- {kind}: " + "; ".join(f"{r['set']} {fmt(r['diff'])} [{fmt(r['lo'])}, {fmt(r['hi'])}] "
                                        f"q {fmt(r['q_focus'], 3)}" for r in xs))
    sl = R.get("step2_sets_by_label", [])
    if sl:
        A("- per label (q<0.1): " + "; ".join(
            f"{r['group']} {r['label']}:{r['set']} lvl {fmt(r['level_sle'])} vs {fmt(r['level_h'])} (q {fmt(r['q_level'], 3)}),"
            f" delta q {fmt(r['q_delta'], 3)}" for r in sl[:6]))
    A("\n## 3. Clinical: ISGlow-flagHigh vs ISGhigh (ctrl, main q)")
    c3 = R.get("step3", {}).get("group", {})
    if c3:
        A(f"n {c3.get('n')}")
        for c in ["status", "sex", "ancestry", "cohort"]:
            if c in c3:
                lv = c3[c]["levels"]
                A(f"- {c} (MC chi2 p {fmt(c3[c]['p_mc_chi2'], 3)}): " + "; ".join(
                    f"{k} {fmt(v['frac_isglow_flaghigh'])} vs {fmt(v['frac_isghigh'])} (Fisher p {fmt(v['p_fisher'], 3)})"
                    for k, v in lv.items())[:600])
        for c, v in c3.get("continuous", {}).items():
            A(f"- {c}: median FH {fmt(v['median_isglow_flaghigh'])} / FL {fmt(v['median_isglow_flaglow'])} / "
              f"ISGhigh {fmt(v['median_isghigh'])}; KW p {fmt(v['p_kw'], 3)}, FH vs ISGhigh p "
              f"{fmt(v['p_mwu_fh_vs_isghigh'], 3)}, FH vs FL p {fmt(v['p_mwu_fh_vs_fl'], 3)}")
    A("\n## 4. Stability of ISGlow-flagHigh membership")
    for fc, r in R.get("step4", {}).items():
        tb = r.get("threshold_bootstrap", {})
        A(f"- {fc}: n {r.get('n_main')}; per seed {r.get('per_seed_n')}; mean fraction of seeds keeping "
          f"{fmt(r.get('mean_frac_seeds_keeping'))}; in all seeds {r.get('n_in_all_seeds')}; Jaccard "
          f"{[round(x, 2) for x in r.get('pairwise_jaccard', [])]}; threshold-bootstrap mean prob "
          f"{fmt(tb.get('mean_prob_main'))}, n prob>=0.8 {tb.get('n_prob_ge_0_8')}")
    A("\n## 5. Leave-one-cohort-out (ctrl)")
    lo = [r for r in R.get("step5", []) if r.get("flag") == "ctrl"]
    for r in lo:
        A(f"- {r['left_out']}: diff {fmt(100 * r['diff'], 2)} [{fmt(100 * r['lo'], 2)}, {fmt(100 * r['hi'], 2)}] pp, "
          f"p {fmt(r['p_perm'], 4)}, n {r.get('n1')}/{r.get('n0')}"
          + (f", ISGlow-flagHigh {int(r['n_isglow_flaghigh'])}" if r.get('n_isglow_flaghigh') is not None
             and np.isfinite(r['n_isglow_flaghigh']) else ""))
    A("\n## Verdict")
    A(verdict())
    if R["errors"]:
        A(f"\nErrors: {list(R['errors'])}")
    txt = "\n".join(L)
    if len(txt.encode()) > 8000:
        txt = txt[:7900] + "\n...(truncated; see results.json)"
    with open(os.path.join(OUT, "RESULTS.md"), "w") as f:
        f.write(txt + "\n")


def verdict():
    s1 = R.get("step1", {})
    keys = [k for k in s1 if k.startswith("ctrl_q")]
    if not keys:
        return "Step 1 failed; no verdict."
    main = s1[keys[0]]
    c = main["isglow_vs_healthy"]
    m = main["vs_isg_matched_healthy"]
    ex = c.get("exceed", {})
    lo = [r for r in R.get("step5", []) if r.get("flag") == "ctrl" and r["left_out"] not in ("none",)
          and not str(r["left_out"]).startswith("only_")]
    loco_ok = bool(lo) and all(np.isfinite(r["lo"]) and r["lo"] > 0 for r in lo)
    strict = s1.get(keys[0].replace("ctrl", "strict"), {}).get("isglow_vs_healthy", {})
    excess = (c.get("lo", np.nan) > 0) and (c.get("p", 1) < 0.05) and (ex.get("p", 1) < 0.05)
    parts = [f"ISG-low SLE excess {eci(c, mult=100, d=2)} pp (p {fmt(c.get('p'), 4)}); "
             f"{fmt(100 * ex['frac'] if ex.get('frac') is not None else np.nan, 0)}% of ISG-low SLE above the healthy q90 (expected 10%, "
             f"p {fmt(ex.get('p'), 4)}); vs ISG-matched healthy {eci(m, mult=100, d=2)} pp; strict flag "
             f"{eci(strict, mult=100, d=2)} pp; LOCO all CI>0: {loco_ok}."]
    if excess and loco_ok:
        parts.append("=> YES: ISG-low patients carry more unmatched cells than healthy variation, robust to cohort.")
    elif excess:
        parts.append("=> PARTLY: excess present overall but not robust in every leave-one-cohort-out fit.")
    else:
        parts.append("=> NO convincing excess beyond healthy variation in ISG-low patients.")
    lr = [r for r in R.get("step2_label_rates", {}).get("ctrl", []) if r.get("q") is not None and r["q"] < 0.1]
    if lr:
        parts.append("Labels carrying it (q<0.1): " + ", ".join(
            f"{r['label']} ({fmt(100 * r['share_of_excess'], 0)}% of excess)" for r in lr[:5]) + ".")
    st = [r for r in R.get("step2_sets", []) if r.get("q_focus") is not None and r["q_focus"] < 0.1]
    if st:
        parts.append("States in flagged cells (q<0.1): " + ", ".join(
            f"{r['set']} ({r['kind']}, {fmt(r['diff'])})" for r in st[:6]) + ".")
    s4 = R.get("step4", {}).get("ctrl", {})
    if s4:
        parts.append(f"Group membership kept across seeds: {fmt(s4.get('mean_frac_seeds_keeping'))}; "
                     f"threshold-bootstrap prob {fmt(s4.get('threshold_bootstrap', {}).get('mean_prob_main'))}.")
    return " ".join(parts)


def main(argv=None):
    global ARGS, OUT, LOGF
    ARGS = parse_args(argv)
    OUT = ARGS.out if os.path.isabs(ARGS.out) else os.path.join(ARGS.root, ARGS.out)
    os.makedirs(OUT, exist_ok=True)
    LOGF = open(os.path.join(OUT, "log.txt"), "w")
    R["args"] = vars(ARGS)
    gs, focus, allsets = load_genesets()
    isg_genes = focus.get("ISG_typeI") or gs.get(ARGS.isg_key)
    D = load_flags()
    if D is None:
        raise RuntimeError("could not load flags")
    E = stream(D, isg_genes)
    if E is None:
        raise RuntimeError("could not stream expression")
    o = donor_table(D, E)
    if o is None:
        raise RuntimeError("donor table failed")
    meta, n, tabs, w = o
    step1(meta)
    if "group" in meta:
        step2_rates(meta, n, tabs, w)
        step2_lfc(meta, E)
        step2_sets(meta, E, allsets, focus, w)
        step3(meta)
        step4(meta)
        step5(meta)
    try:
        write_md()
    except Exception as e:  # noqa: BLE001
        R["errors"]["results_md"] = repr(e)
        log(traceback.format_exc())
    save_results()
    log(f"done in {time.time() - T0:.0f}s; errors: {list(R['errors'])}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        R["errors"]["main"] = repr(e)
        log(traceback.format_exc())
        if OUT:
            save_results()
        sys.exit(1)
