#!/usr/bin/env python
"""Compare ARCANE unknown flags of the full-gene model (tag lupusfull) with the ISG/IEG-blind model
(tag lupusfull_isgblind300: scVI/scANVI retrained without curated IFN/IEG/heat-shock genes and the 300 genes most
correlated with an IFN score within labels) on the full Perez SLE cohort.

Question: does ARCANE detect SLE deviation that an ISG score alone could not produce?

Sections (each wrapped in try/except; failures are logged in isgblind.json["errors"]):
  Q1  donor-level flag-rate excess SLE vs healthy-query donors, composition-standardized, donor-bootstrap CIs,
      for flag definitions ctrl (p_unknown_ctrl<=alpha), bonf (unknown_bonf / p_unknown_bonf<=alpha), pool (unknown),
      for both tags and an ISG-only detector calibrated like p_unknown_ctrl (ISG z > healthy-query q95 within type).
      Fraction of the SLE excess retained after blinding (paired bootstrap). Robustness: ISG-low SLE donors,
      cohort and cohort x ancestry stratified excess, OLS (HC3) with cohort/ancestry/sex/age (+ ISG, GC, IEG scores).
  Q2  within-SLE Spearman of donor flag rate vs donor ISG score (expression, z-scored per fine type against healthy
      query cells), both tags, bootstrap CIs, paired difference; also vs GC (treatment proxy) and IEG (dissociation).
  Q3  fine types that keep an excess after blinding (per-type donor rates, bootstrap CI, MWU, BH).
  Q4  SLE cells flagged by the blind model vs kept cells of the same type: donor-paired pseudobulk DE (dropped genes
      shown but excluded from interpretation and enrichment), hypergeometric gene-set enrichment, and per-donor
      gene-set score deltas (flagged - kept) with CIs, in SLE and (control) healthy-query donors.
  Q5  overlap of full vs blind flags (Jaccard, odds ratio, per seed and seed-stable), seed-to-seed reference,
      overlap with the ISG-only detector, profile and donor excess of blind-only cells.
  Q6  within SLE: blind (and full) flag rate ~ disease_state + cohort + ancestry + age + sex (OLS HC3, plus cluster-
      robust by donor when donors contribute several states; plus a model adding ISG/GC/IEG scores) and a
      within-donor paired flare vs other-state contrast.
  Novel-finding gates N1-N7 evaluated automatically and written into RESULTS.md.

Run from the project directory (~/arcane):
  python isgblind_compare.py [--root .] [--tag_full lupusfull] [--tag_blind lupusfull_isgblind300]
Inputs:  <root>/data/std/<tag_full>.h5ad (raw counts; falls back to <tag_full>.expr.npz stand-in used for testing),
         <root>/<res_dir>/<tag>_seed<s>_cells.npz (res_dir default res2, falls back to res),
         <root>/emb/<tag_blind>_seed0_full_dropped.txt, <root>/genesets.json
Outputs: <root>/summary/isgblind/ : RESULTS.md, isgblind.json, *.csv, *.png, log.txt
Only numpy, pandas, scipy, matplotlib (+ anndata for .h5ad) are required; OLS/HC3 is implemented here.
"""
import argparse
import glob
import json
import os
import re
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

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--tag_full", default="lupusfull")
ap.add_argument("--tag_blind", default="lupusfull_isgblind300")
ap.add_argument("--res_dir", default="res2")
ap.add_argument("--h5ad", default=None, help="default <root>/data/std/<tag_full>.h5ad")
ap.add_argument("--dropped", default=None, help="default <root>/emb/<tag_blind>_seed0_full_dropped.txt")
ap.add_argument("--genesets", default="genesets.json")
ap.add_argument("--out", default="summary/isgblind")
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--primary", default="ctrl", choices=["ctrl", "bonf", "pool"])
ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--min_cells_type", type=int, default=10, help="min cells of a type in a donor for its rate to count")
ap.add_argument("--min_cells_pb", type=int, default=10, help="min flagged and kept cells per donor for pseudobulk")
ap.add_argument("--min_donors_pb", type=int, default=6)
ap.add_argument("--max_pb_types", type=int, default=12)
ap.add_argument("--max_per_group", type=int, default=300, help="cells per donor x group in pseudobulk")
ap.add_argument("--max_expr_per_donor", type=int, default=0, help="subsample cells per donor for expression (0=all)")
ap.add_argument("--chunk", type=int, default=50000)
ap.add_argument("--isg_key", default=None)
ap.add_argument("--gc_key", default=None)
ap.add_argument("--diss_key", default=None)
ap.add_argument("--isg_low_q", type=float, default=0.9,
                help="ISG-low SLE donors: donor ISG <= this quantile of healthy-query donor ISG")
args = ap.parse_args()

ROOT = args.root
OUT = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
os.makedirs(OUT, exist_ok=True)
for _f in os.listdir(OUT):  # remove this script's own stale outputs (earlier versions left orphan CSVs)
    if re.match(r"^(Q\d_.*\.(csv|png)|RESULTS\.md|donor_table\.csv|isgblind\.json)$", _f):
        os.remove(os.path.join(OUT, _f))
LOGF = open(os.path.join(OUT, "log.txt"), "w")
T0 = time.time()
R = {"args": vars(args), "errors": {}, "timing_s": {}}
TAGS = {"full": args.tag_full, "blind": args.tag_blind}
DEFS = ["ctrl", "bonf", "pool"]
# gene sets that are IFN / IEG / technical: never count as "new biology"
EXCLUDE_BIO = r"ISG|IFN|interferon|SIGLEC1|dissoc|(?<!no)IEG|heat|mitochond|ribosom|platelet|erythro|cell_cycle|hypoxia"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n")
    LOGF.flush()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, pd.Series):
        return clean(o.to_dict())
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict(orient="records"))
    return o


def save_results():
    with open(os.path.join(OUT, "isgblind.json"), "w") as f:
        json.dump(clean(R), f, indent=1, default=str)


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


plt.rcParams.update({"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
                     "legend.fontsize": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200})
COL = {"healthy": "#4A7FB0", "SLE": "#C8553D", "full": "#555555", "blind": "#C8553D", "isgdet": "#8A6FB0"}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def wcsv(df, name, index=False):
    df.to_csv(os.path.join(OUT, name), index=index)


def safe_name(s):
    return re.sub(r"[^0-9A-Za-z_]", "_", str(s))


# ----------------------------------------------------------------------------------------------
# statistics
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


def ci(bs):
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    if len(bs) < 20:
        return np.nan, np.nan
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def mwu_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def spearman_boot(x, y, nboot=None, seed=1):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(n=int(n), rho=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot or args.nboot):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        bs.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = ci(bs)
    return dict(n=int(n), rho=float(rho), p=float(p), ci_lo=lo, ci_hi=hi)


def rank_rho(x, y):
    """fast Spearman for bootstrap (no ties handling beyond average ranks)"""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_mean_ci(v, nboot=None, seed=2):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return dict(n=int(len(v)), mean=float(np.mean(v)) if len(v) else np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    p_wilcoxon=np.nan, frac_pos=np.nan)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (nboot or args.nboot, len(v)))].mean(1)
    lo, hi = ci(bs)
    try:
        pw = float(stats.wilcoxon(v).pvalue)
    except Exception:  # noqa: BLE001
        pw = np.nan
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), ci_lo=lo, ci_hi=hi,
                p_wilcoxon=pw, frac_pos=float((v > 0).mean()))


def boot_idx(n_s, n_h, nboot, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_s, (nboot, n_s)), rng.integers(0, n_h, (nboot, n_h))


def excess_boot(vs, vh, nboot=None, seed=7):
    """mean(SLE) - mean(healthy) with stratified donor bootstrap. vs, vh: 1D arrays (NaN dropped)."""
    vs, vh = np.asarray(vs, float), np.asarray(vh, float)
    vs, vh = vs[np.isfinite(vs)], vh[np.isfinite(vh)]
    if len(vs) < 3 or len(vh) < 3:
        return dict(n_sle=int(len(vs)), n_healthy=int(len(vh)), excess=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    mean_sle=np.nan, mean_healthy=np.nan, ratio=np.nan, p_mwu=np.nan)
    iS, iH = boot_idx(len(vs), len(vh), nboot or args.nboot, seed)
    bs = vs[iS].mean(1) - vh[iH].mean(1)
    lo, hi = ci(bs)
    return dict(n_sle=int(len(vs)), n_healthy=int(len(vh)), mean_sle=float(vs.mean()), mean_healthy=float(vh.mean()),
                excess=float(vs.mean() - vh.mean()), ci_lo=lo, ci_hi=hi,
                ratio=float(vs.mean() / vh.mean()) if vh.mean() > 0 else np.nan, p_mwu=mwu_p(vs, vh))


def stratified_excess(df, col, strata, nboot=None, min_n=3, seed=9):
    """Within-stratum SLE - healthy mean difference, combined with weights n_s*n_h/(n_s+n_h); bootstrap within
    stratum x group. Only strata with >= min_n donors in both groups are used."""
    d = df[[col, "is_sle"] + strata].dropna(subset=[col]).copy()
    d["_st"] = d[strata].astype(str).agg("|".join, axis=1)
    rows, parts = [], []
    for st, g in d.groupby("_st"):
        s, h = g[g.is_sle == 1][col].values, g[g.is_sle == 0][col].values
        rows.append(dict(stratum=st, n_sle=len(s), n_healthy=len(h),
                         excess=(s.mean() - h.mean()) if len(s) and len(h) else np.nan))
        if len(s) >= min_n and len(h) >= min_n:
            parts.append((s, h, len(s) * len(h) / (len(s) + len(h))))
    T = pd.DataFrame(rows)
    if not parts:
        return dict(n_strata_used=0, excess=np.nan, ci_lo=np.nan, ci_hi=np.nan), T
    W = np.array([p[2] for p in parts])
    W = W / W.sum()
    est = float(sum(w * (s.mean() - h.mean()) for (s, h, _), w in zip(parts, W)))
    rng = np.random.default_rng(seed)
    B = nboot or args.nboot
    bs = np.zeros(B)
    for (s, h, _), w in zip(parts, W):
        bs += w * (s[rng.integers(0, len(s), (B, len(s)))].mean(1) - h[rng.integers(0, len(h), (B, len(h)))].mean(1))
    lo, hi = ci(bs)
    return dict(n_strata_used=len(parts), n_sle_used=int(sum(len(p[0]) for p in parts)),
                n_healthy_used=int(sum(len(p[1]) for p in parts)), excess=est, ci_lo=lo, ci_hi=hi), T


def collapse_levels(s, min_n=3, na="unknown"):
    s = s.astype(object).where(s.notna(), na).astype(str)
    s = s.where(~s.str.lower().isin(["nan", "none", "", "na", "unknown"]), na)
    vc = s.value_counts()
    rare = vc[vc < min_n].index
    return s.where(~s.isin(rare), "other")


def zs(v):
    v = np.asarray(v, float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / (sd if sd > 0 else 1.0)


def ols_robust(df, y, cats=(), nums=(), refs=None, cluster=None, min_level=3):
    """OLS with HC3 (or CR1 cluster-robust when cluster is given). Returns (coef table, info)."""
    refs = refs or {}
    cols = [y] + list(cats) + list(nums) + ([cluster] if cluster else [])
    d = df[[c for c in dict.fromkeys(cols) if c in df]].copy()
    d = d.dropna(subset=[y] + [c for c in nums if c in d])
    Xc = [pd.Series(1.0, index=d.index, name="Intercept")]
    used = []
    for c in cats:
        if c not in d:
            continue
        v = collapse_levels(d[c], min_level)
        lev = list(v.value_counts().index)
        if len(lev) < 2:
            continue
        ref = refs.get(c, lev[0])
        if ref not in lev:
            ref = lev[0]
        for L in sorted(lev):
            if L == ref:
                continue
            Xc.append(pd.Series((v == L).astype(float).values, index=d.index, name=f"{c}[T.{L}]"))
        used.append(f"{c}(ref={ref})")
    for c in nums:
        if c in d and np.nanstd(d[c].astype(float)) > 0:
            Xc.append(pd.Series(zs(d[c].astype(float).values) if c != "is_sle" else d[c].astype(float).values,
                                index=d.index, name=c))
            used.append(c)
    X = pd.concat(Xc, axis=1)
    # drop linearly dependent columns (keeps the first-listed column)
    keep = []
    for j in range(X.shape[1]):
        trial = keep + [j]
        if np.linalg.matrix_rank(X.values[:, trial]) == len(trial):
            keep.append(j)
    dropped = [X.columns[j] for j in range(X.shape[1]) if j not in keep]
    X = X.iloc[:, keep]
    Xv, yv = X.values, d[y].astype(float).values
    n, p = Xv.shape
    if n <= p + 2:
        raise ValueError(f"too few rows ({n}) for {p} parameters")
    XtXi = np.linalg.pinv(Xv.T @ Xv)
    b = XtXi @ Xv.T @ yv
    e = yv - Xv @ b
    if cluster and cluster in d and d[cluster].nunique() < n:
        g = d[cluster].astype(str).values
        G = len(np.unique(g))
        meat = np.zeros((p, p))
        for gg in np.unique(g):
            m = g == gg
            u = Xv[m].T @ e[m]
            meat += np.outer(u, u)
        meat *= G / (G - 1) * (n - 1) / (n - p)
        dof, kind = G - 1, f"CR1 cluster({cluster}, G={G})"
    else:
        h = np.einsum("ij,jk,ik->i", Xv, XtXi, Xv)
        w = e ** 2 / np.clip(1 - h, 1e-6, None) ** 2
        meat = (Xv * w[:, None]).T @ Xv
        dof, kind = n - p, "HC3"
    V = XtXi @ meat @ XtXi
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    t = b / np.where(se > 0, se, np.nan)
    pv = 2 * stats.t.sf(np.abs(t), dof)
    tq = stats.t.ppf(0.975, dof)
    T = pd.DataFrame(dict(term=X.columns, coef=b, se=se, p=pv, ci_lo=b - tq * se, ci_hi=b + tq * se))
    r2 = 1 - (e ** 2).sum() / max(((yv - yv.mean()) ** 2).sum(), 1e-12)
    return T, dict(n=int(n), k=int(p), se_type=kind, r2=float(r2), terms=used, collinear_dropped=dropped)


# ----------------------------------------------------------------------------------------------
# loading
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


def is_sle_str(s):
    return bool(re.search(r"lupus|\bsle\b", str(s), re.I))


def load_tag(tag):
    rd = os.path.join(ROOT, args.res_dir)
    files = sorted(glob.glob(os.path.join(rd, f"{tag}_seed*_cells.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(ROOT, "res", f"{tag}_seed*_cells.npz")))
    if not files:
        raise FileNotFoundError(f"no cells npz for {tag} in {rd} or res/")
    base, flags, seeds = None, {}, []
    for f in files:
        m = re.search(r"_seed(\d+)_cells\.npz$", f)
        if not m:
            continue
        s = int(m.group(1))
        C = np.load(f, allow_pickle=True)
        names = np.asarray(C["names"]).astype(str)
        if len(np.unique(names)) != len(names):
            raise ValueError(f"{f}: duplicated cell names; joins across seeds/tags/h5ad would be ambiguous")
        classes = np.asarray(C["classes"]).astype(str) if "classes" in C.files else None

        def lab(v):
            v = np.asarray(v)
            if v.dtype.kind in "iu" and classes is not None:
                return np.where(v >= 0, classes[np.clip(v, 0, len(classes) - 1)], "none")
            return v.astype(str)
        df = pd.DataFrame(index=pd.Index(names, name="cell"))
        if base is None:
            df["y"] = lab(C["y"])
            df["cond_raw"] = np.asarray(C["cond"]).astype(str)
            df["donor"] = np.asarray(C["batch"]).astype(str)
        df[f"novelty_s{s}"] = np.asarray(C["novelty"], float) if "novelty" in C.files else np.nan
        a = args.alpha
        fl = {}
        if "p_unknown_ctrl" in C.files:
            fl["ctrl"] = np.asarray(C["p_unknown_ctrl"], float) <= a
        if "unknown_bonf" in C.files:
            fl["bonf"] = np.asarray(C["unknown_bonf"]).astype(bool)
        elif "p_unknown_bonf" in C.files:
            fl["bonf"] = np.asarray(C["p_unknown_bonf"], float) <= a
        if "unknown" in C.files:
            fl["pool"] = np.asarray(C["unknown"]).astype(bool)
        elif "p_unknown" in C.files:
            fl["pool"] = np.asarray(C["p_unknown"], float) <= a
        for k, v in fl.items():
            df[f"f_{k}_s{s}"] = v.astype(np.float32)
        log(f"{tag} seed {s}: {len(df)} cells; " + ", ".join(f"{k}={v.mean():.4f}" for k, v in fl.items()))
        if base is None:
            base = df
        else:
            n0 = len(base)
            base = base.join(df, how="inner")
            if len(base) < n0:
                log(f"WARNING: {tag} seed {s} shares only {len(base)}/{n0} cells with earlier seeds")
        seeds.append(s)
    return base, sorted(seeds)


def load_genesets():
    cands = [args.genesets if os.path.isabs(args.genesets) else os.path.join(ROOT, args.genesets),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "genesets.json")]
    for p in cands:
        if os.path.exists(p):
            gs = json.load(open(p))
            gs = {str(k): [str(g) for g in v] for k, v in gs.items() if isinstance(v, (list, tuple))}
            log("gene sets:", p, len(gs))
            return gs
    raise FileNotFoundError(f"genesets.json not found in {cands}")


def detect_key(gs, user, pats):
    if user:
        return user if user in gs else None
    for p in pats:
        for k in gs:
            if re.search(p, k, re.I):
                return k
    return None


def load_dropped():
    p = args.dropped or os.path.join(ROOT, "emb", f"{args.tag_blind}_seed0_full_dropped.txt")
    if not os.path.exists(p):
        alt = sorted(glob.glob(os.path.join(ROOT, "emb", f"{args.tag_blind}_seed*_dropped.txt")))
        if not alt:
            log("WARNING: dropped-gene list not found:", p)
            return set()
        p = alt[0]
    toks = []
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks += [t for t in re.split(r"[,\s\t]+", line) if t]
    dropped = set(toks)
    log(f"dropped genes ({p}): {len(dropped)}")
    return dropped


def _normalize_rows(X):
    X = sp.csr_matrix(X, dtype=np.float32)
    X.eliminate_zeros()
    samp = X.data[:200000]
    if len(samp) and not np.allclose(samp, np.round(samp)):
        log("WARNING: X chunk is not integer counts; assuming already log-normalized")
        return X, np.asarray(np.expm1(X).sum(1)).ravel(), False
    tot = np.asarray(X.sum(1)).ravel()
    X.data *= np.repeat((1e4 / np.maximum(tot, 1)).astype(np.float32), np.diff(X.indptr))
    np.log1p(X.data, out=X.data)
    return X, tot, True


def _symbols(var_names, var_df):
    var = pd.Index(np.asarray(var_names).astype(str))
    if np.mean(var.str.startswith("ENS")) > 0.5 and var_df is not None:
        for c in ["feature_name", "gene_symbols", "gene_symbol", "symbol", "gene_name"]:
            if c in var_df.columns:
                log(f"var_names look like Ensembl IDs; using var['{c}'] as gene symbols")
                return pd.Index(var_df[c].astype(str).values)
    return var


def load_expression(cells, all_cells):
    """Returns log1p-CP10k CSR matrix for `cells` (in that order), var_names, obs (strings) for `all_cells`
    (all analysed cells, so covariates/disease_state are available even when expression is subsampled), totals.
    Backed h5ad is read in contiguous row windows (bounded memory for sparse and dense X)."""
    h5 = args.h5ad or os.path.join(ROOT, "data", "std", f"{args.tag_full}.h5ad")
    stand = re.sub(r"\.h5ad$", "", h5) + ".expr.npz"
    cells = np.asarray(cells).astype(str)
    all_cells = np.asarray(all_cells).astype(str)
    if os.path.exists(h5):
        import anndata as ad
        try:
            A = ad.read_h5ad(h5, backed="r")
            backed = True
        except Exception as e:  # noqa: BLE001
            log(f"backed read failed ({e!r}); loading fully")
            A = ad.read_h5ad(h5)
            backed = False
        names = pd.Index(np.asarray(A.obs_names).astype(str))
        if not names.is_unique:
            raise ValueError("h5ad obs_names are not unique; cannot match npz names")
        pos_all = names.get_indexer(all_cells)
        if (pos_all < 0).any():
            raise KeyError(f"{(pos_all < 0).sum()} cells not in h5ad, e.g. {all_cells[pos_all < 0][:3]}")
        obs = A.obs.iloc[pos_all].copy()  # subset before converting to str (1M+ rows x many columns)
        for c in obs.columns:
            obs[c] = obs[c].astype(str).values
        var = _symbols(A.var_names, A.var)
        pos = names.get_indexer(cells)
        if not backed and sp.issparse(A.X) and not sp.isspmatrix_csr(A.X):
            A.X = sp.csr_matrix(A.X)
        try:
            is_sparse = sp.issparse(A.X[0:2])
        except Exception:  # noqa: BLE001
            is_sparse = True
        # rows per window: sparse -> args.chunk; dense -> ~1 GB float32
        win = args.chunk if is_sparse else max(1000, int(2.5e8 / max(A.n_vars, 1)))
        order = np.argsort(pos, kind="stable")
        sp_pos = pos[order]
        wid = sp_pos // win
        bounds = np.flatnonzero(np.diff(wid)) + 1
        chunks, tots, done = [], [], 0
        for rows in np.split(sp_pos, bounds):
            if not len(rows):
                continue
            lo, hi = int(rows[0]), int(rows[-1]) + 1
            try:
                Xi = A.X[lo:hi][rows - lo] if backed else A.X[rows]
            except Exception as e:  # noqa: BLE001
                if backed:
                    log(f"backed row read failed ({e!r}); loading h5ad into memory")
                    A = ad.read_h5ad(h5)
                    backed = False
                    if sp.issparse(A.X) and not sp.isspmatrix_csr(A.X):
                        A.X = sp.csr_matrix(A.X)
                Xi = A.X[rows]
            Xi, tot, _ = _normalize_rows(Xi)
            chunks.append(Xi)
            tots.append(tot)
            done += len(rows)
            log(f"  expression rows {done}/{len(sp_pos)}")
        X = sp.vstack(chunks).tocsr()
        del chunks
        tot = np.concatenate(tots)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        X, tot = X[inv], tot[inv]
        del A
    elif os.path.exists(stand):
        log("h5ad not found; using numpy stand-in", stand)
        Z = np.load(stand, allow_pickle=True)
        Xa = sp.csr_matrix((Z["data"], Z["indices"], Z["indptr"]), shape=tuple(Z["shape"]))
        names = pd.Index(np.asarray(Z["obs_names"]).astype(str))
        var = pd.Index(np.asarray(Z["var_names"]).astype(str))
        obs = pd.read_csv(stand.replace(".expr.npz", ".obs.csv"), index_col=0, dtype=str, keep_default_na=False)
        pos = names.get_indexer(cells)
        pos_all = names.get_indexer(all_cells)
        if (pos < 0).any() or (pos_all < 0).any():
            raise KeyError("cells missing from stand-in")
        X, tot, _ = _normalize_rows(Xa[pos])
        obs = obs.loc[names[pos_all]]
    else:
        raise FileNotFoundError(f"neither {h5} nor {stand}")
    obs.index = pd.Index(all_cells)
    return X, var, obs, tot


def score_sets(X, var, sets, rng, n_bins=25, ctrl_size=50, min_det=20):
    """scanpy score_genes algorithm (expression-binned control genes) for many sets with one sparse product."""
    det = np.bincount(X.indices, minlength=X.shape[1])
    ok = det >= min_det
    gm = np.asarray(X.mean(0)).ravel()
    gm_ok = pd.Series(gm[ok], index=np.where(ok)[0])
    n_items = int(np.round(len(gm_ok) / (n_bins - 1)))
    cut = pd.Series(-1, index=np.arange(X.shape[1]))
    cut[gm_ok.index] = (gm_ok.rank(method="min") // max(n_items, 1)).astype(int).values
    cut = cut.values
    vpos = pd.Series(np.arange(len(var)), index=var)
    vpos = vpos[~vpos.index.duplicated()]
    cols, names, used = [], [], {}
    for name, genes in sets.items():
        gi = np.array(sorted(set(vpos.reindex([g for g in genes if g in vpos.index]).values.tolist())), int)
        gi = gi[ok[gi]] if len(gi) else gi
        used[name] = [str(var[i]) for i in gi]
        if len(gi) < 3:
            continue
        ctrl = set()
        for b in np.unique(cut[gi]):
            pool = np.where(cut == b)[0]
            pool = pool[~np.isin(pool, gi)]
            rng.shuffle(pool)
            ctrl.update(pool[:ctrl_size].tolist())
        ctrl = np.array(sorted(ctrl), int)
        w = np.zeros(X.shape[1], np.float32)
        w[gi] = 1.0 / len(gi)
        if len(ctrl):
            w[ctrl] -= 1.0 / len(ctrl)
        cols.append(w)
        names.append(name)
    if not cols:
        return pd.DataFrame(index=np.arange(X.shape[0])), used
    W = sp.csr_matrix(np.column_stack(cols))
    S = np.asarray((X @ W).todense(), dtype=np.float32)
    return pd.DataFrame(S, columns=names), used


def z_by_type(v, y, healthy, min_n=20):
    """z-score of v within fine type against healthy-query cells of that type (pooled healthy if too few)."""
    v = pd.Series(np.asarray(v, float))
    y = pd.Series(np.asarray(y))
    h = np.asarray(healthy, bool)
    out = np.full(len(v), np.nan)
    gmu, gsd = np.nanmean(v[h]), np.nanstd(v[h])
    for t in pd.unique(y):
        m = (y == t).values
        hv = v[m & h].dropna()
        mu, sd = (hv.mean(), hv.std()) if len(hv) >= min_n else (gmu, gsd)
        out[m] = (v[m] - mu) / (sd if sd > 0 else 1.0)
    return out


def std_rates(D, col, w, min_cells, by="donor"):
    """Direct standardization: per unit, sum_t w_t * rate_t over types with >= min_cells cells, renormalized."""
    d = D[[by, "y", col]].dropna()
    g = d.groupby([by, "y"])[col].agg(["sum", "size"])
    N = g["size"].unstack(fill_value=0)
    S = g["sum"].unstack(fill_value=0.0)
    Rt = S / N.where(N >= min_cells)
    W = pd.Series(w).reindex(Rt.columns).fillna(0).values[None, :]
    valid = Rt.notna().values
    num = np.nansum(Rt.values * W, 1)
    den = (W * valid).sum(1)
    std = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=Rt.index)
    raw = d.groupby(by)[col].mean()
    return std, raw, den


# ----------------------------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------------------------
@section("0_load")
def load_all():
    F, sF = load_tag(TAGS["full"])
    B, sB = load_tag(TAGS["blind"])
    common = F.index.intersection(B.index)
    log(f"cells: full {len(F)}, blind {len(B)}, common {len(common)}")
    if len(common) < 0.9 * min(len(F), len(B)):
        log("WARNING: fewer than 90% of cells shared between tags")
    D = F.loc[common, ["y", "cond_raw", "donor"]].copy()
    for tag, T, ss in [("full", F, sF), ("blind", B, sB)]:
        for k in DEFS:
            cs = [f"f_{k}_s{s}" for s in ss if f"f_{k}_s{s}" in T]
            if not cs:
                continue
            M = T.loc[common, cs].values
            D[f"{tag}_{k}"] = M.mean(1)
            for s, c in zip(ss, cs):
                D[f"{tag}_{k}_s{s}"] = M[:, cs.index(c)]
        nv = [f"novelty_s{s}" for s in ss]
        D[f"{tag}_novelty"] = T.loc[common, nv].mean(1).values
    D["is_sle"] = D.cond_raw.map(is_sle_str).astype(int)
    R["n"] = dict(cells=int(len(D)), donors=int(D.donor.nunique()),
                  sle_donors=int(D[D.is_sle == 1].donor.nunique()),
                  healthy_donors=int(D[D.is_sle == 0].donor.nunique()), seeds_full=sF, seeds_blind=sB,
                  cond_levels=D.cond_raw.value_counts().to_dict())
    log(R["n"])
    return D, sF, sB


@section("1_expression_scores")
def expression(D, gs, dropped):
    rng = np.random.default_rng(0)
    cells = D.index.values
    if args.max_expr_per_donor > 0:
        keep = []
        for d, g in D.groupby("donor"):
            idx = g.index.values
            keep.append(rng.choice(idx, min(len(idx), args.max_expr_per_donor), replace=False))
        cells = D.index[D.index.isin(np.concatenate(keep))].values
        log(f"expression subsample: {len(cells)} cells")
    X, var, obs, tot = load_expression(cells, D.index.values)
    # gene sets: full and "no-dropped" versions for sets partially dropped
    sets = dict(gs)
    frac_drop = {}
    vset = set(var)
    for k, g in gs.items():
        inv = [x for x in g if x in vset]
        fd = np.mean([x in dropped for x in inv]) if inv else np.nan
        frac_drop[k] = fd
        nd = [x for x in inv if x not in dropped]
        if 0 < fd < 1 and len(nd) >= 3:
            sets[k + "__nodrop"] = nd
    S, used = score_sets(X, var, sets, rng)
    S.index = pd.Index(cells)
    R["genesets_used_n"] = {k: len(v) for k, v in used.items()}
    R["geneset_frac_dropped"] = frac_drop
    E = S.add_prefix("gs_")
    E["log_counts"] = np.log10(np.maximum(tot, 1))
    return X, var, obs, E, frac_drop


def build_donor_table(D, obs, E, keys):
    cols = list(obs.columns)
    det = dict(
        cohort=find_col(cols, ["Processing_Cohort", "processing_cohort", "cohort"], [r"process.*cohort", r"cohort"]),
        ancestry=find_col(cols, ["self_reported_ethnicity", "pop_cov", "ancestry", "ethnicity", "race"],
                          [r"ethnic", r"ancestr", r"pop_cov", r"race"]),
        sex=find_col(cols, ["sex", "Sex", "sex_cov", "gender"], [r"^sex", r"gender"]),
        age=find_col(cols, ["Age", "age", "age_cov", "age_years"], [r"^age"]),
        dev_stage=find_col(cols, ["development_stage"], [r"development_stage"]),
        status=find_col(cols, ["disease_state", "Status", "status"], [r"disease_state", r"^status$", r"flare"]),
        sample=find_col(cols, ["sample_uuid", "sample_id", "sample"], [r"^sample"]),
        parent=find_col(cols, ["parent"]),
        donor_id=find_col(cols, ["donor_id", "ind_cov", "donor"], [r"^donor"]),
    )
    R["columns"] = det
    log("detected columns:", det)
    O = pd.DataFrame(index=obs.index)
    for k, c in det.items():
        if c:
            O[k] = obs[c].astype(str).values
    D = D.join(O)
    D = D.join(E)
    # unit of replication: if npz `batch` is finer than obs donor_id (e.g. sample/batch per donor), use donor_id,
    # otherwise repeated samples of one donor are treated as independent donors (pseudoreplication)
    if "donor_id" in D:
        bad = D.donor_id.astype(str).str.lower().isin(["nan", "none", "", "na", "unknown"])
        mp = D[~bad].groupby("donor").donor_id.agg(lambda v: v.value_counts().index[0])
        if len(mp) == D.donor.nunique() and mp.nunique() < len(mp):
            log(f"WARNING: npz batch has {len(mp)} levels but obs donor_id only {mp.nunique()}; using donor_id "
                "as the donor unit")
            R["donor_unit"] = "obs.donor_id (batch was finer)"
            D["donor"] = D.donor.map(mp).astype(str)
            mix = D.groupby("donor").is_sle.nunique()
            if (mix > 1).any():
                raise ValueError(f"donors with both SLE and healthy cells: {list(mix[mix > 1].index[:5])}")
        else:
            R["donor_unit"] = "npz batch"
    if "status" in D:
        D["status"] = D.status.str.strip().str.lower()
    healthy = (D.is_sle == 0).values
    has = D[keys["isg"]].notna().values if keys["isg"] in D else np.zeros(len(D), bool)
    for nm, k in keys.items():
        if k and k in D:
            z = np.full(len(D), np.nan)
            z[has] = z_by_type(D.loc[has, k].values, D.loc[has, "y"].values, healthy[has])
            D[nm + "_z"] = z
    # ISG-only detector calibrated like p_unknown_ctrl: ISG z above the healthy-query q95 of its type
    # Healthy donors get leave-one-donor-out thresholds (as p_unknown_ctrl does for controls); an in-sample q95
    # would hold healthy donors at <=5% by construction and inflate the detector's SLE excess.
    if "isg_z" in D:
        q = 1 - args.alpha
        H = D[healthy & has]
        thr = H.groupby("y").isg_z.quantile(q)
        t = D.y.map(thr).astype(float).fillna(np.nanquantile(H.isg_z, q)).values.copy()
        don = D.donor.values
        for d in pd.unique(H.donor):
            md = (don == d) & has
            rest = H[H.donor != d]
            if len(rest) == 0:
                continue
            td = D.y[md].map(rest.groupby("y").isg_z.quantile(q)).astype(float)
            t[md] = td.fillna(np.nanquantile(rest.isg_z, q)).values
        D["isgdet_ctrl"] = np.where(has, (D.isg_z.values > t).astype(float), np.nan)
    # donor table
    g = D.groupby("donor")
    DN = pd.DataFrame(dict(is_sle=g.is_sle.max(), n_cells=g.size()))
    for k in ["cohort", "ancestry", "sex", "age", "dev_stage", "status"]:
        if k in D:
            DN[k] = g[k].agg(lambda s: s.value_counts().index[0] if s.notna().any() else np.nan)
    if "status" in D:
        DN["n_status"] = g.status.nunique()
    src = DN["age"] if "age" in DN else DN.get("dev_stage")
    DN["age_num"] = src.map(parse_age) if src is not None else np.nan
    if "dev_stage" in DN and DN.age_num.isna().mean() > 0.5:
        DN["age_num"] = DN.dev_stage.map(parse_age)
    for nm in ["isg", "gc", "diss"]:
        if nm + "_z" in D:
            DN[nm] = g[nm + "_z"].mean()
    return D, DN, det


@section("Q1_donor_excess")
def q1(D, DN, w, sB):
    out = {}
    rows = []
    defs = [(t, k) for t in ["full", "blind"] for k in DEFS if f"{t}_{k}" in D] + \
           ([("isgdet", "ctrl")] if "isgdet_ctrl" in D else [])
    for t, k in defs:
        col = f"{t}_{k}"
        std, raw, cov = std_rates(D, col, w, args.min_cells_type)
        DN[f"r_{col}"] = std.reindex(DN.index) * 100
        DN[f"raw_{col}"] = raw.reindex(DN.index) * 100
        s = DN[DN.is_sle == 1]
        h = DN[DN.is_sle == 0]
        e = excess_boot(s[f"r_{col}"], h[f"r_{col}"])
        er = excess_boot(s[f"raw_{col}"], h[f"raw_{col}"])
        rows.append(dict(model=t, flag=k, **{f"std_{a}": b for a, b in e.items()},
                         raw_excess=er["excess"], raw_ci_lo=er["ci_lo"], raw_ci_hi=er["ci_hi"],
                         composition_share=1 - e["excess"] / er["excess"] if er["excess"] else np.nan))
    T = pd.DataFrame(rows)
    wcsv(T, "Q1_excess.csv")
    out["excess"] = T
    # retained fraction (paired bootstrap over the same donor resamples)
    s = DN[DN.is_sle == 1]
    h = DN[DN.is_sle == 0]
    iS, iH = boot_idx(len(s), len(h), args.nboot, seed=11)
    ret = {}
    for k in DEFS:
        a, b = f"r_full_{k}", f"r_blind_{k}"
        if a not in DN or b not in DN:
            continue
        sv, hv = s[[a, b]].values, h[[a, b]].values
        exS = np.nanmean(sv[iS], 1)
        exH = np.nanmean(hv[iH], 1)
        ex = exS - exH
        frac_b = ex[:, 1] / np.where(ex[:, 0] != 0, ex[:, 0], np.nan)
        pt_full = np.nanmean(sv[:, 0]) - np.nanmean(hv[:, 0])
        pt_blind = np.nanmean(sv[:, 1]) - np.nanmean(hv[:, 1])
        lo, hi = ci(frac_b)
        dl, dh = ci(ex[:, 0] - ex[:, 1])
        ret[k] = dict(excess_full=pt_full, excess_blind=pt_blind, retained=pt_blind / pt_full if pt_full else np.nan,
                      retained_ci_lo=lo, retained_ci_hi=hi, loss_full_minus_blind=pt_full - pt_blind,
                      loss_ci_lo=dl, loss_ci_hi=dh)
    out["retained"] = ret
    log("retained:", json.dumps(clean(ret)))
    # per-seed blind excess (seed stability)
    ps = {}
    for sd in sB:
        col = f"blind_{args.primary}_s{sd}"
        if col in D:
            std, _, _ = std_rates(D, col, w, args.min_cells_type)
            v = std.reindex(DN.index) * 100
            ps[f"seed{sd}"] = excess_boot(v[DN.is_sle == 1], v[DN.is_sle == 0], nboot=500)
    out["blind_per_seed"] = ps
    # ISG-low SLE donors vs healthy
    if "isg" in DN:
        thr = DN[DN.is_sle == 0].isg.quantile(args.isg_low_q)
        low = DN[(DN.is_sle == 1) & (DN.isg <= thr)]
        # comparator restricted to the same ISG range (healthy ISG <= thr), so residual ISG cannot drive N3
        hl = h[h.isg <= thr]
        il = dict(isg_threshold=float(thr), n_isg_low_sle=int(len(low)), n_sle=int(len(s)),
                  n_healthy_matched=int(len(hl)), isg_mean_low_sle=float(low.isg.mean()),
                  isg_mean_healthy_matched=float(hl.isg.mean()),
                  isg_gap=excess_boot(low.isg, hl.isg, nboot=1000))
        for t in ["full", "blind", "isgdet"]:
            c = f"r_{t}_{args.primary}"
            if c in DN:
                il[t] = excess_boot(low[c], hl[c])
                il[t + "_vs_all_healthy"] = excess_boot(low[c], h[c])
        out["isg_low_sle"] = il
        DN["isg_low"] = ((DN.is_sle == 1) & (DN.isg <= thr)).astype(int)
    # cohort and cohort x ancestry strata
    strat = {}
    for t in ["full", "blind", "isgdet"]:
        c = f"r_{t}_{args.primary}"
        if c not in DN:
            continue
        for nm, st in [("cohort", ["cohort"]), ("cohort_x_ancestry", ["cohort", "ancestry"])]:
            if all(x in DN for x in st):
                e, Ts = stratified_excess(DN, c, st)
                strat[f"{t}|{nm}"] = e
                if t == "blind":
                    wcsv(Ts, f"Q1_strata_{nm}.csv")
    out["stratified"] = strat
    # crosstab cohort vs condition (separation check)
    if "cohort" in DN:
        out["cohort_by_condition"] = pd.crosstab(DN.cohort, DN.is_sle).to_dict()
    # adjusted OLS (HC3)
    adj = {}
    cats = [c for c in ["cohort", "ancestry", "sex"] if c in DN]
    for t in ["full", "blind"]:
        for k in [args.primary, "bonf"]:
            c = f"r_{t}_{k}"
            if c not in DN:
                continue
            for mn, nums in [("demog", ["is_sle", "age_num"]),
                             ("demog+ISG+IEG", ["is_sle", "age_num", "isg", "diss"]),
                             ("demog+ISG+IEG+GC", ["is_sle", "age_num", "isg", "diss", "gc"])]:
                try:
                    Tm, info = ols_robust(DN, c, cats=cats, nums=[x for x in nums if x in DN])
                    row = Tm[Tm.term == "is_sle"]
                    adj[f"{t}|{k}|{mn}"] = dict(info, **(row.iloc[0].to_dict() if len(row) else
                                                         {"error": "is_sle collinear with covariates "
                                                                   "(cohort separates condition?)"}))
                    wcsv(Tm, f"Q1_ols_{t}_{k}_{safe_name(mn)}.csv")
                except Exception as e:  # noqa: BLE001
                    adj[f"{t}|{k}|{mn}"] = dict(error=repr(e))
    out["ols_is_sle"] = adj
    R["Q1"] = out
    # figure: donor standardized rates
    try:
        cols = [c for c in [f"r_full_{args.primary}", f"r_blind_{args.primary}", f"r_isgdet_ctrl"] if c in DN]
        fig, ax = plt.subplots(figsize=(3.2 * len(cols) + 1, 5.5))
        rng = np.random.default_rng(0)
        for i, c in enumerate(cols):
            for j, (grp, sub) in enumerate([("healthy", h), ("SLE", s)]):
                x = i * 3 + j + rng.uniform(-0.25, 0.25, len(sub))
                ax.scatter(x, sub[c], s=14, alpha=0.6, color=COL[grp], label=grp if i == 0 else None)
                ax.plot([i * 3 + j - 0.35, i * 3 + j + 0.35], [sub[c].mean()] * 2, color="k", lw=3)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(cols))])
        ax.set_xticklabels([{"full": "full", "blind": "ISG-blind", "isgdet": "ISG score"}[c.split("_")[1]]
                            for c in cols])
        ax.set_ylabel("flagged cells, std. (%)")
        ax.legend(frameon=False)
        savefig(fig, "Q1_donor_rates.png")
    except Exception as e:  # noqa: BLE001
        log("Q1 figure failed", repr(e))
    return out


@section("Q2_within_SLE_vs_ISG")
def q2(DN):
    out = {}
    s = DN[DN.is_sle == 1]
    rows = []
    for t in ["full", "blind", "isgdet"]:
        for k in DEFS:
            c = f"r_{t}_{k}"
            if c not in DN:
                continue
            for cov in ["isg", "gc", "diss"]:
                if cov not in DN:
                    continue
                for grp, sub in [("SLE", s), ("healthy", DN[DN.is_sle == 0])]:
                    r = spearman_boot(sub[c], sub[cov])
                    rows.append(dict(model=t, flag=k, covariate=cov, group=grp, **r))
    T = pd.DataFrame(rows)
    wcsv(T, "Q2_spearman.csv")
    out["table"] = T
    # paired bootstrap difference rho_full - rho_blind (SLE, ISG)
    diff = {}
    for k in DEFS:
        a, b = f"r_full_{k}", f"r_blind_{k}"
        if a not in s or b not in s or "isg" not in s:
            continue
        d = s[[a, b, "isg"]].dropna().values
        n = len(d)
        if n < 6:
            continue
        rng = np.random.default_rng(5)
        bs = []
        for _ in range(args.nboot):
            i = rng.integers(0, n, n)
            bs.append(rank_rho(d[i, 0], d[i, 2]) - rank_rho(d[i, 1], d[i, 2]))
        lo, hi = ci(bs)
        diff[k] = dict(rho_full=rank_rho(d[:, 0], d[:, 2]), rho_blind=rank_rho(d[:, 1], d[:, 2]),
                       diff=rank_rho(d[:, 0], d[:, 2]) - rank_rho(d[:, 1], d[:, 2]), ci_lo=lo, ci_hi=hi, n=n)
    out["rho_full_minus_blind"] = diff
    R["Q2"] = out
    try:
        k = args.primary
        fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
        for ax, t in zip(axs, ["full", "blind"]):
            c = f"r_{t}_{k}"
            for grp, sub in [("healthy", DN[DN.is_sle == 0]), ("SLE", s)]:
                ax.scatter(sub.isg, sub[c], s=18, alpha=0.7, color=COL[grp], label=grp)
            ax.set_xlabel("donor ISG z")
            ax.set_ylabel(f"{'full' if t == 'full' else 'ISG-blind'} flagged (%)")
        axs[0].legend(frameon=False)
        savefig(fig, "Q2_rate_vs_isg.png")
    except Exception as e:  # noqa: BLE001
        log("Q2 figure failed", repr(e))
    return out


@section("Q3_types")
def q3(D, DN):
    rows = []
    isg_low = set(DN.index[DN.get("isg_low", pd.Series(0, index=DN.index)) == 1])
    for t_ in sorted(D.y.unique()):
        g = D[D.y == t_]
        n = g.groupby("donor").size()
        ok = n[n >= args.min_cells_type].index
        g = g[g.donor.isin(ok)]
        if g.empty:
            continue
        sle = DN.loc[ok, "is_sle"]
        for t in ["full", "blind", "isgdet"]:
            for k in DEFS:
                c = f"{t}_{k}"
                if c not in g:
                    continue
                r = g.groupby("donor")[c].mean() * 100
                vs, vh = r[sle.reindex(r.index) == 1], r[sle.reindex(r.index) == 0]
                e = excess_boot(vs, vh, nboot=1000)
                el = excess_boot(vs[vs.index.isin(isg_low)], vh, nboot=1000) if isg_low else {}
                share = g[g.is_sle == 1][c].sum() / max(D[D.is_sle == 1][c].sum(), 1e-9)
                rows.append(dict(type=t_, model=t, flag=k, share_of_SLE_flags=share, **e,
                                 excess_isg_low_sle=el.get("excess", np.nan),
                                 excess_isg_low_ci_lo=el.get("ci_lo", np.nan),
                                 excess_isg_low_ci_hi=el.get("ci_hi", np.nan), n_isg_low=el.get("n_sle", 0)))
    T = pd.DataFrame(rows)
    T["q"] = np.nan
    for (m, k), idx in T.groupby(["model", "flag"]).groups.items():
        T.loc[idx, "q"] = bh(T.loc[idx, "p_mwu"].values)
    P = T[T.flag == args.primary].pivot(index="type", columns="model", values="excess")
    if "full" in P and "blind" in P:
        T = T.merge((P.blind / P.full.where(P.full > 0)).rename("retained_type").reset_index(), on="type", how="left")
    wcsv(T, "Q3_types.csv")
    kb = T[(T.model == "blind") & (T.flag == args.primary) & (T.q < 0.05) & (T.ci_lo > 0)].sort_values(
        "excess", ascending=False)
    out = dict(types_with_blind_excess=kb.type.tolist(),
               table_primary=T[(T.flag == args.primary)][["type", "model", "excess", "ci_lo", "ci_hi", "q",
                                                         "share_of_SLE_flags", "excess_isg_low_sle"]])
    R["Q3"] = out
    try:
        tp = T[T.flag == args.primary]
        P2 = tp.pivot(index="type", columns="model", values="excess").sort_values("blind")
        lo = tp.pivot(index="type", columns="model", values="ci_lo").reindex(P2.index)
        hi = tp.pivot(index="type", columns="model", values="ci_hi").reindex(P2.index)
        fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(P2) + 2))
        yy = np.arange(len(P2))
        for off, m in [(-0.2, "full"), (0.2, "blind")]:
            if m in P2:
                ax.errorbar(P2[m], yy + off, xerr=[P2[m] - lo[m], hi[m] - P2[m]], fmt="o", color=COL[m],
                            label="full" if m == "full" else "ISG-blind", ms=6)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels(P2.index)
        ax.set_xlabel("SLE - healthy flagged (pp)")
        ax.legend(frameon=False)
        savefig(fig, "Q3_type_excess.png")
    except Exception as e:  # noqa: BLE001
        log("Q3 figure failed", repr(e))
    return out


def binarize(D, tag, k, seeds):
    cs = [f"{tag}_{k}_s{s}" for s in seeds if f"{tag}_{k}_s{s}" in D]
    M = D[cs].values
    return M.min(1) >= 1, M.max(1) <= 0  # flagged in all seeds, flagged in none


def pseudobulk_paired(X, rowpos, g, ga, gb, rng):
    ca, cb = g[ga].groupby("donor").size(), g[gb].groupby("donor").size()
    donors = sorted(set(ca[ca >= args.min_cells_pb].index) & set(cb[cb >= args.min_cells_pb].index))
    if len(donors) < args.min_donors_pb:
        return None, len(donors)
    cells, rows = [], []
    for kd, d in enumerate(donors):
        for j, m in enumerate([ga, gb]):
            c = g.index[m & (g.donor == d).values].values
            if len(c) > args.max_per_group:
                c = rng.choice(c, args.max_per_group, replace=False)
            cells.append(c)
            rows += [2 * kd + j] * len(c)
    cells = np.concatenate(cells)
    rows = np.array(rows)
    Xs = X[rowpos.reindex(cells).values]
    cnt = np.bincount(rows, minlength=2 * len(donors)).astype(float)
    Mi = sp.csr_matrix((1.0 / cnt[rows], (rows, np.arange(len(rows)))), shape=(2 * len(donors), len(rows)))
    Mean = np.asarray((Mi @ Xs).todense())
    Xb = Xs.copy()
    Xb.data[:] = 1
    detf = np.asarray((Mi @ Xb).todense())
    keep = (detf[0::2].mean(0) >= 0.05) | (detf[1::2].mean(0) >= 0.05)
    diff = Mean[0::2][:, keep] - Mean[1::2][:, keep]
    tt = stats.ttest_1samp(diff, 0, axis=0)
    return pd.DataFrame(dict(gi=np.where(keep)[0], mean_diff=diff.mean(0), t=tt.statistic, p=tt.pvalue,
                             frac_donors_up=(diff > 0).mean(0))), len(donors)


def enrich(genes, universe, gs, min_size=3):
    rows = []
    M, N, up = len(universe), len(genes), set(genes)
    for name, g in gs.items():
        G = set(g) & universe
        if len(G) < min_size or N == 0:
            continue
        k = len(G & up)
        p = stats.hypergeom.sf(k - 1, M, len(G), N) if k > 0 else 1.0
        rows.append(dict(set=name, size=len(G), overlap=k, expected=N * len(G) / M, p=p,
                         genes=",".join(sorted(G & up))))
    E = pd.DataFrame(rows)
    if len(E):
        E["q"] = bh(E.p.values)
        E = E.sort_values("p")
    return E


@section("Q4_blind_flagged_DE")
def q4(D, DN, X, var, gs, dropped, sB, sF, frac_drop, q3out):
    out = {}
    rng = np.random.default_rng(13)
    has = D["log_counts"].notna()
    Dx = D[has].copy()
    fl, kp = binarize(Dx, "blind", args.primary, sB)
    Dx["bflag"], Dx["bkept"] = fl, kp
    rowpos = pd.Series(np.arange(int(has.sum())), index=Dx.index)  # X rows follow D order of expressed cells
    gnames = np.asarray(var)
    drop_mask = np.isin(gnames, list(dropped))
    gs_nd = {k: [g for g in v if g not in dropped] for k, v in gs.items()}
    cnt = Dx[(Dx.is_sle == 1) & Dx.bflag].y.value_counts()
    pri = [t for t in (q3out or {}).get("types_with_blind_excess", []) if cnt.get(t, 0) >= 50]
    types = pri + [t for t in cnt[cnt >= 50].index if t not in pri]
    types = types[:args.max_pb_types]
    out["types"] = types
    gs_cols = [c for c in Dx.columns if c.startswith("gs_")]
    delta_rows = []
    DV = {}  # per-donor delta vectors, key "type|group|set" (used by gate N6 for SLE-vs-healthy contrast)
    for t in types:
        r = {}
        for cn, sle in [("SLE", 1), ("healthy", 0)]:
            g = Dx[(Dx.y == t) & (Dx.is_sle == sle)]
            # gene-set score deltas flagged - kept, per donor
            ga, gb = g.bflag.values, g.bkept.values
            ca = g[ga].groupby("donor").size()
            cb = g[gb].groupby("donor").size()
            min_fl = 3
            dd = sorted(set(ca[ca >= 3].index) & set(cb[cb >= 3].index))
            if len(dd) < 3 and sle == 0:  # thin healthy control: accept donors with >=1 flagged cell
                min_fl = 1
                dd = sorted(set(ca[ca >= 1].index) & set(cb[cb >= 3].index))
            if len(dd) >= 3:
                ma = g[ga & g.donor.isin(dd).values].groupby("donor")[gs_cols].mean()
                mb = g[gb & g.donor.isin(dd).values].groupby("donor")[gs_cols].mean()
                dlt = ma - mb
                for c in gs_cols:
                    e = boot_mean_ci(dlt[c].values, nboot=1000)
                    nm = c[3:]
                    DV[f"{t}|{cn}|{nm}"] = dlt[c].values
                    delta_rows.append(dict(type=t, group=cn, set=nm, min_flagged_cells=min_fl,
                                           frac_genes_dropped=frac_drop.get(
                        nm.replace("__nodrop", ""), np.nan) if not nm.endswith("__nodrop") else 0.0, **e))
            df, nd = pseudobulk_paired(X, rowpos, g, ga, gb, rng)
            if df is None:
                r[cn] = dict(n_donors=nd, note=f"<{args.min_donors_pb} donors with >={args.min_cells_pb} "
                                               "flagged and kept cells")
                continue
            df["gene"] = gnames[df.gi.values]
            df["dropped"] = drop_mask[df.gi.values]
            df["q"] = bh(df.p.values)
            df["q_nodrop"] = np.nan
            nd_m = ~df.dropped.values
            df.loc[nd_m, "q_nodrop"] = bh(df.loc[nd_m, "p"].values)
            df = df.drop(columns="gi").sort_values("t", ascending=False)
            df.head(400).to_csv(os.path.join(OUT, f"Q4_pb_{safe_name(t)}_{cn}.csv"), index=False)
            df.tail(200).to_csv(os.path.join(OUT, f"Q4_pb_{safe_name(t)}_{cn}_down.csv"), index=False)
            sel = (df.q_nodrop < 0.05) & (df.frac_donors_up >= 0.7) & (df.mean_diff > 0.1)
            seld = (df.q_nodrop < 0.05) & (df.frac_donors_up <= 0.3) & (df.mean_diff < -0.1)
            up, dn = df[sel].gene.tolist(), df[seld].sort_values("t").gene.tolist()
            up_all = df[(df.q < 0.05) & (df.frac_donors_up >= 0.7) & (df.mean_diff > 0.1)].gene.tolist()
            univ = set(df.gene[nd_m])
            E = enrich(up[:200], univ, gs_nd)
            Ed = enrich(dn[:200], univ, gs_nd)
            if len(E):
                E.to_csv(os.path.join(OUT, f"Q4_enrich_up_{safe_name(t)}_{cn}.csv"), index=False)
            if len(Ed):
                Ed.to_csv(os.path.join(OUT, f"Q4_enrich_down_{safe_name(t)}_{cn}.csv"), index=False)
            r[cn] = dict(n_donors=nd, n_up_nodrop=len(up), n_down_nodrop=len(dn), top_up=up[:20], top_down=dn[:12],
                         n_up_incl_dropped=len(up_all),
                         dropped_genes_in_top50_up_incl_dropped=int(np.isin(up_all[:50], list(dropped)).sum()),
                         top_up_dropped=[x for x in up_all if x in dropped][:10],
                         enriched_up=E[E.q < 0.05][["set", "overlap", "q"]].values.tolist() if len(E) else [],
                         enriched_down=Ed[Ed.q < 0.05][["set", "overlap", "q"]].values.tolist() if len(Ed) else [])
        if "top_up" in r.get("SLE", {}) and "top_up" in r.get("healthy", {}):
            r["SLE_specific_up"] = [x for x in r["SLE"]["top_up"] if x not in set(r["healthy"]["top_up"])]
        out[t] = r
        log(f"  {t}: SLE up {r.get('SLE', {}).get('top_up', [])[:10]}")
    Dl = pd.DataFrame(delta_rows)
    if len(Dl):
        Dl["q"] = np.nan
        for (tt, gg), idx in Dl.groupby(["type", "group"]).groups.items():
            Dl.loc[idx, "q"] = bh(Dl.loc[idx, "p_wilcoxon"].values)
        wcsv(Dl, "Q4_geneset_delta.csv")
    out["geneset_delta"] = Dl
    out["delta_values"] = DV
    R["Q4"] = {k: v for k, v in out.items() if k not in ("geneset_delta", "delta_values")}
    return out


@section("Q5_overlap")
def q5(D, DN, w, sF, sB):
    out = {}
    common = sorted(set(sF) & set(sB)) or sB
    k = args.primary
    Ff, _ = binarize(D, "full", k, common)
    Bf, _ = binarize(D, "blind", k, common)

    def ov(a, b):
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        n11, n10, n01 = (a & b).sum(), (a & ~b).sum(), (~a & b).sum()
        n00 = len(a) - n11 - n10 - n01
        lor = np.log((n11 + .5) * (n00 + .5) / ((n10 + .5) * (n01 + .5)))
        se = np.sqrt(1 / (n11 + .5) + 1 / (n00 + .5) + 1 / (n10 + .5) + 1 / (n01 + .5))
        return dict(n_a=int(a.sum()), n_b=int(b.sum()), n_both=int(n11),
                    jaccard=float(n11 / max(n11 + n10 + n01, 1)), frac_b_in_a=float(n11 / max(b.sum(), 1)),
                    odds_ratio=float(np.exp(lor)), or_ci_lo=float(np.exp(lor - 1.96 * se)),
                    or_ci_hi=float(np.exp(lor + 1.96 * se)))
    rows = []
    for grp, m in [("all", np.ones(len(D), bool)), ("SLE", (D.is_sle == 1).values), ("healthy", (D.is_sle == 0).values)]:
        rows.append(dict(comparison=f"full_vs_blind(all {len(common)} common seeds)", group=grp, **ov(Ff[m], Bf[m])))
        for s in common:
            a, b = D[f"full_{k}_s{s}"].values[m] > 0, D[f"blind_{k}_s{s}"].values[m] > 0
            rows.append(dict(comparison=f"full_s{s}_vs_blind_s{s}", group=grp, **ov(a, b)))
        for tag, ss in [("full", sF), ("blind", sB)]:
            if len(ss) >= 2:
                a, b = D[f"{tag}_{k}_s{ss[0]}"].values[m] > 0, D[f"{tag}_{k}_s{ss[1]}"].values[m] > 0
                rows.append(dict(comparison=f"{tag}_s{ss[0]}_vs_s{ss[1]} (reference)", group=grp, **ov(a, b)))
        if "isgdet_ctrl" in D:
            hm = m & D.isgdet_ctrl.notna().values
            idt = D.isgdet_ctrl.values[hm] > 0
            rows.append(dict(comparison="full_vs_ISGdetector", group=grp, **ov(Ff[hm], idt)))
            rows.append(dict(comparison="blind_vs_ISGdetector", group=grp, **ov(Bf[hm], idt)))
            # Jaccard is bounded by |small|/|large|: if the ISG detector flags many more cells than the blind
            # consensus, J<0.2 is automatic. Size-matched detector: per type, the top-n ISG-z cells where n is
            # the number of blind-flagged cells of that type in this group.
            if "isg_z" in D:
                for tag, FF in [("full", Ff), ("blind", Bf)]:
                    sm = np.zeros(len(D), bool)
                    yv, zv = D.y.values, D.isg_z.values
                    for t_ in pd.unique(yv[hm]):
                        mt = hm & (yv == t_)
                        nb = int(FF[mt].sum())
                        if nb == 0:
                            continue
                        ii = np.flatnonzero(mt)
                        sm[ii[np.argsort(-np.nan_to_num(zv[ii], nan=-np.inf), kind="stable")[:nb]]] = True
                    rows.append(dict(comparison=f"{tag}_vs_ISGdetector_sizematched", group=grp, **ov(FF[hm], sm[hm])))
    T = pd.DataFrame(rows)
    wcsv(T, "Q5_overlap.csv")
    out["overlap"] = T
    # per type overlap and blind-only
    D = D.copy()
    D["fcat"] = np.select([Ff & Bf, Ff & ~Bf, ~Ff & Bf], ["both", "full_only", "blind_only"], "neither")
    D["blind_only"] = (D.fcat == "blind_only").astype(float)
    D["full_only"] = (D.fcat == "full_only").astype(float)
    tt = []
    for t_, g in D.groupby("y"):
        r = ov(Ff[D.y.values == t_], Bf[D.y.values == t_])
        r.update(type=t_, n_cells=len(g), blind_only_SLE=int(((g.fcat == "blind_only") & (g.is_sle == 1)).sum()),
                 blind_only_healthy=int(((g.fcat == "blind_only") & (g.is_sle == 0)).sum()))
        tt.append(r)
    wcsv(pd.DataFrame(tt), "Q5_overlap_by_type.csv")
    # profile of categories (donor-averaged within category, SLE)
    prof = []
    for c in ["isg_z", "gc_z", "diss_z", "log_counts"]:
        if c not in D:
            continue
        for grp, sle in [("SLE", 1), ("healthy", 0)]:
            m = D[(D.is_sle == sle)]
            dm = m.groupby(["donor", "fcat"])[c].mean().unstack()
            for cat in ["both", "full_only", "blind_only", "neither"]:
                if cat in dm:
                    prof.append(dict(var=c, group=grp, cat=cat, donor_mean=float(dm[cat].mean()),
                                     n_donors=int(dm[cat].notna().sum())))
    P = pd.DataFrame(prof)
    wcsv(P, "Q5_category_profile.csv")
    out["profile"] = P
    ex = {}
    for c in ["blind_only", "full_only"]:
        std, _, _ = std_rates(D, c, w, args.min_cells_type)
        v = std.reindex(DN.index) * 100
        DN[f"r_{c}"] = v
        ex[c] = excess_boot(v[DN.is_sle == 1], v[DN.is_sle == 0])
        if "isg" in DN:
            ex[c + "_rho_isg_SLE"] = spearman_boot(v[DN.is_sle == 1], DN.isg[DN.is_sle == 1])
    out["donor_excess"] = ex
    out["blind_only_by_type_SLE"] = D[(D.fcat == "blind_only") & (D.is_sle == 1)].y.value_counts().head(10).to_dict()
    R["Q5"] = out
    return out


@section("Q6_disease_state")
def q6(D, DN, w):
    out = {}
    if "status" not in D:
        raise KeyError("no disease_state/status column")
    S = D[(D.is_sle == 1) & D.status.notna() & ~D.status.isin(["na", "nan", "none", "", "unknown"])].copy()
    S["unit"] = S.donor + "|" + S.status
    U = pd.DataFrame(index=sorted(S.unit.unique()))
    U["donor"] = [u.split("|")[0] for u in U.index]
    U["status"] = [u.split("|", 1)[1] for u in U.index]
    nU = S.groupby("unit").size()
    U["n_cells"] = nU
    U = U[U.n_cells >= 100]
    for c in ["cohort", "ancestry", "sex", "age_num"]:
        if c in DN:
            U[c] = DN.loc[U.donor, c].values
    for nm in ["isg", "gc", "diss"]:
        if nm + "_z" in S:
            U[nm] = S.groupby("unit")[nm + "_z"].mean().reindex(U.index)
    for t in ["full", "blind"]:
        for k in [args.primary, "bonf"]:
            c = f"{t}_{k}"
            if c in S:
                std, _, _ = std_rates(S, c, w, args.min_cells_type, by="unit")
                U[f"r_{c}"] = std.reindex(U.index) * 100
    out["status_counts_units"] = U.status.value_counts().to_dict()
    out["donors_multi_state"] = int((U.groupby("donor").size() > 1).sum())
    lev = U.status.value_counts()
    ref = "managed" if "managed" in lev.index else lev.index[0]
    U["status_c"] = U.status.where(U.status.isin(lev[lev >= 3].index), np.nan)
    Uc = U.dropna(subset=["status_c"])
    cats = [c for c in ["cohort", "ancestry", "sex"] if c in Uc]
    cl = "donor" if out["donors_multi_state"] > 0 else None
    res = {}
    for t in ["full", "blind"]:
        for k in [args.primary, "bonf"]:
            c = f"r_{t}_{k}"
            if c not in Uc:
                continue
            for mn, nums in [("demog", ["age_num"]), ("demog+ISG+IEG+GC", ["age_num", "isg", "diss", "gc"])]:
                for se_kind, clus in [("HC3", None), ("cluster", cl)]:
                    if se_kind == "cluster" and clus is None:
                        continue
                    try:
                        Tm, info = ols_robust(Uc, c, cats=["status_c"] + cats, nums=[x for x in nums if x in Uc],
                                              refs={"status_c": ref}, cluster=clus)
                        key = f"{t}|{k}|{mn}|{se_kind}"
                        res[key] = dict(info=info, status_terms=Tm[Tm.term.str.startswith("status_c")])
                        wcsv(Tm, f"Q6_ols_{t}_{k}_{safe_name(mn)}_{se_kind}.csv")
                    except Exception as e:  # noqa: BLE001
                        res[f"{t}|{k}|{mn}|{se_kind}"] = dict(error=repr(e))
    out["ols"] = res
    # unadjusted means per status
    out["means"] = {c: Uc.groupby("status_c")[c].mean().to_dict() for c in Uc.columns if c.startswith("r_")}
    # within-donor paired: flare vs other state of the same donor
    pair = {}
    for t in ["full", "blind"]:
        c = f"r_{t}_{args.primary}"
        if c not in U:
            continue
        P = U.pivot_table(index="donor", columns="status", values=c)
        for other in [x for x in P.columns if x != "flare"]:
            if "flare" in P:
                dv = (P["flare"] - P[other]).dropna()
                if len(dv) >= 3:
                    pair[f"{t}|flare_minus_{other}"] = boot_mean_ci(dv.values)
    out["paired_within_donor"] = pair
    wcsv(U.reset_index().rename(columns={"index": "unit"}), "Q6_units.csv")
    R["Q6"] = out
    try:
        want = "cluster" if out["donors_multi_state"] > 0 else "HC3"
        key = f"blind|{args.primary}|demog|{want}"
        key2 = f"full|{args.primary}|demog|{want}"
        fig, ax = plt.subplots(figsize=(7.5, 4))
        yy = 0
        lab = []
        for kk, col, nm in [(key2, COL["full"], "full"), (key, COL["blind"], "ISG-blind")]:
            if kk in res and "status_terms" in res[kk]:
                for _, r in res[kk]["status_terms"].iterrows():
                    ax.errorbar(r.coef, yy, xerr=[[r.coef - r.ci_lo], [r.ci_hi - r.coef]], fmt="o", color=col, ms=7)
                    lab.append(f"{nm}: {r.term.split('T.')[-1].rstrip(']')}")
                    yy += 1
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(lab)))
        ax.set_yticklabels(lab)
        ax.set_xlabel(f"vs {ref}: flagged (pp)")
        savefig(fig, "Q6_status_coef.png")
    except Exception as e:  # noqa: BLE001
        log("Q6 figure failed", repr(e))
    return out


# ----------------------------------------------------------------------------------------------
# gates and RESULTS.md
# ----------------------------------------------------------------------------------------------
def f3(x, d=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(x) else f"{x:.{d}f}"


def eci(e, d=2):
    if not isinstance(e, dict):
        return "NA"
    v = e.get("excess", e.get("mean", e.get("rho", e.get("coef"))))
    return f"{f3(v, d)} [{f3(e.get('ci_lo'), d)}, {f3(e.get('ci_hi'), d)}]"


def gates(q1o, q2o, q3o, q4o, q5o, q6o):
    G = {}
    k = args.primary
    try:
        T = q1o["excess"]
        b = T[(T.model == "blind") & (T.flag == k)].iloc[0]
        bb = T[(T.model == "blind") & (T.flag == "bonf")]
        ret = q1o["retained"][k]["retained"]
        ok = b.std_ci_lo > 0 and ret >= 0.5 and (len(bb) == 0 or bb.iloc[0].std_ci_lo > 0)
        G["N1 retained excess (blind CI>0, retained>=50%, bonf CI>0)"] = (bool(ok), f"retained {f3(ret)}")
    except Exception as e:  # noqa: BLE001
        G["N1 retained excess"] = (False, repr(e)[:80])
    try:
        T = q2o["table"]
        r = T[(T.model == "blind") & (T.flag == k) & (T.covariate == "isg") & (T.group == "SLE")].iloc[0]
        dd = q2o["rho_full_minus_blind"].get(k, {})
        ok = (r.ci_hi < 0.3 or r.ci_lo <= 0) and dd.get("ci_lo", -1) > 0
        G["N2 decoupled from ISG (blind rho CI<0.3 or incl. 0; full-blind rho CI>0)"] = (
            bool(ok), f"rho_blind {f3(r.rho)}, diff {f3(dd.get('diff'))}")
    except Exception as e:  # noqa: BLE001
        G["N2 decoupled from ISG"] = (False, repr(e)[:80])
    try:
        il = q1o["isg_low_sle"]
        ok = il["blind"]["ci_lo"] > 0 and il["n_isg_low_sle"] >= 10
        G["N3 excess in ISG-low SLE donors (n>=10, CI>0)"] = (bool(ok), f"n={il['n_isg_low_sle']}, "
                                                                    f"{eci(il['blind'])}")
    except Exception as e:  # noqa: BLE001
        G["N3 ISG-low SLE donors"] = (False, repr(e)[:80])
    try:
        a = q1o["ols_is_sle"][f"blind|{k}|demog+ISG+IEG"]
        a2 = q1o["ols_is_sle"].get(f"blind|{k}|demog+ISG+IEG+GC", {})
        st = q1o["stratified"].get("blind|cohort", {})
        ok = a.get("ci_lo", -1) > 0 and st.get("ci_lo", -1) > 0
        treat = bool(ok and a2 and np.isfinite(a2.get("ci_lo", np.nan)) and a2["ci_lo"] <= 0)
        R["treatment_associated"] = treat
        G["N4 is_sle CI>0 adj. cohort/ancestry/sex/age/ISG/IEG, and cohort-stratified CI>0"] = (
            bool(ok), f"OLS {eci(a)}; +GC {eci(a2)}; strat {eci(st)}"
            + ("; excess lost after GC -> TREATMENT-ASSOCIATED" if treat else ""))
    except Exception as e:  # noqa: BLE001
        G["N4 adjusted"] = (False, repr(e)[:80])
    try:
        T = q5o["overlap"]
        j = T[(T.comparison == "blind_vs_ISGdetector") & (T.group == "SLE")].iloc[0].jaccard
        jm = T[(T.comparison == "blind_vs_ISGdetector_sizematched") & (T.group == "SLE")]
        jm = jm.iloc[0].jaccard if len(jm) else np.nan
        ok = j < 0.2 and (not np.isfinite(jm) or jm < 0.2)
        G["N5 blind flags != ISG-detector flags (Jaccard<0.2, SLE; also size-matched)"] = (
            bool(ok), f"J={f3(j)}, size-matched J={f3(jm)}")
    except Exception as e:  # noqa: BLE001
        G["N5 ISG detector"] = (False, repr(e)[:80])
    try:
        Dl = q4o["geneset_delta"]
        DV = q4o.get("delta_values", {})
        hits, notes = [], []
        for t in q3o["types_with_blind_excess"]:
            r = q4o.get(t, {}).get("SLE", {})
            sets = [s for s, _, q in r.get("enriched_up", []) if not re.search(EXCLUDE_BIO, s, re.I)]
            d = Dl[(Dl.type == t)]
            diss = d[(d.group == "SLE") & d.set.str.contains(r"dissoc|(?<!no)IEG|heat", case=False)]
            diss_ok = len(diss) == 0 or not (diss.ci_lo > 0).any()
            if not diss_ok:
                notes.append(f"{t}: flagged IEG-high")
            for s in sets:
                nm = s + "__nodrop" if f"{t}|SLE|{s}__nodrop" in DV else s
                vs, vh = DV.get(f"{t}|SLE|{nm}"), DV.get(f"{t}|healthy|{nm}")
                if vs is None or vh is None or len(vh) < 3:
                    notes.append(f"{t}:{s} no healthy control")  # absence of a control is not a pass
                    continue
                e = excess_boot(vs, vh, nboot=1000)  # SLE delta - healthy delta (donor bootstrap)
                if diss_ok and e["ci_lo"] > 0:
                    hits.append(f"{t}:{s} (SLE-healthy delta {f3(e['excess'])} [{f3(e['ci_lo'])}, {f3(e['ci_hi'])}])")
        G["N6 non-IFN/IEG gene programme in a type with blind excess, SLE-specific, not IEG-high"] = (
            bool(hits), ";".join(hits[:4]) or ("none; " + ";".join(notes[:4])))
    except Exception as e:  # noqa: BLE001
        G["N6 biology"] = (False, repr(e)[:80])
    try:
        res = q6o["ols"]
        want = "cluster" if q6o.get("donors_multi_state", 0) > 0 else "HC3"
        st = res[f"blind|{k}|demog|{want}"]["status_terms"]
        sig = st[(st.ci_lo > 0) | (st.ci_hi < 0)]
        pr = {kk: v for kk, v in q6o.get("paired_within_donor", {}).items() if kk.startswith("blind|")}
        psig = [kk for kk, v in pr.items() if v.get("ci_lo", 0) > 0 or v.get("ci_hi", 0) < 0]
        G["N7 (supportive) blind rate ~ disease_state (adjusted OLS or within-donor paired CI excl. 0)"] = (
            bool(len(sig) or psig), ";".join(f"{r.term.split('T.')[-1].rstrip(']')} {f3(r.coef)}"
                                             for _, r in st.iterrows()) +
            "; paired " + ";".join(f"{kk.split('|')[1]} {eci(v)}" for kk, v in pr.items()))
    except Exception as e:  # noqa: BLE001
        G["N7 disease_state"] = (False, repr(e)[:80])
    core = [v[0] for kk, v in G.items() if kk[:2] in ["N1", "N2", "N3", "N4", "N5", "N6"]]
    verdict = "NOVEL (all N1-N6 pass)" if core and all(core) else \
        f"NOT established ({sum(core)}/6 core gates pass)"
    if R.get("treatment_associated"):
        verdict += "; SLE excess of blind flags is TREATMENT-ASSOCIATED (vanishes after GC adjustment)"
    return G, verdict


def write_results(q1o, q2o, q3o, q4o, q5o, q6o):
    L = []
    k = args.primary
    n = R.get("n", {})
    L.append("# ISG-blind vs full ARCANE (SLE, Perez)")
    L.append(f"Tags: full=`{TAGS['full']}` seeds {n.get('seeds_full')}, blind=`{TAGS['blind']}` seeds "
             f"{n.get('seeds_blind')}. Cells {n.get('cells')}, donors SLE {n.get('sle_donors')} / healthy-query "
             f"{n.get('healthy_donors')}. Primary flag `{k}` (alpha {args.alpha}); rates are composition-standardized "
             f"to the healthy-query type mix, in % of cells, mean over seeds; CIs = 95% donor bootstrap "
             f"({args.nboot}). ISG-detector = cells with type-adjusted ISG z above the healthy-query q95 "
             "(same calibration as p_unknown_ctrl).")
    if R["errors"]:
        L.append(f"Errors: {', '.join(R['errors'])} (see log.txt).")
    try:
        G, verdict = gates(q1o, q2o, q3o, q4o, q5o, q6o)
        L.append(f"\n## Verdict: {verdict}")
        L.append("| gate | pass | value |\n|---|---|---|")
        for kk, (ok, v) in G.items():
            L.append(f"| {kk} | {'yes' if ok else 'no'} | {v} |")
        L.append("Novel = N1-N6 all pass. If N4 passes but the +GC coefficient CI includes 0 the state is "
                 "reported as treatment-associated.")
        R["gates"] = {kk: dict(pass_=v[0], value=v[1]) for kk, v in G.items()}
        R["verdict"] = verdict
    except Exception as e:  # noqa: BLE001
        L.append(f"gates failed: {e!r}")
    try:
        if q1o:
            L.append("\n## Q1 SLE - healthy flag excess (pp)")
            L.append("| model | flag | SLE | healthy | excess [CI] | raw excess | composition share |\n|---|---|---|---|---|---|---|")
            for _, r in q1o["excess"][q1o["excess"].flag != "pool"].iterrows():
                L.append(f"| {r.model} | {r.flag} | {f3(r.std_mean_sle)} | {f3(r.std_mean_healthy)} | "
                         f"{f3(r.std_excess)} [{f3(r.std_ci_lo)}, {f3(r.std_ci_hi)}] | {f3(r.raw_excess)} | "
                         f"{f3(r.composition_share)} |")
            for kk, v in [x for x in q1o.get("retained", {}).items() if x[0] != "pool"]:
                L.append(f"- retained ({kk}): {f3(v['retained'])} [{f3(v['retained_ci_lo'])}, "
                         f"{f3(v['retained_ci_hi'])}]; loss {f3(v['loss_full_minus_blind'])} "
                         f"[{f3(v['loss_ci_lo'])}, {f3(v['loss_ci_hi'])}]")
            il = q1o.get("isg_low_sle")
            if il:
                L.append(f"- ISG-low SLE donors (ISG <= healthy q{int(args.isg_low_q * 100)}, n={il['n_isg_low_sle']}"
                         f"/{il['n_sle']}, vs {il.get('n_healthy_matched')} healthy with ISG <= thr; ISG gap "
                         f"{eci(il.get('isg_gap'))}): blind {eci(il.get('blind'))}, full {eci(il.get('full'))}, "
                         f"ISG-detector {eci(il.get('isgdet'))}")
            L.append("- stratified excess: " + "; ".join(f"{kk} {eci(v)} ({v.get('n_strata_used')} strata)"
                                                         for kk, v in q1o.get("stratified", {}).items()))
            L.append("- OLS HC3 is_sle coef (pp; covariates z-scored): " + "; ".join(
                f"{kk.split('|', 2)[0]} {kk.split('|', 2)[2]} {eci(v)}" for kk, v in q1o.get("ols_is_sle", {}).items()
                if "coef" in v and kk.split("|")[1] == k))
            if "cohort_by_condition" in q1o:
                L.append(f"- donors cohort x SLE(1)/healthy(0): {q1o['cohort_by_condition']}")
            ps = q1o.get("blind_per_seed", {})
            if ps:
                L.append("- blind per seed: " + "; ".join(f"{s} {eci(v)}" for s, v in ps.items()))
    except Exception as e:  # noqa: BLE001
        L.append(f"(section Q1 summary failed: {e!r})")
    try:
        if q2o:
            L.append("\n## Q2 within-SLE Spearman, donor flag rate vs covariate")
            T = q2o["table"]
            T = T[(T.group == "SLE") & (T.flag.isin([k, "bonf"]))]
            L.append("| model | flag | ISG | GC (treatment proxy) | IEG (dissociation) |\n|---|---|---|---|---|")
            for (m, f), g in T.groupby(["model", "flag"], sort=False):
                cells = {r.covariate: f"{f3(r.rho)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]" for _, r in g.iterrows()}
                L.append(f"| {m} | {f} | {cells.get('isg', 'NA')} | {cells.get('gc', 'NA')} | {cells.get('diss', 'NA')} |")
            for kk, v in q2o.get("rho_full_minus_blind", {}).items():
                L.append(f"- rho(full) - rho(blind), {kk}: {f3(v['diff'])} [{f3(v['ci_lo'])}, {f3(v['ci_hi'])}]")
    except Exception as e:  # noqa: BLE001
        L.append(f"(section Q2 summary failed: {e!r})")
    try:
        if q3o:
            L.append(f"\n## Q3 types with blind excess (q<0.05, CI>0): {', '.join(q3o['types_with_blind_excess']) or 'none'}")
            T = q3o["table_primary"]
            P = T.pivot(index="type", columns="model", values="excess")
            Q = T.pivot(index="type", columns="model", values="q")
            lo = T.pivot(index="type", columns="model", values="ci_lo")
            hi = T.pivot(index="type", columns="model", values="ci_hi")
            L.append("| type | full excess | blind excess [CI] | q | ISG-det excess |\n|---|---|---|---|---|")
            for t in (P.sort_values("blind", ascending=False).index if "blind" in P else P.index)[:8]:
                L.append(f"| {t} | {f3(P.loc[t].get('full'))} | {f3(P.loc[t].get('blind'))} "
                         f"[{f3(lo.loc[t].get('blind'))}, {f3(hi.loc[t].get('blind'))}] | {f3(Q.loc[t].get('blind'), 3)} | "
                         f"{f3(P.loc[t].get('isgdet'))} |")
            L.append(f"(top 8 of {len(P)} types; all in Q3_types.csv)")
    except Exception as e:  # noqa: BLE001
        L.append(f"(section Q3 summary failed: {e!r})")
    try:
        if q4o:
            L.append("\n## Q4 blind-flagged vs kept, donor-paired pseudobulk (dropped genes excluded)")
            Dl = q4o.get("geneset_delta")
            for t in q4o.get("types", [])[:3]:
                r = q4o.get(t, {})
                s, h = r.get("SLE", {}), r.get("healthy", {})
                L.append(f"- **{t}** SLE donors {s.get('n_donors')}: up {s.get('n_up_nodrop', 'NA')} "
                         f"({', '.join(s.get('top_up', [])[:8])}); down ({', '.join(s.get('top_down', [])[:4])}); "
                         f"enriched {', '.join(f'{a} q={f3(q, 3)}' for a, _, q in s.get('enriched_up', [])[:3])}; "
                         f"dropped in top50 {s.get('dropped_genes_in_top50_up_incl_dropped', 'NA')}; "
                         f"healthy donors {h.get('n_donors')} up {h.get('n_up_nodrop', 'NA')}")
                if Dl is not None and len(Dl):
                    d = Dl[(Dl.type == t) & (Dl.group == "SLE")]
                    key = d[d.set.str.contains(r"ISG_typeI$|dissociation_IEG$", regex=True)]
                    top = d[~d.set.str.contains(EXCLUDE_BIO, case=False, regex=True)].sort_values("q").head(3)
                    L.append("  delta flag-kept: " + "; ".join(
                        f"{r2.set} {f3(r2['mean'])} [{f3(r2.ci_lo)}, {f3(r2.ci_hi)}]"
                        for _, r2 in pd.concat([key, top]).iterrows()))
    except Exception as e:  # noqa: BLE001
        L.append(f"(section Q4 summary failed: {e!r})")
    try:
        if q5o:
            L.append("\n## Q5 overlap full vs blind flags")
            T = q5o["overlap"]
            keep = T.comparison.str.contains(r"common seeds|reference|ISGdetector") | T.comparison.str.contains("_s0_vs_blind_s0")
            for _, r in T[(T.group == "SLE") & keep].iterrows():
                L.append(f"- {r.comparison} (SLE): Jaccard {f3(r.jaccard)}, OR {f3(r.odds_ratio, 1)} "
                         f"[{f3(r.or_ci_lo, 1)}, {f3(r.or_ci_hi, 1)}], blind-in-full {f3(r.frac_b_in_a)}")
            ex = q5o.get("donor_excess", {})
            L.append(f"- blind-only donor excess {eci(ex.get('blind_only'))}; rho with ISG (SLE) "
                     f"{eci(ex.get('blind_only_rho_isg_SLE'))}; full-only excess {eci(ex.get('full_only'))}, "
                     f"rho ISG {eci(ex.get('full_only_rho_isg_SLE'))}")
            L.append(f"- blind-only SLE cells by type: {q5o.get('blind_only_by_type_SLE')}")
            P = q5o.get("profile")
            if P is not None and len(P):
                pp = P[(P.group == "SLE") & (P["var"] != "log_counts")].pivot(index="var", columns="cat",
                                                                              values="donor_mean")
                L.append("- SLE donor-mean profile: " + "; ".join(
                    f"{v}: " + ", ".join(f"{c} {f3(pp.loc[v, c])}" for c in pp.columns) for v in pp.index))
    except Exception as e:  # noqa: BLE001
        L.append(f"(section Q5 summary failed: {e!r})")
    try:
        if q6o:
            L.append(f"\n## Q6 disease_state within SLE (units {q6o.get('status_counts_units')}, "
                     f"donors with >1 state {q6o.get('donors_multi_state')})")
            for kk, v in q6o.get("ols", {}).items():
                want = "cluster" if q6o.get("donors_multi_state", 0) > 0 else "HC3"
                if "status_terms" in v and kk.split("|")[1] == args.primary and kk.endswith(want):
                    L.append(f"- {kk} (n={v['info']['n']}): " + "; ".join(
                        f"{r.term.split('T.')[-1].rstrip(']')} {f3(r.coef)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]"
                        for _, r in v["status_terms"].iterrows()))
            for kk, v in q6o.get("paired_within_donor", {}).items():
                L.append(f"- paired within donor {kk}: {eci(v)} (n={v.get('n')})")
    except Exception as e:  # noqa: BLE001
        L.append(f"(section Q6 summary failed: {e!r})")
    L.append("\nTables/figures: Q*.csv, Q*.png, donor_table.csv, isgblind.json.")
    txt = "\n".join(L)
    if len(txt.encode()) > 8000:
        txt = txt.encode()[:7850].decode("utf-8", "ignore") + "\n... (truncated; see isgblind.json)"
    open(os.path.join(OUT, "RESULTS.md"), "w").write(txt)
    log(txt)


def main():
    gs = load_genesets()
    dropped = load_dropped()
    R["n_dropped_genes"] = len(dropped)
    keys = dict(isg=detect_key(gs, args.isg_key, [r"^ISG_typeI$", r"^isg", r"type.?i.?(ifn|interferon)"]),
                gc=detect_key(gs, args.gc_key, [r"^glucocorticoid_core$", r"glucocorticoid"]),
                diss=detect_key(gs, args.diss_key, [r"^dissociation_IEG$", r"dissoc", r"\bieg"]))
    R["keys"] = keys
    log("keys:", keys)
    res = load_all()
    if res is None:
        write_results(None, None, None, None, None, None)
        return
    D, sF, sB = res
    ex = expression(D, gs, dropped)
    if ex is not None:
        X, var, obs, E, frac_drop = ex
        D, DN, det = build_donor_table(D, obs, E, {k: ("gs_" + v if v else None) for k, v in keys.items()})
    else:
        X = var = None
        frac_drop = {}
        g = D.groupby("donor")
        DN = pd.DataFrame(dict(is_sle=g.is_sle.max(), n_cells=g.size()))
    w = D[D.is_sle == 0].y.value_counts(normalize=True)
    R["healthy_composition"] = w
    q1o = q1(D, DN, w, sB)
    q2o = q2(DN)
    q3o = q3(D, DN)
    q4o = q4(D, DN, X, var, gs, dropped, sB, sF, frac_drop, q3o) if X is not None else None
    q5o = q5(D, DN, w, sF, sB)
    q6o = q6(D, DN, w)
    try:
        wcsv(DN.reset_index(), "donor_table.csv")
    except Exception as e:  # noqa: BLE001
        log("donor table write failed", repr(e))
    try:
        write_results(q1o, q2o, q3o, q4o, q5o, q6o)
    except Exception as e:  # noqa: BLE001
        R["errors"]["RESULTS"] = repr(e)
        log(traceback.format_exc())
    R["timing_s"]["total"] = round(time.time() - T0, 1)
    save_results()


if __name__ == "__main__":
    main()
