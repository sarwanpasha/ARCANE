#!/usr/bin/env python
"""View divergence: is the disease signal specific to the label-free scVI view (U) and hidden in the label-shaped
scANVI view (S)?  Donor-level analysis for SLE (lupusfull) and COVID-19 (covid_stephenson, covid_yoshida if present).

Sections (each wrapped in try/except; failures logged in viewdiv.json["errors"]):
  A  per fine label, kNN distance in each view normalized by the healthy-query median of that label (log2 ratio);
     per donor x label median; disease vs healthy-query donors: Hodges-Lehmann shift, standardized shift (HL / robust
     SD of healthy donors), AUROC, stratified donor bootstrap CIs; paired AUROC_U - AUROC_S; composition-standardized
     donor score across labels (healthy label weights) with within-cohort/site permutation.
  B  per donor fraction of cells U-only deviant (p_U<=a & p_S>a), S-only, both; medians, HL shift, AUROC with CIs,
     paired AUROC(U-only) - AUROC(S-only), stratified AUROC with within-stratum permutation (lupus Processing_Cohort,
     COVID site), composition-standardized variant.
  C  supervised-view confidence: among U-only deviant cells of patients, scanvi_maxprob and pmax_S (fraction >= thr,
     median) vs kept cells of the same donor (raw and label-matched), donor-level paired differences with CIs.
  D  severity: COVID ordered severity_std (Spearman with CIs, Jonckheere with within-site permutation, per-level
     medians); SLE disease_state per donor x state unit (per-state medians, KW, flare - managed HL shift).
  E  ISG: donor ISG score (expression, chunked normalisation, z-scored per label vs healthy-query cells) vs U-only
     fraction (Spearman, CIs), ISG-high donors removed (> healthy-donor q90): AUROC with CIs; OLS HC3
     logit(U-only) ~ disease + ISG + stratum.
  F  seed consistency: per-seed AUROCs with CIs, between-seed Spearman of donor U-only fractions, cell-level
     Jaccard of U-only flags, sign agreement of per-label AUROC_U - AUROC_S.
  Verdict gates written to RESULTS.md.

Run from the project directory (~/arcane):
  python view_divergence.py [--root .] [--tags lupusfull,covid_stephenson,covid_yoshida] [--out summary/viewdiv]
Inputs:  <root>/res2/<tag>_seed<s>_cells.npz; <root>/data/std/<tag>.h5ad (obs metadata, raw counts for ISG);
         <root>/data/std/<tag>_donors.csv (optional); <root>/genesets.json
Outputs: <root>/<out>/ RESULTS.md, viewdiv.json, *.csv, *.png, log.txt
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
ap.add_argument("--tags", default="lupusfull,covid_stephenson,covid_yoshida")
ap.add_argument("--res_dir", default="res2")
ap.add_argument("--seeds", default=None, help="comma list; default all found")
ap.add_argument("--genesets", default="genesets.json")
ap.add_argument("--out", default="summary/viewdiv")
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--nperm", type=int, default=2000)
ap.add_argument("--min_cells", type=int, default=10, help="min cells per donor x label")
ap.add_argument("--min_donor_cells", type=int, default=50, help="min analysed cells per donor")
ap.add_argument("--min_donors_label", type=int, default=4, help="min donors per group for a per-label test")
ap.add_argument("--conf_thr", type=float, default=0.9)
ap.add_argument("--isg_high_q", type=float, default=0.9)
ap.add_argument("--isg_key", default="ISG_typeI")
ap.add_argument("--max_expr_per_donor", type=int, default=1000, help="cells per donor for ISG scoring (0=all)")
ap.add_argument("--chunk", type=int, default=40000)
ap.add_argument("--no_expr", action="store_true")
ap.add_argument("--lupus_order", default="healthy,managed,flare", help="ordered SLE states for the trend test")
args = ap.parse_args()

ROOT = args.root
OUT = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
os.makedirs(OUT, exist_ok=True)
for _f in os.listdir(OUT):
    if re.match(r"^([A-F]_.*\.(csv|png)|fig_.*\.png|RESULTS\.md|viewdiv\.json|donors_.*\.csv)$", _f):
        os.remove(os.path.join(OUT, _f))
LOGF = open(os.path.join(OUT, "log.txt"), "w")
T0 = time.time()
R = {"args": vars(args), "errors": {}, "timing_s": {}, "tags": {}}
SEV_ORDER = ["healthy", "asymptomatic", "mild", "moderate", "severe", "critical"]
SEV_ORD = {s: i for i, s in enumerate(SEV_ORDER)}
VIEWS = ["U", "S"]
CATS = ["Uonly", "Sonly", "both"]
FALLBACK_ISG = ["IFI27", "IFI44L", "IFIT1", "IFIT3", "ISG15", "MX1", "RSAD2", "OAS1", "OAS3", "SIGLEC1", "IFI6",
                "HERC5", "EPSTI1", "LY6E", "XAF1", "IFI44", "IFIT2", "OAS2", "STAT1", "IRF7"]


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
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, pd.Series):
        return clean(o.to_dict())
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict(orient="records"))
    return o


def save_results():
    with open(os.path.join(OUT, "viewdiv.json"), "w") as f:
        json.dump(clean(R), f, indent=1, default=str)


def section(name):
    def deco(fn):
        def run(tag, *a, **k):
            t = time.time()
            key = f"{tag}:{name}"
            log(f"\n=== [{time.time() - T0:7.0f}s] {key}")
            try:
                out = fn(tag, *a, **k)
            except Exception as e:  # noqa: BLE001
                R["errors"][key] = repr(e)
                log(f"!!! section {key} failed: {e!r}")
                log(traceback.format_exc())
                out = None
            R["timing_s"][key] = round(time.time() - t, 1)
            save_results()
            return out
        return run
    return deco


plt.rcParams.update({"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
                     "legend.fontsize": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200})
COL = {"healthy": "#4A7FB0", "disease": "#C8553D", "U": "#C8553D", "S": "#4A7FB0", "kept": "#888888"}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def wcsv(df, name, index=False):
    df.to_csv(os.path.join(OUT, name), index=index)


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------
def ci(bs):
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    if len(bs) < 20:
        return np.nan, np.nan
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def auroc(a, b):
    """P(a > b) + 0.5 P(tie); a = disease, b = healthy"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    r = stats.rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def hl(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.median(a[:, None] - b[None, :]))


def robust_sd(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan
    s = 1.4826 * np.median(np.abs(v - np.median(v)))
    return float(s) if s > 0 else float(np.std(v)) if np.std(v) > 0 else np.nan


def two_group(A, B, nboot=None, seed=7, hl_on=True):
    """A (n_a x k), B (n_b x k): disease and healthy donor values for k paired metrics (same donors, complete
    rows). Stratified donor bootstrap shared across metrics -> paired CIs for differences between metrics."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    A = A[:, None] if A.ndim == 1 else A
    B = B[:, None] if B.ndim == 1 else B
    k = A.shape[1]
    na, nb = A.shape[0], B.shape[0]
    out = dict(n_dis=int(na), n_h=int(nb))
    if na < 2 or nb < 2:
        return out, None, None
    rng = np.random.default_rng(seed)
    B_ = nboot or args.nboot
    ia, ib = rng.integers(0, na, (B_, na)), rng.integers(0, nb, (B_, nb))
    auc_b = np.full((B_, k), np.nan)
    hl_b = np.full((B_, k), np.nan)
    for j in range(k):
        a, b = A[:, j], B[:, j]
        out[f"auc{j}"] = auroc(a, b)
        out[f"hl{j}"] = hl(a, b) if hl_on else np.nan
        out[f"med_dis{j}"] = float(np.median(a))
        out[f"med_h{j}"] = float(np.median(b))
        out[f"sdh{j}"] = robust_sd(b)
        # vectorized bootstrap AUROC via ranks per replicate
        for t in range(B_):
            aa, bb = a[ia[t]], b[ib[t]]
            auc_b[t, j] = auroc(aa, bb)
            if hl_on:
                hl_b[t, j] = np.median(aa[:, None] - bb[None, :])
        out[f"auc{j}_lo"], out[f"auc{j}_hi"] = ci(auc_b[:, j])
        out[f"hl{j}_lo"], out[f"hl{j}_hi"] = ci(hl_b[:, j]) if hl_on else (np.nan, np.nan)
        out[f"d{j}"] = out[f"hl{j}"] / out[f"sdh{j}"] if out[f"sdh{j}"] and np.isfinite(out[f"sdh{j}"]) else np.nan
        sdb = np.array([robust_sd(b[ib[t]]) for t in range(0, B_, max(1, B_ // 400))])
        hlb_sub = hl_b[::max(1, B_ // 400), j]
        out[f"d{j}_lo"], out[f"d{j}_hi"] = ci(hlb_sub / sdb) if hl_on else (np.nan, np.nan)
        try:
            out[f"p_mwu{j}"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            out[f"p_mwu{j}"] = np.nan
    if k >= 2:
        out["auc_diff01"] = out["auc0"] - out["auc1"]
        out["auc_diff01_lo"], out["auc_diff01_hi"] = ci(auc_b[:, 0] - auc_b[:, 1])
        out["auc_diff01_pboot"] = float(min(1, 2 * min(np.mean(auc_b[:, 0] - auc_b[:, 1] <= 0),
                                                        np.mean(auc_b[:, 0] - auc_b[:, 1] >= 0))))
    return out, auc_b, hl_b


def bh(p):
    """Benjamini-Hochberg q-values; NaN p stay NaN and do not count towards m"""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    m = np.isfinite(p)
    if m.any():
        pv = p[m]
        o = np.argsort(pv)
        qq = np.minimum.accumulate((pv[o] * len(pv) / (np.arange(len(pv)) + 1))[::-1])[::-1]
        tmp = np.empty(len(pv))
        tmp[o] = np.minimum(qq, 1)
        q[m] = tmp
    return q


def strat_auc(v, g, s, min_n=1):
    num, den = 0.0, 0.0
    for st in np.unique(s):
        m = s == st
        a, b = v[m & (g == 1)], v[m & (g == 0)]
        if len(a) >= min_n and len(b) >= min_n:
            w = len(a) * len(b)
            num += w * auroc(a, b)
            den += w
    return num / den if den > 0 else np.nan


def strat_boot(V, g, s, nboot=None, seed=8):
    """Within-stratum AUROC (pairs only within cohort/site; strata lacking either group are dropped) for k paired
    metrics, with a bootstrap that resamples donors within stratum x group -> CIs for each metric and for the
    difference metric0 - metric1. Removes cohort/site confounding of the raw pooled AUROC."""
    V = np.asarray(V, float)
    V = V[:, None] if V.ndim == 1 else V
    g, s = np.asarray(g, int), np.asarray(s).astype(str)
    m = np.isfinite(V).all(1)
    V, g, s = V[m], g[m], s[m]
    keep = np.zeros(len(g), bool)
    for st in np.unique(s):
        ix = s == st
        if (g[ix] == 1).any() and (g[ix] == 0).any():
            keep |= ix
    V, g, s = V[keep], g[keep], s[keep]
    k = V.shape[1]
    out = dict(n_dis=int((g == 1).sum()), n_h=int((g == 0).sum()), n_strata=int(len(np.unique(s))))
    if out["n_dis"] < 2 or out["n_h"] < 2:
        return out
    cells = [np.where((s == st) & (g == gg))[0] for st in np.unique(s) for gg in (0, 1)]
    rng = np.random.default_rng(seed)
    nb = nboot or args.nboot
    bs = np.full((nb, k), np.nan)
    for t in range(nb):
        ix = np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])
        for j in range(k):
            bs[t, j] = strat_auc(V[ix, j], g[ix], s[ix])
    for j in range(k):
        out[f"auc{j}"] = strat_auc(V[:, j], g, s)
        out[f"auc{j}_lo"], out[f"auc{j}_hi"] = ci(bs[:, j])
    if k >= 2:
        out["auc_diff01"] = out["auc0"] - out["auc1"]
        out["auc_diff01_lo"], out["auc_diff01_hi"] = ci(bs[:, 0] - bs[:, 1])
    return out


def strat_perm(v, g, s, nperm=None, seed=0):
    """stratified AUROC (pooled within-stratum pairs) with label permutation within strata (two-sided vs 0.5)"""
    v, g, s = np.asarray(v, float), np.asarray(g, int), np.asarray(s).astype(str)
    m = np.isfinite(v)
    v, g, s = v[m], g[m], s[m]
    obs = strat_auc(v, g, s)
    if not np.isfinite(obs):
        return dict(auc_strat=np.nan, p_perm=np.nan, n_strata=0)
    rng = np.random.default_rng(seed)
    idx = [np.where(s == st)[0] for st in np.unique(s)]
    null = np.empty(nperm or args.nperm)
    for t in range(len(null)):
        gp = g.copy()
        for ix in idx:
            gp[ix] = g[rng.permutation(ix)]
        null[t] = strat_auc(v, gp, s)
    p = (1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5) - 1e-12)) / (1 + len(null))
    nst = sum(1 for ix in idx if (g[ix] == 1).any() and (g[ix] == 0).any())
    return dict(auc_strat=float(obs), p_perm=float(p), n_strata=int(nst))


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
        rx, ry = stats.rankdata(x[i]), stats.rankdata(y[i])
        if np.std(rx) > 0 and np.std(ry) > 0:
            bs.append(np.corrcoef(rx, ry)[0, 1])
    lo, hi = ci(bs)
    return dict(n=int(n), rho=float(rho), p=float(p), ci_lo=lo, ci_hi=hi)


def boot_median(v, nboot=None, seed=2):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return dict(n=int(len(v)), med=float(np.median(v)) if len(v) else np.nan, lo=np.nan, hi=np.nan, p_w=np.nan)
    rng = np.random.default_rng(seed)
    bs = np.median(v[rng.integers(0, len(v), (nboot or args.nboot, len(v)))], 1)
    lo, hi = ci(bs)
    try:
        pw = float(stats.wilcoxon(v).pvalue) if np.any(v != 0) else 1.0
    except Exception:  # noqa: BLE001
        pw = np.nan
    return dict(n=int(len(v)), med=float(np.median(v)), mean=float(np.mean(v)), lo=lo, hi=hi, p_w=pw,
                frac_pos=float((v > 0).mean()))


def jt_stat(x, g):
    lv = np.unique(g)
    s = 0.0
    for i in range(len(lv)):
        for j in range(i + 1, len(lv)):
            a, b = x[g == lv[i]], x[g == lv[j]]
            if len(a) and len(b):
                s += (b[None, :] > a[:, None]).sum() + 0.5 * (b[None, :] == a[:, None]).sum()
    return s


def jonckheere(x, g, strata=None, nperm=None, seed=5):
    """Jonckheere-Terpstra statistic (ordered integer groups g) with permutation within strata; returns
    standardized JT (z vs permutation null) and two-sided p."""
    x, g = np.asarray(x, float), np.asarray(g, int)
    m = np.isfinite(x)
    x, g = x[m], g[m]
    s = np.zeros(len(x), int) if strata is None else pd.factorize(np.asarray(strata)[m])[0]
    if len(np.unique(g)) < 2:
        return dict(n=int(len(x)), z=np.nan, p_perm=np.nan)
    obs = jt_stat(x, g)
    rng = np.random.default_rng(seed)
    idx = [np.where(s == k)[0] for k in np.unique(s)]
    null = np.empty(nperm or args.nperm)
    for t in range(len(null)):
        gp = g.copy()
        for ix in idx:
            gp[ix] = g[rng.permutation(ix)]
        null[t] = jt_stat(x, gp)
    mu, sd = null.mean(), null.std()
    p = (1 + np.sum(np.abs(null - mu) >= abs(obs - mu) - 1e-9)) / (1 + len(null))
    return dict(n=int(len(x)), z=float((obs - mu) / sd) if sd > 0 else np.nan, p_perm=float(p))


def ols_hc3(y, X, names):
    y, X = np.asarray(y, float), np.asarray(X, float)
    m = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[m], X[m]
    n, k = X.shape
    if n <= k + 2:
        return None
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (e / np.clip(1 - h, 1e-6, None)) ** 2
    V = XtX_inv @ (X.T * w) @ X @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    z = b / np.where(se > 0, se, np.nan)
    p = 2 * stats.t.sf(np.abs(z), n - k)
    return pd.DataFrame(dict(term=names, coef=b, se=se, ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se, p=p, n=n))


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


def map_severity(s):
    t = str(s).strip().lower().replace("-", "_").replace(" ", "_")
    if t in ("", "nan", "none", "na", "unknown", "<na>"):
        return "na"
    if "lps" in t:
        return "LPS"
    if re.search(r"non_?covid|noncovid|respiratory|other_?resp|ards", t):
        return "noncovid"
    if "critical" in t or re.search(r"who_?[6-8]", t):
        return "critical"
    if "severe" in t:
        return "severe"
    if "moderate" in t:
        return "moderate"
    if "mild" in t:
        return "mild"
    if "asymptom" in t:
        return "asymptomatic"
    if re.search(r"healthy|normal|control|^hc$", t):
        return "healthy"
    return "other"


def map_lupus_state(s):
    t = str(s).strip().lower()
    for k in ["flare", "managed", "treated"]:
        if k in t:
            return k
    return "na"


def open_h5ad(tag):
    h5 = os.path.join(ROOT, "data", "std", f"{tag}.h5ad")
    if not os.path.exists(h5):
        return None
    import anndata as ad
    try:
        A = ad.read_h5ad(h5, backed="r")
        backed = True
    except Exception as e:  # noqa: BLE001
        log(f"backed read failed ({e!r}); reading fully")
        A = ad.read_h5ad(h5)
        backed = False
    fmt = getattr(A.X, "format", None) or getattr(A.X, "format_str", None)
    if backed and fmt == "csc":
        log("backed X is CSC; loading into memory as CSR")
        A = ad.read_h5ad(h5)
        backed = False
    if sp.issparse(A.X) and not sp.isspmatrix_csr(A.X):
        A.X = sp.csr_matrix(A.X)
    log(f"opened {h5}: {len(A.obs_names)} cells x {len(A.var_names)} genes (backed={backed})")
    return dict(A=A, h5=h5, backed=backed, names=pd.Index(np.asarray(A.obs_names).astype(str)),
                var=pd.Index(np.asarray(A.var_names).astype(str)))


def load_cells(tag):
    rd = os.path.join(ROOT, args.res_dir)
    files = sorted(glob.glob(os.path.join(rd, f"{tag}_seed*_cells.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(ROOT, "res", f"{tag}_seed*_cells.npz")))
    want = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    base, used = None, []
    need = ["knn_dist_U", "knn_dist_S", "p_unknown_U", "p_unknown_S"]
    for f in files:
        m = re.search(r"_seed(\d+)_cells\.npz$", f)
        if not m:
            continue
        s = int(m.group(1))
        if want is not None and s not in want:
            continue
        C = np.load(f, allow_pickle=True)
        miss = [k for k in need if k not in C.files]
        if miss:
            log(f"{f}: missing {miss}; skipped")
            continue
        names = np.asarray(C["names"]).astype(str)
        classes = np.asarray(C["classes"]).astype(str) if "classes" in C.files else None
        df = pd.DataFrame(index=pd.Index(names, name="cell"))
        if base is None:
            yv = np.asarray(C["y"])
            if yv.dtype.kind in "iu" and classes is not None:
                yv = np.where(yv >= 0, classes[np.clip(yv, 0, len(classes) - 1)], "none")
            df["y"] = yv.astype(str)
            df["cond"] = np.asarray(C["cond"]).astype(str)
            df["batch"] = np.asarray(C["batch"]).astype(str)
        df[f"dU_{s}"] = np.asarray(C["knn_dist_U"], np.float32)
        df[f"dS_{s}"] = np.asarray(C["knn_dist_S"], np.float32)
        df[f"pU_{s}"] = np.asarray(C["p_unknown_U"], np.float32)
        df[f"pS_{s}"] = np.asarray(C["p_unknown_S"], np.float32)
        df[f"pmaxS_{s}"] = np.asarray(C["pmax_S"], np.float32) if "pmax_S" in C.files else np.nan
        df[f"mp_{s}"] = np.asarray(C["scanvi_maxprob"], np.float32) if "scanvi_maxprob" in C.files else np.nan
        if df.index.duplicated().any():
            df = df[~df.index.duplicated()]
        base = df if base is None else base.join(df, how="inner")
        used.append(s)
        log(f"{tag} seed {s}: {len(df)} cells")
    if base is None:
        return None, []
    return base, used


def build_meta(tag, D, H):
    """adds donor, grp (healthy/disease/excl), state, strata, ancestry columns to D (cells)"""
    info = {}
    obs = None
    if H is not None:
        obs = H["A"].obs
        obs = obs.copy()
        obs.index = H["names"]
        if obs.index.duplicated().any():  # reindex() would raise on duplicated obs_names
            info["obs_dup_names"] = int(obs.index.duplicated().sum())
            log(f"WARNING: {info['obs_dup_names']} duplicated obs_names in h5ad; keeping first occurrence")
            obs = obs[~obs.index.duplicated()]
        found = D.index.isin(obs.index)
        info["obs_matched"] = float(found.mean())
        if found.mean() < 0.9:
            log(f"WARNING: only {found.mean():.1%} npz cells found in h5ad obs; metadata from obs ignored")
            obs = None
        else:
            obs = obs.reindex(D.index)
    cols = list(obs.columns) if obs is not None else []
    det = dict(donor=find_col(cols, ["donor_id", "patient_id"], [r"^donor", r"patient"]),
               strata=find_col(cols, ["Processing_Cohort", "site_std", "Site", "site"], [r"cohort", r"^site"]),
               ancestry=find_col(cols, ["self_reported_ethnicity", "ethnicity", "ancestry"], [r"ethnic", r"ancestr"]),
               state=find_col(cols, ["disease_state"], [r"disease_state", r"sledai"]),
               sev=find_col(cols, ["severity_std", "Status_on_day_collection_summary", "severity", "Status"],
                            [r"severity", r"status_on_day"]),
               lps=find_col(cols, ["time_after_LPS"], [r"lps"]))
    info["columns"] = det
    NA_STR = {"", "nan", "none", "na", "<na>", "unknown", "not applicable"}

    def get(k, fill=None):
        """obs column as str; missing values (NaN / unmatched cells) -> fill (default: keep 'nan' string)"""
        if not det.get(k):
            return None
        raw = obs[det[k]]
        na = raw.isna().values  # NaN incl. cells unmatched in obs (astype(str) keeps NaN under pandas>=3)
        v = pd.Series(np.where(na, "nan", raw.astype(object).astype(str).values), index=raw.index).str.strip()
        if fill is not None:
            v = v.where(~v.str.lower().isin(NA_STR), fill)
        return v.values.astype(object)
    # person-level donor id: if both donor_id and patient_id exist, use the coarser one (a per-sample id would
    # turn repeat samples of one person into pseudo-replicates)
    cand = [c for c in ["donor_id", "patient_id"] if c in cols]
    if len(cand) == 2:
        nl = {c: int(obs[c].nunique(dropna=True)) for c in cand}
        det["donor"] = min(cand, key=lambda c: nl[c])
        info["donor_levels"] = nl
    if det["donor"]:
        dv = pd.Series(get("donor", fill="__NA__"), index=D.index)
        nmiss = int((dv == "__NA__").sum())
        if nmiss:
            log(f"WARNING: {nmiss} cells lack {det['donor']}; using npz batch for them")
            info["n_cells_donor_from_batch"] = nmiss
        D["donor"] = dv.where(dv != "__NA__", D["batch"]).values
        nb, nd = D.batch.nunique(), D.donor.nunique()
        info["n_batch"], info["n_donor_col"] = int(nb), int(nd)
        if nd > nb * 1.05:
            log(f"donor column {det['donor']} has more levels ({nd}) than batch ({nb}); using batch")
            D["donor"] = D["batch"].values
    else:
        D["donor"] = D["batch"].values
    D["strata"] = get("strata", fill="unknown") if det["strata"] else "all"
    D["ancestry"] = get("ancestry", fill="unknown") if det["ancestry"] else "unknown"
    cond = D.cond.str.lower()
    healthy_c = cond.str.contains(r"normal|healthy|control").values
    is_lupus = "lupus" in tag.lower() or cond.str.contains(r"lupus|sle").any()
    if is_lupus:
        info["disease"] = "SLE"
        dis = cond.str.contains(r"lupus|sle").values
        D["grp"] = np.where(healthy_c, "healthy", np.where(dis, "disease", "excl"))
        st = pd.Series(get("state"), index=D.index).map(map_lupus_state) if det["state"] else pd.Series("na", index=D.index)
        D["state"] = np.where(D.grp == "healthy", "healthy", np.where(D.grp == "disease", st.values, "excl"))
        D["sev_ord"] = np.nan
    else:
        info["disease"] = "COVID-19"
        sev = pd.Series(get("sev"), index=D.index).map(map_severity) if det["sev"] else pd.Series("na", index=D.index)
        if det["lps"]:
            lv = pd.Series(get("lps")).str.lower()
            islps = (~lv.isin(["", "nan", "none", "na", "healthy", "0", "unknown", "not applicable", "<na>"])
                     & lv.str.contains(r"\d|lps|min|hour")).values
            sev = sev.where(~islps, "LPS")
        sev = sev.where(~(sev.isin(["na", "other"]) & healthy_c), "healthy")
        covid_c = cond.str.contains(r"covid|sars").values
        sev = sev.where(~(sev.isin(["na", "other"]) & covid_c), "covid_unk")
        D["grp"] = np.where(sev == "healthy", "healthy",
                            np.where(sev.isin(SEV_ORDER[1:] + ["covid_unk"]), "disease", "excl"))
        D["state"] = sev.values
        D["sev_ord"] = sev.map(SEV_ORD).astype(float).values
        info["sev_cells"] = sev.value_counts().to_dict()
    info["grp_cells"] = D.grp.value_counts().to_dict()
    # donor csv for strata (site) if missing
    dcsv = os.path.join(ROOT, "data", "std", f"{tag}_donors.csv")
    if D.strata.isin(["all", "unknown"]).all() and os.path.exists(dcsv):
        try:
            dd = pd.read_csv(dcsv, dtype=str)
            idc = max(dd.columns, key=lambda c: dd[c].isin(D.donor.unique()).sum())
            sc = find_col(list(dd.columns), patterns=[r"site", r"cohort"])
            if sc and dd[idc].isin(D.donor.unique()).sum() > 0:
                D["strata"] = D.donor.map(dd.drop_duplicates(idc).set_index(idc)[sc]).fillna("unknown").values
                info["strata_from_csv"] = sc
        except Exception as e:  # noqa: BLE001
            log(f"donor csv read failed: {e!r}")
    return D, info


def mode(s):
    vc = pd.Series(s).value_counts()
    return vc.index[0] if len(vc) else np.nan


# ----------------------------------------------------------------------------------------------
# expression: ISG score
# ----------------------------------------------------------------------------------------------
def normalize_rows(X):
    X = sp.csr_matrix(X, dtype=np.float32)
    X.eliminate_zeros()
    samp = X.data[:200000]
    if len(samp) and not np.allclose(samp, np.round(samp)):
        return X
    tot = np.asarray(X.sum(1)).ravel()
    X.data *= np.repeat((1e4 / np.maximum(tot, 1)).astype(np.float32), np.diff(X.indptr))
    np.log1p(X.data, out=X.data)
    return X


def isg_scores(H, cells, genes, rng):
    """scanpy-like score: mean of ISG genes - mean of expression-binned control genes. Two chunked passes over raw
    counts (normalize per chunk; full matrix never densified or kept)."""
    nm = H["names"]
    if nm.is_unique:
        pos = nm.get_indexer(cells)
    else:  # duplicated obs_names: first occurrence (get_indexer raises on a non-unique index)
        first = pd.Series(np.arange(len(nm)), index=nm)
        first = first[~first.index.duplicated()]
        pos = first.reindex(cells).fillna(-1).astype(int).values
    if (pos < 0).any():
        raise KeyError(f"{(pos < 0).sum()} cells not in h5ad")
    order = np.argsort(pos)
    spos = pos[order]
    A = H["A"]
    G = len(H["var"])

    def chunks():
        for i in range(0, len(spos), args.chunk):
            yield i, normalize_rows(A.X[spos[i:i + args.chunk]])
    gsum = np.zeros(G)
    det = np.zeros(G)
    for i, X in chunks():
        gsum += np.asarray(X.sum(0)).ravel()
        det += np.bincount(X.indices, minlength=G)
    gm = gsum / len(spos)
    vpos = pd.Series(np.arange(G), index=H["var"])
    vpos = vpos[~vpos.index.duplicated()]
    gi = np.array(sorted(set(vpos.reindex([g for g in genes if g in vpos.index]).values.tolist())), int)
    ok = det >= min(20, max(3, len(spos) // 200))
    gi = gi[ok[gi]]
    if len(gi) < 3:
        raise ValueError(f"only {len(gi)} ISG genes detected")
    okidx = np.where(ok)[0]
    rk = pd.Series(gm[okidx]).rank(method="min").values
    nb = 25
    cut = np.full(G, -1)
    cut[okidx] = (rk // max(1, int(np.round(len(okidx) / (nb - 1))))).astype(int)
    ctrl = set()
    for b in np.unique(cut[gi]):
        pool = np.where(cut == b)[0]
        pool = pool[~np.isin(pool, gi)]
        rng.shuffle(pool)
        ctrl.update(pool[:50].tolist())
    ctrl = np.array(sorted(ctrl), int)
    w = np.zeros(G, np.float32)
    w[gi] = 1.0 / len(gi)
    if len(ctrl):
        w[ctrl] -= 1.0 / len(ctrl)
    sc = np.empty(len(spos), np.float32)
    for i, X in chunks():
        sc[i:i + X.shape[0]] = X @ w
        log(f"  ISG score rows {i + X.shape[0]}/{len(spos)}")
    out = np.empty_like(sc)
    out[order] = sc
    return out, [str(H["var"][i]) for i in gi]


def load_isg_genes():
    p = args.genesets if os.path.isabs(args.genesets) else os.path.join(ROOT, args.genesets)
    if os.path.exists(p):
        gs = json.load(open(p))
        if args.isg_key in gs:
            return [str(g) for g in gs[args.isg_key]]
        for k in gs:
            if re.search(r"ISG|IFN", k):
                return [str(g) for g in gs[k]]
    log("ISG set not found; using fallback list")
    return FALLBACK_ISG


# ----------------------------------------------------------------------------------------------
# per-tag analyses
# ----------------------------------------------------------------------------------------------
def cell_cats(D, s):
    a = args.alpha
    pu, ps = D[f"pU_{s}"].values, D[f"pS_{s}"].values
    ok = np.isfinite(pu) & np.isfinite(ps)  # cells with a missing p are excluded, not counted as 'kept'
    u, v = ok & (pu <= a), ok & (ps <= a)
    return {"Uonly": u & ~v, "Sonly": ~u & v & ok, "both": u & v, "anyU": u, "anyS": v, "valid": ok}


def donor_fracs(D, seeds, wts):
    """per donor (avg over seeds) fraction of cells in each category + composition-standardized U-only/S-only"""
    rows = {}
    per_seed = {}
    for s in seeds:
        cc = cell_cats(D, s)
        ok = cc.pop("valid")
        F = pd.DataFrame({k: np.where(ok, v, np.nan).astype(np.float32) for k, v in cc.items()}, index=D.index)
        F["donor"], F["y"] = D.donor.values, D.y.values
        g = F.groupby("donor")[list(cc)].mean()
        for k in ["Uonly", "Sonly"]:
            gl = F.groupby(["donor", "y"])[k].agg(["mean", "size"])
            Rt = gl["mean"].where(gl["size"] >= args.min_cells).unstack()
            W = pd.Series(wts).reindex(Rt.columns).fillna(0).values[None, :]
            val = Rt.notna().values
            num, den = np.nansum(Rt.values * W, 1), (W * val).sum(1)
            g[k + "_std"] = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=Rt.index)
        per_seed[s] = g
    keys = list(per_seed[seeds[0]].columns)
    for k in keys:
        rows[k] = pd.concat([per_seed[s][k] for s in seeds], axis=1).mean(1)
    return pd.DataFrame(rows), per_seed


def donor_table(D, seeds, wts):
    g = D.groupby("donor")
    T = pd.DataFrame(dict(n_cells=g.size(), grp=g.grp.agg(mode), strata=g.strata.agg(mode),
                          ancestry=g.ancestry.agg(mode)))
    if D.sev_ord.notna().any():
        T["sev_ord"] = g.sev_ord.max()
        T["severity"] = T.sev_ord.map(lambda v: SEV_ORDER[int(v)] if np.isfinite(v) else "unknown")
    else:
        T["severity"] = g.state.agg(lambda s: ",".join(sorted(set(s))))
    F, per_seed = donor_fracs(D, seeds, wts)
    T = T.join(F)
    T["is_dis"] = (T.grp == "disease").astype(int)
    return T, per_seed


@section("A_label_distances")
def sec_A(tag, D, T, seeds, wts):
    h = (D.grp == "healthy").values
    labs = pd.Series(D.y.values)
    per_seed_auc = {}
    DL_all = {}
    for s in seeds:
        rows = {}
        for v in VIEWS:
            d = D[f"d{v}_{s}"].values.astype(float)
            med = pd.Series(d[h]).groupby(labs[h].values).median()
            gmed = np.nanmedian(d[h])
            ref = labs.map(med).fillna(gmed).values
            r = np.log2(np.maximum(d, 1e-12) / np.maximum(ref, 1e-12))
            rows[v] = r
        X = pd.DataFrame(dict(donor=D.donor.values, y=D.y.values, rU=rows["U"], rS=rows["S"]))
        g = X.groupby(["donor", "y"])
        M = g[["rU", "rS"]].median()
        M["n"] = g.size()
        DL_all[s] = M[M.n >= args.min_cells]
    # average donor x label medians over seeds
    DL = pd.concat([DL_all[s][["rU", "rS"]] for s in seeds]).groupby(level=[0, 1]).mean()
    DL = DL.join(T[["is_dis", "grp", "strata"]], on="donor")
    DL = DL[DL.grp.isin(["healthy", "disease"])]
    out_rows = []
    for lab, g in DL.groupby(level="y"):
        a = g[g.is_dis == 1][["rU", "rS"]].dropna().values
        b = g[g.is_dis == 0][["rU", "rS"]].dropna().values
        if len(a) < args.min_donors_label or len(b) < args.min_donors_label:
            continue
        o, _, _ = two_group(a, b, nboot=max(200, args.nboot // 4), seed=11)
        out_rows.append(dict(label=lab, n_dis=o["n_dis"], n_h=o["n_h"],
                             auc_U=o["auc0"], auc_U_lo=o["auc0_lo"], auc_U_hi=o["auc0_hi"],
                             auc_S=o["auc1"], auc_S_lo=o["auc1_lo"], auc_S_hi=o["auc1_hi"],
                             auc_diff=o["auc_diff01"], auc_diff_lo=o["auc_diff01_lo"], auc_diff_hi=o["auc_diff01_hi"],
                             hl_U=o["hl0"], hl_U_lo=o["hl0_lo"], hl_U_hi=o["hl0_hi"],
                             hl_S=o["hl1"], hl_S_lo=o["hl1_lo"], hl_S_hi=o["hl1_hi"],
                             d_U=o["d0"], d_U_lo=o["d0_lo"], d_U_hi=o["d0_hi"],
                             d_S=o["d1"], d_S_lo=o["d1_lo"], d_S_hi=o["d1_hi"],
                             p_U=o["p_mwu0"], p_S=o["p_mwu1"]))
    L = pd.DataFrame(out_rows)
    if len(L):
        for c in ["p_U", "p_S"]:
            L["q" + c[1:]] = bh(L[c].values)
        L = L.sort_values("auc_diff", ascending=False)
    wcsv(L, f"A_{tag}_per_label.csv")
    # composition-standardized donor score
    W = pd.Series(wts)
    comp = {}
    for v in ["rU", "rS"]:
        P = DL[v].unstack()
        Wm = W.reindex(P.columns).fillna(0).values[None, :]
        val = P.notna().values
        num, den = np.nansum(P.values * Wm, 1), (Wm * val).sum(1)
        comp[v] = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=P.index)
    C = pd.DataFrame(comp).join(T[["is_dis", "strata", "n_cells"]])
    C = C.dropna(subset=["rU", "rS"])
    C = C[C.n_cells >= args.min_donor_cells]  # same donor set as sections B/F
    o, _, _ = two_group(C[C.is_dis == 1][["rU", "rS"]].values, C[C.is_dis == 0][["rU", "rS"]].values, seed=12)
    o["perm_U"] = strat_perm(C.rU.values, C.is_dis.values, C.strata.values, seed=13)
    o["perm_S"] = strat_perm(C.rS.values, C.is_dis.values, C.strata.values, seed=14)
    o["strat"] = strat_boot(C[["rU", "rS"]].values, C.is_dis.values, C.strata.values, seed=15)
    res = dict(overall=o, n_labels=int(len(L)),
               n_labels_aucU_gt_aucS=int((L.auc_diff > 0).sum()) if len(L) else 0,
               n_labels_diff_ci_pos=int((L.auc_diff_lo > 0).sum()) if len(L) else 0,
               n_labels_diff_ci_neg=int((L.auc_diff_hi < 0).sum()) if len(L) else 0,
               n_labels_U_sig=int((L.q_U <= 0.05).sum()) if len(L) else 0,
               n_labels_S_sig=int((L.q_S <= 0.05).sum()) if len(L) else 0,
               median_dU=float(L.d_U.median()) if len(L) else np.nan,
               median_dS=float(L.d_S.median()) if len(L) else np.nan)
    # per-seed overall AUROC (for consistency section)
    for s in seeds:
        M = DL_all[s].join(T[["is_dis", "grp"]], on="donor")
        M = M[M.grp.isin(["healthy", "disease"])]
        r = {}
        for v in ["rU", "rS"]:
            P = M[v].unstack()
            Wm = W.reindex(P.columns).fillna(0).values[None, :]
            val = P.notna().values
            num, den = np.nansum(P.values * Wm, 1), (Wm * val).sum(1)
            sc = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=P.index)
            isd = T.is_dis.reindex(sc.index).values
            r[v] = auroc(sc.values[isd == 1], sc.values[isd == 0])
        lab_diff = {}
        for lab, g in M.groupby(level="y"):
            a, b = g[g.is_dis == 1], g[g.is_dis == 0]
            if len(a) >= args.min_donors_label and len(b) >= args.min_donors_label:
                lab_diff[lab] = auroc(a.rU, b.rU) - auroc(a.rS, b.rS)
        per_seed_auc[s] = dict(auc_U=r["rU"], auc_S=r["rS"], label_diff=lab_diff)
    res["per_seed"] = per_seed_auc
    C.to_csv(os.path.join(OUT, f"A_{tag}_donor_score.csv"))
    # figure: per-label AUROC U vs S
    if len(L):
        Ls = L.sort_values("auc_U")
        fig, ax = plt.subplots(figsize=(8, max(5, 0.42 * len(Ls) + 1.5)))
        yy = np.arange(len(Ls))
        for v, off in [("U", 0.15), ("S", -0.15)]:
            ax.errorbar(Ls[f"auc_{v}"], yy + off, xerr=[Ls[f"auc_{v}"] - Ls[f"auc_{v}_lo"], Ls[f"auc_{v}_hi"] - Ls[f"auc_{v}"]],
                        fmt="o", color=COL[v], ms=7, capsize=0, lw=1.5, label="scVI (U)" if v == "U" else "scANVI (S)")
        ax.axvline(0.5, color="k", lw=1, ls="--")
        ax.set_yticks(yy)
        ax.set_yticklabels(Ls.label, fontsize=13)
        ax.set_xlabel(f"Donor AUROC, {R['tags'][tag]['disease']} vs healthy\n(kNN distance / healthy median)")
        ax.legend(frameon=False, loc="lower right")
        savefig(fig, f"A_{tag}_per_label_auroc.png")
    log(f"{tag} A overall: AUC U {o.get('auc0'):.3f} [{o.get('auc0_lo'):.3f},{o.get('auc0_hi'):.3f}] "
        f"S {o.get('auc1'):.3f}; labels U>S {res['n_labels_aucU_gt_aucS']}/{len(L)}")
    return res


@section("B_fractions")
def sec_B(tag, T):
    Tm = T[T.grp.isin(["healthy", "disease"]) & (T.n_cells >= args.min_donor_cells)]
    res = {}
    A_, B_ = Tm[Tm.is_dis == 1], Tm[Tm.is_dis == 0]
    if len(A_) < 2 or len(B_) < 2:
        raise ValueError(f"too few donors: disease {len(A_)}, healthy {len(B_)}")
    for nm, cols in [("raw", ["Uonly", "Sonly", "both"]), ("std", ["Uonly_std", "Sonly_std"])]:
        c2 = [c for c in cols if c in Tm]
        a, b = A_[c2].dropna().values, B_[c2].dropna().values
        o, _, _ = two_group(a, b, seed=21)
        for j, c in enumerate(c2):
            o[f"perm{j}"] = strat_perm(Tm[c].values, Tm.is_dis.values, Tm.strata.values, seed=22 + j)
        o["cols"] = c2
        sub = Tm.dropna(subset=c2)
        o["strat"] = strat_boot(sub[c2].values, sub.is_dis.values, sub.strata.values, seed=24)
        if nm == "raw" and Tm.ancestry.nunique() > 1:
            # cohort x ancestry strata (ancestry differs between SLE and healthy in Perez et al.)
            sa = (Tm.strata.astype(str) + "|" + Tm.ancestry.astype(str)).values
            o["perm_anc"] = {c: strat_perm(Tm[c].values, Tm.is_dis.values, sa, seed=26 + j)
                             for j, c in enumerate(c2[:2])}
            o["strat_anc"] = strat_boot(sub[c2[:2]].values, sub.is_dis.values,
                                        (sub.strata.astype(str) + "|" + sub.ancestry.astype(str)).values, seed=28)
        res[nm] = o
    # excess over healthy and ratio of U-only to S-only excess
    ex = {}
    for c in ["Uonly", "Sonly", "both"]:
        ex[c] = float(A_[c].median() - B_[c].median())
    res["median_excess"] = ex
    rng = np.random.default_rng(23)
    bs = []
    for _ in range(args.nboot):
        a, b = A_.iloc[rng.integers(0, len(A_), len(A_))], B_.iloc[rng.integers(0, len(B_), len(B_))]
        eu, es = a.Uonly.median() - b.Uonly.median(), a.Sonly.median() - b.Sonly.median()
        bs.append(eu - es)
    res["excess_U_minus_S"] = dict(est=ex["Uonly"] - ex["Sonly"], lo=ci(bs)[0], hi=ci(bs)[1])
    # per-stratum table
    rows = []
    for st, g in Tm.groupby("strata"):
        a, b = g[g.is_dis == 1], g[g.is_dis == 0]
        rows.append(dict(stratum=st, n_dis=len(a), n_h=len(b),
                         Uonly_med_dis=a.Uonly.median(), Uonly_med_h=b.Uonly.median(),
                         Sonly_med_dis=a.Sonly.median(), Sonly_med_h=b.Sonly.median(),
                         auc_Uonly=auroc(a.Uonly, b.Uonly) if len(a) and len(b) else np.nan,
                         auc_Sonly=auroc(a.Sonly, b.Sonly) if len(a) and len(b) else np.nan))
    wcsv(pd.DataFrame(rows), f"B_{tag}_by_stratum.csv")
    o = res["raw"]
    log(f"{tag} B: U-only dis {o['med_dis0']:.4f} vs h {o['med_h0']:.4f}, AUC {o['auc0']:.3f} "
        f"[{o['auc0_lo']:.3f},{o['auc0_hi']:.3f}]; S-only AUC {o['auc1']:.3f}; diff {o['auc_diff01']:.3f} "
        f"[{o['auc_diff01_lo']:.3f},{o['auc_diff01_hi']:.3f}]")
    return res


@section("C_confidence")
def sec_C(tag, D, T, seeds):
    thr = args.conf_thr
    rows = []
    hist = {"Uonly": [], "kept": [], "Sonly": []}
    rng = np.random.default_rng(31)
    for s in seeds:
        cc = cell_cats(D, s)
        kept = cc["valid"] & ~(cc["anyU"] | cc["anyS"])
        X = pd.DataFrame(dict(donor=D.donor.values, y=D.y.values, grp=D.grp.values,
                              mp=D[f"mp_{s}"].values, pm=D[f"pmaxS_{s}"].values,
                              cat=np.where(cc["Uonly"], "Uonly", np.where(cc["Sonly"], "Sonly",
                                                                          np.where(cc["both"], "both",
                                                                                   np.where(kept, "kept", "na"))))))
        X = X[X.grp.isin(["healthy", "disease"])]
        X["mp_hi"] = (X.mp >= thr).astype(float).where(X.mp.notna())
        X["pm_hi"] = (X.pm >= thr).astype(float).where(X.pm.notna())
        if s == seeds[0]:
            dd = X[X.grp == "disease"]
            for k in hist:
                v = dd[dd.cat == k].mp.dropna().values
                hist[k] = v[rng.permutation(len(v))[:200000]]
        G = X.groupby(["donor", "cat"])[["mp", "pm", "mp_hi", "pm_hi"]].agg(["mean", "median", "size"])
        GL = X.groupby(["donor", "y", "cat"])[["mp_hi", "pm_hi"]].agg(["mean", "size"])
        for d, g in X.groupby("donor"):
            grp = g.grp.iloc[0]
            r = dict(seed=s, donor=d, grp=grp)
            for cat in ["Uonly", "Sonly", "kept"]:
                if (d, cat) in G.index:
                    gg = G.loc[(d, cat)]
                    r[f"n_{cat}"] = int(gg[("mp", "size")])
                    r[f"mp_hi_{cat}"] = gg[("mp_hi", "mean")]
                    r[f"pm_hi_{cat}"] = gg[("pm_hi", "mean")]
                    r[f"mp_med_{cat}"] = gg[("mp", "median")]
                    r[f"pm_med_{cat}"] = gg[("pm", "median")]
            # label-matched kept expectation for U-only cells
            try:
                gl = GL.loc[d]
                if "Uonly" in gl.index.get_level_values("cat") and "kept" in gl.index.get_level_values("cat"):
                    u = gl.xs("Uonly", level="cat")
                    k = gl.xs("kept", level="cat")
                    k = k[k[("mp_hi", "size")] >= 5]
                    common = u.index.intersection(k.index)
                    if len(common):
                        w = u.loc[common, ("mp_hi", "size")].values.astype(float)
                        for f in ["mp_hi", "pm_hi"]:
                            r[f"{f}_kept_matched"] = float(np.average(k.loc[common, (f, "mean")].values, weights=w))
                            r[f"{f}_Uonly_matched"] = float(np.average(u.loc[common, (f, "mean")].values, weights=w))
            except KeyError:
                pass
            rows.append(r)
    P = pd.DataFrame(rows)
    num = [c for c in P.columns if c not in ("seed", "donor", "grp")]
    Pm = P.groupby(["donor", "grp"])[num].mean().reset_index()
    wcsv(Pm, f"C_{tag}_donor_confidence.csv")
    res = {}
    for grp in ["disease", "healthy"]:
        g = Pm[(Pm.grp == grp) & (Pm.get("n_Uonly", pd.Series(0, index=Pm.index)) >= 5)]
        rr = dict(n_donors=int(len(g)))
        for f in ["mp", "pm"]:
            for cat in ["Uonly", "Sonly", "kept"]:
                if f"{f}_hi_{cat}" in g:
                    rr[f"{f}_hi_{cat}"] = boot_median(g[f"{f}_hi_{cat}"].values, seed=32)
                if f"{f}_med_{cat}" in g:
                    rr[f"{f}_med_{cat}"] = float(np.nanmedian(g[f"{f}_med_{cat}"]))
            if f"{f}_hi_Uonly_matched" in g:
                rr[f"{f}_diff_matched"] = boot_median((g[f"{f}_hi_Uonly_matched"] - g[f"{f}_hi_kept_matched"]).values,
                                                      seed=33)
                rr[f"{f}_diff_raw"] = boot_median((g[f"{f}_hi_Uonly"] - g[f"{f}_hi_kept"]).values, seed=34)
        res[grp] = rr
    # cell-level pooled (descriptive, patients)
    if len(hist["Uonly"]) and len(hist["kept"]):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        bins = np.linspace(0, 1, 41)
        for k, c in [("kept", COL["kept"]), ("Uonly", COL["U"]), ("Sonly", COL["S"])]:
            if len(hist[k]):
                ax.hist(hist[k], bins=bins, density=True, histtype="step", lw=2.5, color=c,
                        label={"kept": "Kept", "Uonly": "U-only deviant", "Sonly": "S-only deviant"}[k])
        ax.axvline(thr, color="k", ls="--", lw=1)
        ax.set_xlabel("scANVI max class probability")
        ax.set_ylabel("Density (patient cells)")
        ax.legend(frameon=False, loc="upper left")
        savefig(fig, f"C_{tag}_scanvi_confidence.png")
    d = res.get("disease", {})
    if "mp_hi_Uonly" in d:
        log(f"{tag} C: patients U-only maxprob>={thr}: {d['mp_hi_Uonly']['med']:.3f}; kept "
            f"{d.get('mp_hi_kept', {}).get('med', np.nan):.3f}")
    return res


@section("D_severity")
def sec_D(tag, D, T, seeds):
    res = {}
    if R["tags"][tag]["disease"] == "COVID-19":
        Tm = T[T.grp.isin(["healthy", "disease"]) & T.sev_ord.notna() & (T.n_cells >= args.min_donor_cells)].copy()
        rows = []
        for lv, g in Tm.groupby("severity"):
            rows.append(dict(level=lv, ord=SEV_ORD.get(lv, np.nan), n=len(g),
                             **{f"{c}_{k}": v for c in ["Uonly", "Sonly"] for k, v in
                                boot_median(g[c].values, seed=41).items() if k in ("med", "lo", "hi")}))
        L = pd.DataFrame(rows).sort_values("ord")
        wcsv(L, f"D_{tag}_by_severity.csv")
        cov = Tm[Tm.is_dis == 1]
        for c in ["Uonly", "Sonly"]:
            res[f"rho_covid_{c}"] = spearman_boot(cov.sev_ord.values, cov[c].values, seed=42)
            res[f"rho_all_{c}"] = spearman_boot(Tm.sev_ord.values, Tm[c].values, seed=43)
            res[f"jt_covid_{c}"] = jonckheere(cov[c].values, cov.sev_ord.values.astype(int), cov.strata.values, seed=44)
        res["levels"] = L.to_dict(orient="records")
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        lv = [l for l in SEV_ORDER if l in set(Tm.severity)]
        for j, (c, off) in enumerate([("Uonly", -0.18), ("Sonly", 0.18)]):
            data = [100 * Tm[Tm.severity == l][c].values for l in lv]
            bp = ax.boxplot(data, positions=np.arange(len(lv)) + off, widths=0.32, patch_artist=True,
                            showfliers=False, medianprops=dict(color="k"))
            for b in bp["boxes"]:
                b.set_facecolor(COL["U" if c == "Uonly" else "S"])
                b.set_alpha(0.7)
            ax.plot([], [], "s", color=COL["U" if c == "Uonly" else "S"], label="U-only" if c == "Uonly" else "S-only")
        ax.set_xticks(np.arange(len(lv)))
        ax.set_xticklabels(lv, rotation=30, ha="right")
        ax.set_ylabel("Deviant cells per donor (%)")
        ax.legend(frameon=False)
        savefig(fig, f"D_{tag}_severity.png")
    else:
        # unit = donor x disease_state (a donor can be sampled in several states)
        frs = []
        for s in seeds:
            cc = cell_cats(D, s)
            F = pd.DataFrame(dict(Uonly=cc["Uonly"], Sonly=cc["Sonly"]), index=D.index).astype(float)
            F[~cc["valid"]] = np.nan
            F["donor"], F["state"], F["grp"] = D.donor.values, D.state.values, D.grp.values
            frs.append(F.groupby(["donor", "state", "grp"])[["Uonly", "Sonly"]].mean())
        U = pd.concat(frs).groupby(level=[0, 1, 2]).mean()
        U["n"] = D.groupby(["donor", "state", "grp"]).size().reindex(U.index)
        U = U[U.n >= args.min_donor_cells].reset_index()
        U = U[U.grp.isin(["healthy", "disease"])]
        U = U.join(T[["strata"]], on="donor")
        rows = []
        for st, g in U.groupby("state"):
            rows.append(dict(state=st, n_units=len(g), n_donors=g.donor.nunique(),
                             **{f"{c}_{k}": v for c in ["Uonly", "Sonly"] for k, v in
                                boot_median(g[c].values, seed=45).items() if k in ("med", "lo", "hi")}))
        L = pd.DataFrame(rows)
        wcsv(L, f"D_{tag}_by_state.csv")
        res["states"] = L.to_dict(orient="records")
        # unpaired tests need independent units: donors sampled in several states contribute ONE unit, chosen at
        # random (seeded; not by state, which would bias flare vs managed). Within-donor change -> paired test below.
        Ud = U.sample(frac=1.0, random_state=44).drop_duplicates("donor").sort_index()
        res["n_units_dedup"] = int(len(Ud))
        sle = Ud[Ud.grp == "disease"]
        sle = sle[sle.state != "na"]
        for c in ["Uonly", "Sonly"]:
            gl = [g[c].values for _, g in sle.groupby("state") if len(g) >= 3]
            res[f"kw_{c}"] = float(stats.kruskal(*gl).pvalue) if len(gl) >= 2 else np.nan
            a, b = sle[sle.state == "flare"][c].values, sle[sle.state == "managed"][c].values
            if len(a) >= 3 and len(b) >= 3:
                o, _, _ = two_group(a[:, None], b[:, None], seed=46)
                res[f"flare_vs_managed_{c}"] = dict(hl=o["hl0"], lo=o["hl0_lo"], hi=o["hl0_hi"], auc=o["auc0"],
                                                    auc_lo=o["auc0_lo"], auc_hi=o["auc0_hi"], n_flare=len(a),
                                                    n_managed=len(b))
            order = [x.strip() for x in args.lupus_order.split(",")]
            Uo = Ud[Ud.state.isin(order)]
            res[f"jt_{c}"] = jonckheere(Uo[c].values, Uo.state.map({k: i for i, k in enumerate(order)}).values,
                                        Uo.strata.values, seed=47)
            res[f"rho_order_{c}"] = spearman_boot(Uo.state.map({k: i for i, k in enumerate(order)}).values,
                                                  Uo[c].values, seed=48)
        # paired within-donor flare vs other state
        pw = U[U.grp == "disease"].pivot_table(index="donor", columns="state", values="Uonly")
        if "flare" in pw and pw.shape[1] > 1:
            oth = pw.drop(columns=["flare"]).mean(1)
            dlt = (pw["flare"] - oth).dropna()
            res["paired_flare_minus_other_Uonly"] = boot_median(dlt.values, seed=49)
        res["n_units"] = int(len(U))
        res["n_donors_multi_state"] = int((U[U.grp == "disease"].groupby("donor").size() > 1).sum())
        fig, ax = plt.subplots(figsize=(8, 5.5))
        lv = [l for l in ["healthy", "treated", "managed", "flare", "na"] if l in set(U.state)]
        for c, off in [("Uonly", -0.18), ("Sonly", 0.18)]:
            data = [100 * U[U.state == l][c].values for l in lv]
            bp = ax.boxplot(data, positions=np.arange(len(lv)) + off, widths=0.32, patch_artist=True,
                            showfliers=False, medianprops=dict(color="k"))
            for b in bp["boxes"]:
                b.set_facecolor(COL["U" if c == "Uonly" else "S"])
                b.set_alpha(0.7)
            ax.plot([], [], "s", color=COL["U" if c == "Uonly" else "S"], label="U-only" if c == "Uonly" else "S-only")
        ax.set_xticks(np.arange(len(lv)))
        ax.set_xticklabels(lv)
        ax.set_ylabel("Deviant cells per donor (%)")
        ax.legend(frameon=False)
        savefig(fig, f"D_{tag}_state.png")
    return res


@section("E_isg")
def sec_E(tag, D, T, H):
    if H is None or args.no_expr:
        return dict(skipped="no h5ad / --no_expr")
    rng = np.random.default_rng(51)
    Dm = D[D.grp.isin(["healthy", "disease"])]
    inh = Dm.index.isin(H["names"])
    if not inh.all():
        log(f"WARNING: {int((~inh).sum())} cells not in h5ad; excluded from ISG scoring")
        Dm = Dm[inh]
    keep = []
    for _, g in Dm.groupby("donor"):
        idx = g.index.values
        n = len(idx) if args.max_expr_per_donor <= 0 else min(len(idx), args.max_expr_per_donor)
        keep.append(rng.choice(idx, n, replace=False))
    cells = np.concatenate(keep)
    log(f"ISG scoring {len(cells)} cells")
    sc, used = isg_scores(H, cells, load_isg_genes(), rng)
    E = pd.DataFrame(dict(isg=sc, y=D.loc[cells, "y"].values, donor=D.loc[cells, "donor"].values,
                          h=(D.loc[cells, "grp"] == "healthy").values), index=cells)
    z = np.full(len(E), np.nan)
    hv = E[E.h]
    gmu, gsd = hv.isg.mean(), hv.isg.std()
    for t, g in E.groupby("y"):
        hh = g[g.h].isg
        mu, sd = (hh.mean(), hh.std()) if len(hh) >= 20 else (gmu, gsd)
        z[E.index.get_indexer(g.index)] = (g.isg - mu) / (sd if sd > 0 else 1)
    E["z"] = z
    T["isg"] = E.groupby("donor").z.mean().reindex(T.index)
    Tm = T[T.grp.isin(["healthy", "disease"]) & (T.n_cells >= args.min_donor_cells) & T.isg.notna()]
    res = dict(n_isg_genes=len(used))
    dis, hl_ = Tm[Tm.is_dis == 1], Tm[Tm.is_dis == 0]
    res["auc_isg"] = auroc(dis.isg, hl_.isg)
    for c in ["Uonly", "Sonly", "both"]:
        res[f"rho_dis_{c}"] = spearman_boot(dis.isg.values, dis[c].values, seed=52)
        res[f"rho_h_{c}"] = spearman_boot(hl_.isg.values, hl_[c].values, seed=53)
    thr = float(np.quantile(hl_.isg, args.isg_high_q))
    res["isg_high_thr"] = thr
    low = dis[dis.isg <= thr]
    res["n_dis_isg_low"], res["n_dis_isg_high"] = int(len(low)), int((dis.isg > thr).sum())
    if len(low) >= 3:
        o, _, _ = two_group(low[["Uonly", "Sonly"]].values, hl_[["Uonly", "Sonly"]].values, seed=54)
        sub = pd.concat([low, hl_])
        o["perm_Uonly"] = strat_perm(sub.Uonly.values, sub.is_dis.values, sub.strata.values, seed=55)
        # residual ISG imbalance: 'ISG <= healthy q90' patients still sit in the upper healthy range, and U-only
        # tracks ISG, so AUROC(U-only) > 0.5 here can be pure ISG. Report it so it can be judged.
        o["auc_isg_balance"] = auroc(low.isg, hl_.isg)
        o["strat"] = strat_boot(sub[["Uonly", "Sonly"]].values, sub.is_dis.values, sub.strata.values, seed=56)
        res["isg_low_vs_h"] = o
    # cross-fitted ISG-low test: threshold from one random half of healthy donors (split within stratum), test
    # against the other half -> the healthy donors that define 'ISG-low' are never the comparison group.
    hidx = hl_.index.values
    if len(hidx) >= 8:
        rs = np.random.default_rng(57)
        cf = []
        for _ in range(200):
            fa = []
            for _, gg in hl_.groupby("strata"):
                ii = rs.permutation(gg.index.values)
                fa.extend(ii[: len(ii) // 2] if rs.random() < 0.5 else ii[: (len(ii) + 1) // 2])
            fa = np.asarray(fa)
            hb = hl_.loc[~hl_.index.isin(fa)]
            if len(fa) < 3 or len(hb) < 3:
                continue
            t_ = np.quantile(hl_.loc[fa, "isg"], args.isg_high_q)
            lo_ = dis[dis.isg <= t_]
            if len(lo_) < 3:
                continue
            cf.append((auroc(lo_.Uonly, hb.Uonly), auroc(lo_.Sonly, hb.Sonly), auroc(lo_.isg, hb.isg), len(lo_)))
        if cf:
            cf = np.asarray(cf)
            res["isg_low_crossfit"] = dict(n_splits=int(len(cf)), auc_U_med=float(np.median(cf[:, 0])),
                                           auc_U_q025=float(np.quantile(cf[:, 0], .025)),
                                           auc_U_q975=float(np.quantile(cf[:, 0], .975)),
                                           auc_S_med=float(np.median(cf[:, 1])), auc_isg_med=float(np.median(cf[:, 2])),
                                           frac_splits_U_gt_S=float(np.mean(cf[:, 0] > cf[:, 1])),
                                           n_low_med=float(np.median(cf[:, 3])))
    # OLS (HC3): logit U-only ~ disease + ISG (+ strata dummies)
    eps = 0.5 / max(1, Tm.n_cells.median())
    for c in ["Uonly", "Sonly"]:
        yv = np.log((Tm[c] + eps) / (1 - Tm[c] + eps)).values
        st = pd.get_dummies(Tm.strata.astype(str), drop_first=True).astype(float)
        st = st.loc[:, st.sum() >= 3]
        an = pd.get_dummies(Tm.ancestry.astype(str), drop_first=True).astype(float)
        an = an.loc[:, an.sum() >= 3]
        Xm = np.column_stack([np.ones(len(Tm)), Tm.is_dis.values, Tm.isg.values, st.values, an.values])
        names = (["Intercept", "disease", "ISG"] + [f"strata[{x}]" for x in st.columns]
                 + [f"ancestry[{x}]" for x in an.columns])
        tb = ols_hc3(yv, Xm, names)
        if tb is not None:
            res[f"ols_{c}"] = tb.set_index("term").loc[["disease", "ISG"]].reset_index().to_dict(orient="records")
            wcsv(tb, f"E_{tag}_ols_{c}.csv")
    T[["isg"]].to_csv(os.path.join(OUT, f"E_{tag}_donor_isg.csv"))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for g, cl, lab in [(hl_, COL["healthy"], "Healthy"), (dis, COL["disease"], R["tags"][tag]["disease"])]:
        ax.scatter(g.isg, 100 * g.Uonly, s=30, color=cl, alpha=0.75, label=lab, edgecolor="none")
    ax.axvline(thr, color="k", ls="--", lw=1)
    ax.set_xlabel("Donor ISG score (z vs healthy)")
    ax.set_ylabel("U-only deviant cells (%)")
    ax.legend(frameon=False)
    savefig(fig, f"E_{tag}_isg_vs_Uonly.png")
    r = res["rho_dis_Uonly"]
    log(f"{tag} E: rho(ISG, U-only) in patients {r['rho']:.2f} [{r['ci_lo']:.2f},{r['ci_hi']:.2f}]; "
        f"ISG-low patients {res['n_dis_isg_low']}")
    return res


@section("F_seeds")
def sec_F(tag, D, T, per_seed, seeds, A):
    res = dict(seeds=seeds)
    Tm = T[T.grp.isin(["healthy", "disease"]) & (T.n_cells >= args.min_donor_cells)]
    rows = []
    for s in seeds:
        g = per_seed[s].reindex(Tm.index)
        o, _, _ = two_group(g[Tm.is_dis == 1][["Uonly", "Sonly"]].values, g[Tm.is_dis == 0][["Uonly", "Sonly"]].values,
                            nboot=max(200, args.nboot // 4), seed=61, hl_on=False)
        r = dict(seed=s, auc_Uonly=o["auc0"], auc_Uonly_lo=o["auc0_lo"], auc_Uonly_hi=o["auc0_hi"],
                 auc_Sonly=o["auc1"], auc_Sonly_lo=o["auc1_lo"], auc_Sonly_hi=o["auc1_hi"],
                 diff=o["auc_diff01"], diff_lo=o["auc_diff01_lo"], diff_hi=o["auc_diff01_hi"],
                 med_Uonly_dis=float(g[Tm.is_dis == 1].Uonly.median()), med_Uonly_h=float(g[Tm.is_dis == 0].Uonly.median()))
        if A and "per_seed" in A and s in A["per_seed"]:
            r["auc_distU"], r["auc_distS"] = A["per_seed"][s]["auc_U"], A["per_seed"][s]["auc_S"]
        rows.append(r)
    S = pd.DataFrame(rows)
    wcsv(S, f"F_{tag}_per_seed.csv")
    res["per_seed"] = S.to_dict(orient="records")
    rhos, jac = [], []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = per_seed[seeds[i]].reindex(Tm.index).Uonly, per_seed[seeds[j]].reindex(Tm.index).Uonly
            rhos.append(stats.spearmanr(a, b, nan_policy="omit")[0])
            ca, cb = cell_cats(D, seeds[i])["Uonly"], cell_cats(D, seeds[j])["Uonly"]
            jac.append(float((ca & cb).sum() / max(1, (ca | cb).sum())))
    res["rho_donor_Uonly_between_seeds"] = [float(x) for x in rhos]
    res["jaccard_cell_Uonly_between_seeds"] = jac
    if A and "per_seed" in A and len(seeds) > 1:
        LD = pd.DataFrame({s: A["per_seed"][s]["label_diff"] for s in seeds})
        LD = LD.dropna()
        if len(LD):
            sg = np.sign(LD.values)
            res["label_sign_agree"] = float(np.mean((sg == sg[:, [0]]).all(1)))
            res["label_diff_pos_all_seeds"] = int((LD > 0).all(1).sum())
            res["n_labels_seed"] = int(len(LD))
    return res


@section("load")
def load_tag(tag):
    D, seeds = load_cells(tag)
    if D is None:
        return None
    H = None
    try:
        H = open_h5ad(tag)
    except Exception as e:  # noqa: BLE001
        log(f"h5ad open failed: {e!r}")
    D, info = build_meta(tag, D, H)
    n0 = len(D)
    D = D[D.grp != "excl"].copy()
    info["n_cells_npz"], info["n_cells_used"] = int(n0), int(len(D))
    hy = D[D.grp == "healthy"].y.value_counts(normalize=True)
    wts = hy.to_dict()
    T, per_seed = donor_table(D, seeds, wts)
    info["seeds"] = seeds
    info["n_donors"] = T.grp.value_counts().to_dict()
    info["n_donors_used"] = T[T.n_cells >= args.min_donor_cells].grp.value_counts().to_dict()
    info["strata_levels"] = T.groupby(["strata", "grp"]).size().unstack(fill_value=0).to_dict(orient="index")
    R["tags"][tag] = info
    log(f"{tag}: {info}")
    return D, T, per_seed, seeds, wts, H


def fig_main(TT):
    tags = [t for t in TT if TT[t] is not None]
    if not tags:
        return
    fig, axes = plt.subplots(1, len(tags), figsize=(6.3 * len(tags), 6), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        T = TT[tag]
        T = T[T.grp.isin(["healthy", "disease"]) & (T.n_cells >= args.min_donor_cells)]
        dis = R["tags"][tag]["disease"]
        for g, cl, lab in [("healthy", COL["healthy"], "Healthy"), ("disease", COL["disease"], dis)]:
            d = T[T.grp == g]
            ax.scatter(100 * d.Sonly, 100 * d.Uonly, s=34, color=cl, alpha=0.75, edgecolor="none",
                       label=f"{lab} (n={len(d)})")
        m = 100 * max(T.Sonly.max(), T.Uonly.max()) * 1.05
        ax.plot([0, m], [0, m], color="k", lw=1, ls="--")
        ax.set_xlim(0, m)
        ax.set_ylim(0, m)
        ax.set_xlabel("S-only deviant cells (%)")
        ax.set_ylabel("U-only deviant cells (%)")
        ax.legend(frameon=False, loc="upper left")
        ax.text(0.98, 0.04, {"SLE": "SLE", "COVID-19": "COVID-19"}.get(dis, dis) + f"\n{tag}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=15, color="#444444")
    savefig(fig, "fig_view_divergence.png")


# ----------------------------------------------------------------------------------------------
# verdict + RESULTS.md
# ----------------------------------------------------------------------------------------------
def f3(x, d=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{d}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{d}f}"


def eci(e, lo, hi, d=3, pct=False):
    k = 100 if pct else 1
    try:
        return f"{f3(k * e, d if not pct else 1)} [{f3(k * lo, d if not pct else 1)}, {f3(k * hi, d if not pct else 1)}]"
    except TypeError:
        return "NA"


def gates(res):
    G = {}
    for tag, r in res.items():
        B, A, E, F = r.get("B"), r.get("A"), r.get("E"), r.get("F")
        g = {}
        if B:
            o = B["raw"]
            g["G1_Uonly_separates"] = bool(o.get("auc0_lo", 0) > 0.5)
            g["G2_Uonly_beats_Sonly"] = bool(o.get("auc_diff01_lo", -1) > 0)
            g["G3_within_strata"] = bool(o.get("perm0", {}).get("p_perm", 1) < 0.05)
            g["G4_Sonly_weak"] = bool(not (o.get("auc1_lo", 0) > 0.6))
            so = o.get("strat", {})
            g["G1s_Uonly_within_strata"] = bool((so.get("auc0_lo") or 0) > 0.5)
            g["G2s_U_beats_S_within_strata"] = bool((so.get("auc_diff01_lo") or -1) > 0)
        if A:
            o = A["overall"]
            g["G5_dist_U_beats_S"] = bool(o.get("auc_diff01_lo", -1) > 0)
            g["G6_labels_majority"] = bool(A["n_labels"] > 0 and A["n_labels_aucU_gt_aucS"] > A["n_labels"] / 2)
        if E and "isg_low_vs_h" in E:
            cf = E.get("isg_low_crossfit", {})
            g["G7_ISG_low_persists"] = bool(E["isg_low_vs_h"].get("auc0_lo", 0) > 0.5
                                            and (cf.get("auc_U_med") or 0) > 0.5)
        if E and E.get("ols_Uonly"):
            g["G7b_beyond_ISG_ols"] = bool((E["ols_Uonly"][0].get("ci_lo") or 0) > 0)
        if F and F.get("per_seed"):
            ps = F["per_seed"]
            g["G8_seeds"] = bool(all((x.get("auc_Uonly_lo") or 0) > 0.5 and (x.get("diff_lo") or -1) > 0 for x in ps))
        G[tag] = g
    core = ["G1_Uonly_separates", "G1s_Uonly_within_strata", "G2_Uonly_beats_Sonly", "G2s_U_beats_S_within_strata",
            "G5_dist_U_beats_S"]
    per = {t: all(g.get(k, False) for k in core) for t, g in G.items()}
    n_ok = sum(per.values())
    if len(per) >= 1 and n_ok == len(per):
        v = (f"YES: in all {len(per)} dataset(s) ({', '.join(per)}) the disease signal is concentrated in the label-free "
             f"scVI view (core gates G1, G1s, G2, G2s, G5 pass)." + (" Single dataset only." if len(per) == 1 else ""))
    elif n_ok >= 1:
        v = f"PARTIAL: core gates pass in {n_ok}/{len(per)} datasets ({', '.join(t for t, ok in per.items() if ok)})."
    else:
        v = "NO: the disease signal is not specific to the label-free view (core gates fail)."
    return G, v


def write_md(res, G, verdict):
    L = ["# View divergence: label-free scVI (U) vs label-shaped scANVI (S)", ""]
    L.append(f"alpha={args.alpha}; donors with >= {args.min_donor_cells} cells; CIs = 95% donor bootstrap stratified by group "
             f"({args.nboot}; 'within-stratum' rows: by stratum x group); perm = within-cohort/site label permutation "
             f"({args.nperm}). U-only: p_U<=a & p_S>a.")
    L.append("")
    L.append(f"**Verdict: {verdict}**")
    L.append("")
    for tag, r in res.items():
        info = R["tags"].get(tag, {})
        dis = info.get("disease", "?")
        L.append(f"## {tag} ({dis}); seeds {info.get('seeds')}; donors {info.get('n_donors_used')}")
        B = r.get("B")
        if B:
            o = B["raw"]
            L.append(f"- U-only % cells: {dis} {f3(100 * o['med_dis0'], 2)} vs healthy {f3(100 * o['med_h0'], 2)} (median); "
                     f"AUROC {eci(o['auc0'], o['auc0_lo'], o['auc0_hi'])}; strat AUROC {f3(o['perm0']['auc_strat'])}, "
                     f"perm p {f3(o['perm0']['p_perm'])} ({o['perm0']['n_strata']} strata)")
            L.append(f"- S-only % cells: {f3(100 * o['med_dis1'], 2)} vs {f3(100 * o['med_h1'], 2)}; AUROC "
                     f"{eci(o['auc1'], o['auc1_lo'], o['auc1_hi'])}; perm p {f3(o['perm1']['p_perm'])}")
            L.append(f"- both: {f3(100 * o['med_dis2'], 2)} vs {f3(100 * o['med_h2'], 2)}; AUROC "
                     f"{eci(o['auc2'], o['auc2_lo'], o['auc2_hi'])}")
            L.append(f"- AUROC(U-only) - AUROC(S-only): {eci(o['auc_diff01'], o['auc_diff01_lo'], o['auc_diff01_hi'])}; "
                     f"median-excess U-only - S-only: {eci(B['excess_U_minus_S']['est'], B['excess_U_minus_S']['lo'], B['excess_U_minus_S']['hi'], pct=True)} pp")
            so = o.get("strat", {})
            if "auc0" in so:
                L.append(f"- within-stratum ({so['n_strata']} strata, {so['n_dis']} vs {so['n_h']} donors): AUROC U-only "
                         f"{eci(so['auc0'], so['auc0_lo'], so['auc0_hi'])}, S-only {eci(so['auc1'], so['auc1_lo'], so['auc1_hi'])}, "
                         f"diff {eci(so['auc_diff01'], so['auc_diff01_lo'], so['auc_diff01_hi'])}")
            sa, pa = o.get("strat_anc", {}), o.get("perm_anc", {})
            if "auc0" in sa:
                L.append(f"- stratum x ancestry ({sa['n_strata']} strata, {sa['n_dis']} vs {sa['n_h']}): AUROC U-only "
                         f"{eci(sa['auc0'], sa['auc0_lo'], sa['auc0_hi'])}, diff U-S "
                         f"{eci(sa.get('auc_diff01'), sa.get('auc_diff01_lo'), sa.get('auc_diff01_hi'))}; perm p U-only "
                         f"{f3(pa.get('Uonly', {}).get('p_perm'))}")
            s = B.get("std", {})
            if "auc0" in s:
                L.append(f"- composition-standardized: AUROC U-only {eci(s['auc0'], s['auc0_lo'], s['auc0_hi'])}, "
                         f"S-only {eci(s['auc1'], s['auc1_lo'], s['auc1_hi'])}")
        A = r.get("A")
        if A:
            o = A["overall"]
            L.append(f"- kNN distance / healthy-label median (donor score, composition-std): AUROC U "
                     f"{eci(o['auc0'], o['auc0_lo'], o['auc0_hi'])} (d={f3(o['d0'], 2)}), S "
                     f"{eci(o['auc1'], o['auc1_lo'], o['auc1_hi'])} (d={f3(o['d1'], 2)}); diff "
                     f"{eci(o['auc_diff01'], o['auc_diff01_lo'], o['auc_diff01_hi'])}; perm p U {f3(o['perm_U']['p_perm'])}, "
                     f"S {f3(o['perm_S']['p_perm'])}")
            L.append(f"- per label ({A['n_labels']}): AUROC_U > AUROC_S in {A['n_labels_aucU_gt_aucS']}, diff CI>0 in "
                     f"{A['n_labels_diff_ci_pos']}, CI<0 in {A['n_labels_diff_ci_neg']}; BH q<=.05: U {A['n_labels_U_sig']}, "
                     f"S {A['n_labels_S_sig']}; median standardized shift d_U {f3(A['median_dU'], 2)}, d_S {f3(A['median_dS'], 2)}")
        C = r.get("C")
        if C and "disease" in C:
            c = C["disease"]
            parts = []
            for f, nm in [("mp", "scANVI maxprob"), ("pm", "pmax_S")]:
                if f"{f}_hi_Uonly" in c and "med" in c[f"{f}_hi_Uonly"]:
                    k = c.get(f"{f}_hi_kept", {})
                    dm = c.get(f"{f}_diff_matched", {})
                    parts.append(f"{nm}>={args.conf_thr}: U-only {eci(c[f'{f}_hi_Uonly']['med'], c[f'{f}_hi_Uonly']['lo'], c[f'{f}_hi_Uonly']['hi'], 2)} "
                                 f"vs kept {f3(k.get('med'), 2)}; label-matched diff {eci(dm.get('med'), dm.get('lo'), dm.get('hi'), 2)}")
            L.append(f"- patients' U-only cells (donor medians, n={c['n_donors']}): " + "; ".join(parts))
        Dd = r.get("D")
        if Dd:
            if "rho_covid_Uonly" in Dd:
                a, b = Dd["rho_covid_Uonly"], Dd["rho_covid_Sonly"]
                L.append(f"- severity (COVID donors): rho U-only {eci(a['rho'], a['ci_lo'], a['ci_hi'], 2)}, S-only "
                         f"{eci(b['rho'], b['ci_lo'], b['ci_hi'], 2)}; JT within site z {f3(Dd['jt_covid_Uonly']['z'], 2)} "
                         f"p {f3(Dd['jt_covid_Uonly']['p_perm'])} (S-only p {f3(Dd['jt_covid_Sonly']['p_perm'])})")
                lv = "; ".join(f"{x['level']} {f3(100 * x['Uonly_med'], 1)}" for x in Dd.get("levels", []))
                L.append(f"  - U-only % by severity: {lv}")
            else:
                st = "; ".join(f"{x['state']} {f3(100 * x['Uonly_med'], 1)} [{f3(100 * x['Uonly_lo'], 1)},{f3(100 * x['Uonly_hi'], 1)}] (n={x['n_units']})"
                               for x in Dd.get("states", []))
                L.append(f"- disease_state U-only %: {st}")
                fm = Dd.get("flare_vs_managed_Uonly")
                jt = Dd.get("jt_Uonly", {})
                L.append(f"  - KW within SLE p {f3(Dd.get('kw_Uonly'))}; flare - managed HL "
                         + (f"{eci(fm['hl'], fm['lo'], fm['hi'], pct=True)} pp" if fm else "NA")
                         + f"; JT ({args.lupus_order}) within cohort p {f3(jt.get('p_perm'))}; "
                         f"multi-state donors {Dd.get('n_donors_multi_state')}")
        E = r.get("E")
        if E and "rho_dis_Uonly" in E:
            a, b = E["rho_dis_Uonly"], E["rho_dis_Sonly"]
            L.append(f"- ISG ({E['n_isg_genes']} genes; ISG AUROC {f3(E['auc_isg'])}): rho with U-only in patients "
                     f"{eci(a['rho'], a['ci_lo'], a['ci_hi'], 2)}, S-only {eci(b['rho'], b['ci_lo'], b['ci_hi'], 2)}")
            if "isg_low_vs_h" in E:
                o = E["isg_low_vs_h"]
                L.append(f"  - ISG-low patients (n={E['n_dis_isg_low']}, ISG-high removed n={E['n_dis_isg_high']}) vs healthy: "
                         f"AUROC U-only {eci(o['auc0'], o['auc0_lo'], o['auc0_hi'])}, S-only {eci(o['auc1'], o['auc1_lo'], o['auc1_hi'])}, "
                         f"diff {eci(o['auc_diff01'], o['auc_diff01_lo'], o['auc_diff01_hi'])}, perm p {f3(o['perm_Uonly']['p_perm'])}; "
                         f"residual ISG AUROC {f3(o.get('auc_isg_balance'), 2)}")
            cf = E.get("isg_low_crossfit")
            if cf:
                L.append(f"  - cross-fitted ISG-low (threshold from other healthy half, {cf['n_splits']} splits): AUROC U-only "
                         f"median {f3(cf['auc_U_med'], 2)} [{f3(cf['auc_U_q025'], 2)}, {f3(cf['auc_U_q975'], 2)}] (split range), "
                         f"S-only {f3(cf['auc_S_med'], 2)}, residual ISG {f3(cf['auc_isg_med'], 2)}")
            else:
                L.append(f"  - ISG-low patients: n={E.get('n_dis_isg_low')} (< 3, not tested; ISG-high n={E.get('n_dis_isg_high')})")
            for c in ["Uonly", "Sonly"]:
                ol = E.get(f"ols_{c}")
                if ol:
                    d0 = ol[0]
                    L.append(f"  - OLS logit {c} ~ disease + ISG + stratum + ancestry: disease {eci(d0['coef'], d0['ci_lo'], d0['ci_hi'], 2)} p {f3(d0['p'])}")
        F = r.get("F")
        if F and F.get("per_seed"):
            ps = "; ".join(f"s{x['seed']} {eci(x['auc_Uonly'], x['auc_Uonly_lo'], x['auc_Uonly_hi'], 2)} vs S {f3(x['auc_Sonly'], 2)}"
                           for x in F["per_seed"])
            L.append(f"- seeds AUROC U-only: {ps}; donor rho between seeds "
                     f"{', '.join(f3(x, 2) for x in F['rho_donor_Uonly_between_seeds'])}; cell Jaccard "
                     f"{', '.join(f3(x, 2) for x in F['jaccard_cell_Uonly_between_seeds'])}"
                     + (f"; label diff sign agrees {f3(F.get('label_sign_agree'), 2)}" if "label_sign_agree" in F else ""))
        g = G.get(tag, {})
        L.append("- gates: " + ", ".join(f"{k}={'PASS' if v else 'fail'}" for k, v in g.items()))
        L.append("")
    L.append("Gates: G1 AUROC(U-only) CI>0.5; G2 AUROC(U-only)-AUROC(S-only) CI>0; G3 within-stratum perm p<.05; "
             "G4 S-only AUROC CI not >0.6; G5 distance-score AUROC_U-AUROC_S CI>0; G6 majority of labels U>S; "
             "G1s/G2s = G1/G2 with within-stratum AUROC and stratum x group bootstrap; "
             "G7 ISG-low patients still separate (U-only CI>0.5 and cross-fitted median AUROC>0.5); G7b OLS disease "
             "coef on logit U-only CI>0 (adj. ISG, stratum, ancestry); G8 every seed G1&G2. Core = G1,G1s,G2,G2s,G5.")
    L.append("Files: fig_view_divergence.png (U-only vs S-only per donor), A_* per-label, B_* strata, C_* confidence, "
             "D_* severity, E_* ISG, F_* seeds, donors_<tag>.csv.")
    if R["errors"]:
        L.append("")
        L.append("Errors: " + "; ".join(f"{k}: {v[:120]}" for k, v in R["errors"].items()))
    txt = "\n".join(L) + "\n"
    if len(txt.encode()) > 8000:
        txt = txt[:7900] + "\n...(truncated)\n"
    open(os.path.join(OUT, "RESULTS.md"), "w").write(txt)


def main():
    res, TT = {}, {}
    for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
        out = load_tag(tag)
        if out is None:
            log(f"{tag}: no cells npz with view fields; skipped")
            continue
        D, T, per_seed, seeds, wts, H = out
        r = {}
        r["A"] = sec_A(tag, D, T, seeds, wts)
        r["B"] = sec_B(tag, T)
        r["C"] = sec_C(tag, D, T, seeds)
        r["D"] = sec_D(tag, D, T, seeds)
        r["E"] = sec_E(tag, D, T, H)
        r["F"] = sec_F(tag, D, T, per_seed, seeds, r["A"])
        T.to_csv(os.path.join(OUT, f"donors_{tag}.csv"))
        TT[tag] = T
        res[tag] = r
        R["tags"][tag]["results"] = r
        save_results()
        del D, H
    try:
        fig_main(TT)
    except Exception as e:  # noqa: BLE001
        R["errors"]["fig_main"] = repr(e)
        log(traceback.format_exc())
    G, verdict = gates(res)
    R["gates"], R["verdict"] = G, verdict
    try:
        write_md(res, G, verdict)
    except Exception as e:  # noqa: BLE001
        R["errors"]["write_md"] = repr(e)
        log(traceback.format_exc())
    save_results()
    log(f"\nVERDICT: {verdict}\ndone in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
