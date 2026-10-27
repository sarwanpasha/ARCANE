#!/usr/bin/env python
"""Targeted tests of candidate cell states seen in ARCANE-flagged SLE cells (full Perez lupus, tag lupusfull).

Run from ~/arcane:
    python candidate_states.py [--root .] [--tag lupusfull] [--seeds 0,1,2] [--out summary/candstates]

States (gene programs; each evaluated in the fine labels matched by a regex, proliferating labels excluded):
  C1Q_ncM / C1Q_cM        C1QA C1QB C1QC FCGR1A CPVL                  in ncM / cM
  cDC1_like               CLEC9A DNASE1L3 WDFY4 XCR1 IDO1 CADM1        in cDC
  cDC1_vs_cDC2            cDC1 genes minus (CD1C FCER1A CLEC10A)        in cDC
  cycling_NK / cycling_CD8 MKI67 TYMS PCLAF TK1 MCM7 STMN1              in NK / CD8 labels
  HLADR_CD8               HLA-DRA CD74 CD38                             in CD8 labels
  FCRL5_B                 SOX4 FCRL5 ITGAX TBX21 JCHAIN MZB1            in naive/transitional B
  platelet_T              PF4 PPBP TUBB1 RGS18                          in T labels
  IFNg_mono               GBP1 GBP5 CXCL9 CXCL10 FCGR1A                 in cM+ncM
  typeI_mono (control)    IFI27 IFI44L RSAD2 SIGLEC1                    in cM+ncM
  IFNg_vs_typeI_mono      IFN-gamma genes minus type-I genes            in cM+ncM
Cell score = mean over genes of z(log1p CP10k) against REFERENCE healthy cells (role=='ref') of the same label
(sd floored, z clipped).  State-positive: score > 99th percentile of reference cells of that label (reference cells
themselves are scored against an out-of-fold threshold from the other reference fold).
Donor (= individual, donor_id) fraction of positive cells among cells of the state's labels, then
  (a) SLE vs healthy-query donors: within-cohort permutation p (logit fraction), Hodges-Lehmann diff + bootstrap CI
  (b) within SLE Spearman with donor type-I ISG (type-adjusted z, state genes removed) + CI; OLS HC3
      logit(frac) ~ SLE + ISG + cohort + ancestry + sex over SLE + healthy-query donors
  (c) flare / managed / treated (samples = individual x disease_state): Kruskal raw and cohort-centred; paired
      flare - treated within individuals
  (d) Mantel-Haenszel OR (strata = donors) that state-positive cells are ARCANE-flagged (p_unknown_ctrl<=alpha, seed
      majority) vs state-negative cells of the same labels; same for an ISG-only detector flagging the top-n ISG cells
      per label (n = ARCANE flags in that label); share of state-positive cells flagged by each
  (e) fraction of donors with >= 3 positive cells: reference (out-of-fold threshold), healthy query, SLE
Verdict "SLE-expanded beyond ISG": within-cohort p < 0.01 and HL > 0, ISG-adjusted SLE coefficient CI > 0,
|rho ISG| < 0.5.
Memory: expression read in row windows (backed h5ad), only the ~60 needed genes are kept (n_cells x genes float32).
Outputs: <out>/RESULTS.md, results.json, state_summary.csv, donor_state_fractions.csv, sample_state_fractions.csv,
         per_label_tests.csv, fig_*.png, log.txt.
"""
import argparse
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

MONO = r"^(cM|ncM)$|monocyte"
CD8 = r"CytoT|^T8|CD8"
TLAB = r"^T4|^T8|CytoT|CD4|CD8|Treg|MAIT|gdT|^T$|^T_"
CDC = r"^cDC|conventional dendritic|^DC\d?$"
CDC1 = ["CLEC9A", "DNASE1L3", "WDFY4", "XCR1", "IDO1", "CADM1"]
CDC2 = ["CD1C", "FCER1A", "CLEC10A"]
CYC = ["MKI67", "TYMS", "PCLAF", "TK1", "MCM7", "STMN1"]
C1Q = ["C1QA", "C1QB", "C1QC", "FCGR1A", "CPVL"]
IFNG = ["GBP1", "GBP5", "CXCL9", "CXCL10", "FCGR1A"]
IFNA = ["IFI27", "IFI44L", "RSAD2", "SIGLEC1"]
STATES = [
    dict(name="C1Q_ncM", up=C1Q, labels=r"^ncM$|non.?classical|CD16\+? ?mono"),
    dict(name="C1Q_cM", up=C1Q, labels=r"^cM$|^classical mono|CD14\+? ?mono"),
    dict(name="cDC1_like", up=CDC1, labels=CDC),
    dict(name="cDC1_vs_cDC2", up=CDC1, down=CDC2, labels=CDC),
    dict(name="cycling_NK", up=CYC, labels=r"^NK|natural killer"),
    dict(name="cycling_CD8", up=CYC, labels=CD8),
    dict(name="HLADR_CD8", up=["HLA-DRA", "CD74", "CD38"], labels=CD8),
    dict(name="FCRL5_B", up=["SOX4", "FCRL5", "ITGAX", "TBX21", "JCHAIN", "MZB1"],
         labels=r"B_naive|naive B|transitional|B_trans|B_imm"),
    dict(name="platelet_T", up=["PF4", "PPBP", "TUBB1", "RGS18"], labels=TLAB),
    dict(name="IFNg_mono", up=IFNG, labels=MONO),
    dict(name="typeI_mono", up=IFNA, labels=MONO, control=True),
    dict(name="IFNg_vs_typeI_mono", up=IFNG, down=IFNA, labels=MONO),
]
EXCL_LAB = r"prolif|cycling|progen|doublet|unknown|^nan$|platelet|^Plt|eryth|^RBC"


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tag", default="lupusfull")
    ap.add_argument("--h5ad", default=None, help="default <root>/data/std/<tag>.h5ad")
    ap.add_argument("--res_dir", default="res2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=0.05, help="ARCANE flag: p_unknown_ctrl <= alpha (seed majority)")
    ap.add_argument("--genesets", default="genesets.json")
    ap.add_argument("--isg_key", default="ISG_typeI")
    ap.add_argument("--out", default="summary/candstates")
    ap.add_argument("--q", type=float, default=99.0, help="reference percentile threshold")
    ap.add_argument("--sd_floor", type=float, default=0.1)
    ap.add_argument("--zclip", type=float, default=10.0)
    ap.add_argument("--min_ref_cells", type=int, default=50, help="min reference cells of a label to set a threshold")
    ap.add_argument("--min_cells", type=int, default=20, help="min cells of the state labels per donor/sample")
    ap.add_argument("--min_pos", type=int, default=3, help="(e) donors with >= this many positive cells")
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--label_map", default=None,
                    help="optional JSON {state: regex} overriding label regexes")
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
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_results():
    try:
        with open(os.path.join(OUT, "results.json"), "w") as f:
            json.dump(clean(R), f, indent=1, default=str)
    except Exception as e:  # noqa: BLE001
        log("save_results failed", repr(e))


def guarded(name, fn, *a, **k):
    t = time.time()
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001
        R["errors"][name] = repr(e)
        log(f"!!! {name} failed: {e!r}")
        log(traceback.format_exc())
        return None
    finally:
        R["timing_s"][name] = round(time.time() - t, 1)


plt.rcParams.update({"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 150})


def savefig(fig, name):
    try:
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    finally:
        plt.close(fig)


def wcsv(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)


def fmt(x, d=2):
    if x is None:
        return "NA"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "NA"
    if x != 0 and (abs(x) < 10 ** (-d) or abs(x) >= 1e4):
        return f"{x:.1e}"
    return f"{x:.{d}f}"


def eci(e, lo, hi, d=2):
    return f"{fmt(e, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


# ------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------
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


def is_sle_str(s):
    return bool(re.search(r"lupus|\bsle\b", str(s), re.I))


def collapse_levels(s, min_n=3, na="unknown"):
    s = pd.Series(s).astype(object)
    s = s.where(s.notna(), na).astype(str)
    s = s.replace({"nan": na, "None": na, "": na})
    vc = s.value_counts()
    return s.where(~s.isin(vc[vc < min_n].index), "other")


def hl_diff(a, b):
    """Hodges-Lehmann shift: median of all pairwise differences a_i - b_j."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.median(np.subtract.outer(a, b)))


def hl_boot(a, b, nboot, seed=3):
    a, b = np.asarray(a, float), np.asarray(b, float)
    est = hl_diff(a, b)
    if len(a) < 3 or len(b) < 3:
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = np.array([hl_diff(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
                   for _ in range(nboot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def strat_perm(y, g, strata, nperm, seed=0):
    """Within-stratum permutation of group labels; statistic = sum_s w_s (mean_1 - mean_0), w_s = n1 n0 / n.
    Only strata containing both groups contribute.  Two-sided p."""
    y, g, strata = np.asarray(y, float), np.asarray(g, int), np.asarray(strata).astype(str)
    rng = np.random.default_rng(seed)
    obs_stat, perm_stat, used, wsum = 0.0, np.zeros(nperm), [], 0.0
    for s in np.unique(strata):
        m = np.where(strata == s)[0]
        gg, yy = g[m], y[m]
        n1, n0 = int(gg.sum()), int(len(gg) - gg.sum())
        if n1 < 1 or n0 < 1:
            continue
        w = n1 * n0 / len(gg)
        tot = yy.sum()
        obs_stat += w * (yy[gg == 1].sum() / n1 - (tot - yy[gg == 1].sum()) / n0)
        # permuted: choose n1 positions at random (argsort of random keys)
        P = np.argsort(rng.random((nperm, len(gg))), axis=1)[:, :n1]
        s1 = yy[P].sum(1)
        perm_stat += w * (s1 / n1 - (tot - s1) / n0)
        used.append(dict(stratum=s, n1=n1, n0=n0))
        wsum += w
    if not used:
        return dict(stat=np.nan, p=np.nan, strata=[])
    p = (1 + np.sum(np.abs(perm_stat) >= abs(obs_stat) - 1e-12)) / (nperm + 1)
    return dict(stat=float(obs_stat / wsum), p=float(p), strata=used)


def spearman_ci(x, y, nboot, seed=1):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 6 or np.std(x) == 0 or np.std(y) == 0:
        return dict(rho=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=int(len(x)))
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
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
    p = 2 * stats.t.sf(np.abs(t), df)
    return {nm: dict(coef=float(b[i]), se=float(se[i]), lo=float(b[i] - tq * se[i]), hi=float(b[i] + tq * se[i]),
                     p=float(p[i])) for i, nm in enumerate(names)} | {"_n": n, "_k": k}


def design(df, nums, cats):
    cols, names = [np.ones(len(df))], ["Intercept"]
    for c in nums:
        cols.append(df[c].values.astype(float))
        names.append(c)
    for c in cats:
        lv = collapse_levels(df[c].values)
        levels = lv.value_counts().index.tolist()  # most frequent level = reference
        for l in levels[1:]:
            cols.append((lv.values == l).astype(float))
            names.append(f"{c}[{l}]")
    return np.column_stack(cols), names


def mantel_haenszel(tabs):
    """tabs: list of (a, b, c, d) per stratum; a = pos&flag, b = pos&noflag, c = neg&flag, d = neg&noflag.
    Returns MH OR with Robins-Breslow-Greenland 95% CI."""
    T = np.array([t for t in tabs if (t[0] + t[1]) > 0 and (t[2] + t[3]) > 0], float)
    if len(T) == 0:
        return dict(OR=np.nan, lo=np.nan, hi=np.nan, n_strata=0)
    a, b, c, d = T.T
    n = T.sum(1)
    Rr, S = a * d / n, b * c / n
    P, Q = (a + d) / n, (b + c) / n
    sR, sS = Rr.sum(), S.sum()
    if sR <= 0 or sS <= 0:
        return dict(OR=np.nan if sS <= 0 and sR <= 0 else (np.inf if sS <= 0 else 0.0), lo=np.nan, hi=np.nan,
                    n_strata=int(len(T)))
    orr = sR / sS
    var = (P * Rr).sum() / (2 * sR ** 2) + (P * S + Q * Rr).sum() / (2 * sR * sS) + (Q * S).sum() / (2 * sS ** 2)
    se = np.sqrt(var)
    return dict(OR=float(orr), lo=float(np.exp(np.log(orr) - 1.96 * se)), hi=float(np.exp(np.log(orr) + 1.96 * se)),
                n_strata=int(len(T)))


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - hw), min(1.0, c + hw)


def logit_frac(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    return np.log((k + 0.5) / (n - k + 0.5))


# ------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------
def load_genesets():
    p = ARGS.genesets if os.path.isabs(ARGS.genesets) else os.path.join(ARGS.root, ARGS.genesets)
    gs = json.load(open(p))
    return {str(k): [str(g) for g in v] for k, v in gs.items() if isinstance(v, (list, tuple))}


def load_flags():
    seeds = [int(s) for s in ARGS.seeds.split(",") if s.strip()]
    flags, names0, y0 = [], None, None
    for s in seeds:
        f = os.path.join(ARGS.root, ARGS.res_dir, f"{ARGS.tag}_seed{s}_cells.npz")
        if not os.path.exists(f):
            log("missing", f)
            continue
        C = np.load(f, allow_pickle=True)
        nm = pd.Index(np.asarray(C["names"]).astype(str))
        if "p_unknown_ctrl" in C.files:
            pv = np.asarray(C["p_unknown_ctrl"], float)
            # FIX: NaN p-values must stay missing (NaN <= alpha is False, which silently counted them as unflagged)
            fl = np.where(np.isfinite(pv), (pv <= ARGS.alpha).astype(float), np.nan)
        else:
            log("WARNING: p_unknown_ctrl missing; using `unknown`")
            fl = np.asarray(C["unknown"]).astype(float)
        if np.isnan(fl).any():
            log(f"seed {s}: {int(np.isnan(fl).sum())} cells with missing p_unknown_ctrl -> flag NA")
        dup = nm.duplicated()
        if dup.any():
            log(f"WARNING seed {s}: {int(dup.sum())} duplicated cell names in npz; keeping first")
        s_ = pd.Series(fl, index=nm)[~dup]
        if names0 is None:
            names0 = s_.index
            if "y" in C.files:
                y0 = pd.Series(np.asarray(C["y"]).astype(str), index=nm)[~dup]
        flags.append(s_.reindex(names0))
        log(f"seed {s}: {len(nm)} query cells, flag rate {np.nanmean(fl):.4f}")
    if not flags:
        raise FileNotFoundError("no cells npz")
    F = pd.concat(flags, axis=1).astype(float)
    R["n_seeds"] = len(flags)
    if len(flags) % 2 == 0:
        log(f"NOTE: {len(flags)} seeds (even): 'majority' = more than half, i.e. ties count as unflagged")
    return (F.mean(1) > 0.5).astype(float).where(F.notna().all(1)), y0


def symbols(var_names, var_df):
    var = pd.Index(np.asarray(var_names).astype(str))
    if np.mean(var.str.startswith("ENS")) > 0.5 and var_df is not None:
        for c in ["feature_name", "gene_symbols", "gene_symbol", "symbol", "gene_name"]:
            if c in var_df.columns:
                log(f"using var['{c}'] as gene symbols")
                return pd.Index(var_df[c].astype(str).values)
    return var


def load_expression(genes):
    """Returns obs (strings, needed columns), present genes, E (n_cells x n_genes float32 log1p CP10k)."""
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
    var = symbols(A.var_names, A.var)
    first = {}
    for i, g in enumerate(var):
        first.setdefault(g, i)
    present = [g for g in genes if g in first]
    gidx = np.array([first[g] for g in present], int)
    log(f"genes present {len(present)}/{len(genes)}; missing: {sorted(set(genes) - set(present))}")
    obs = A.obs
    cols = list(obs.columns)
    want = dict(label=find_col(cols, ["label"], [r"^label"]), role=find_col(cols, ["role"]),
                cond=find_col(cols, ["condition", "disease"]), fold=find_col(cols, ["fold"]),
                cohort=find_col(cols, ["Processing_Cohort"], [r"cohort"]),
                eth=find_col(cols, ["self_reported_ethnicity"], [r"ethnic", r"ancestry"]),
                sex=find_col(cols, ["sex"]), dstate=find_col(cols, ["disease_state"], [r"disease_state"]),
                indiv=find_col(cols, ["donor_id"], [r"^ind_cov$", r"individual"]), batch=find_col(cols, ["batch"]))
    log("obs columns:", want)
    R["obs_columns"] = want
    O = pd.DataFrame(index=pd.Index(np.asarray(A.obs_names).astype(str), name="cell"))
    for k, c in want.items():
        O[k] = obs[c].astype(str).values if c else "na"
    if want["indiv"] is None:
        O["indiv"] = O["batch"]
    n = A.n_obs if hasattr(A, "n_obs") else A.X.shape[0]
    E = np.zeros((n, len(gidx)), np.float32)
    lib = np.full(n, np.nan, np.float32)  # log10 total counts per cell (depth covariate)
    nonint = False
    for lo in range(0, n, ARGS.chunk):
        hi = min(n, lo + ARGS.chunk)
        try:
            Xi = A.X[lo:hi]
        except Exception as e:  # noqa: BLE001
            if not backed:
                raise
            log(f"backed row read failed ({e!r}); loading fully")
            A = ad.read_h5ad(h5)
            backed = False
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
            nonint = True  # already log-normalized: use as is
            E[lo:hi] = sub
        else:
            E[lo:hi] = np.log1p(sub * (1e4 / np.maximum(tot, 1))[:, None])
            lib[lo:hi] = np.log10(np.maximum(tot, 1))
        if (lo // ARGS.chunk) % 5 == 0:
            log(f"  rows {hi}/{n}")
    if nonint:
        log("WARNING: X not integer counts; assumed already log-normalized")
        R["X_nonint"] = True
    try:
        if backed:
            A.file.close()
    except Exception:  # noqa: BLE001
        pass
    del A
    if nonint:
        lib[:] = np.nan
    O["log_umi"] = lib
    return O, present, E


# ------------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------------
OOF = {}  # state -> {fold: float32 scores of ALL cells, valid for reference cells, using z params fit without that fold}


def _fold_ids(O):
    """Reference folds must be donor-disjoint for out-of-fold calibration to be honest."""
    fold = O.fold.astype(str).str.replace(r"\.0$", "", regex=True).values.copy()
    bad = np.isin(fold, ["nan", "na", "None", ""])
    r = O.is_ref.values
    ref = pd.DataFrame(dict(indiv=O.indiv.values[r], fold=fold[r], bad=bad[r]))
    nf = ref[~ref.bad].groupby("indiv").fold.nunique()
    ok = len(ref) and (~ref.bad).mean() > 0.99 and (nf <= 1).all() and ref.loc[~ref.bad, "fold"].nunique() >= 2
    if not ok:
        donors = sorted(ref.indiv.unique())
        rng = np.random.default_rng(12345)
        perm = rng.permutation(len(donors))
        dmap = {d: str(perm[i] % 2) for i, d in enumerate(donors)}
        log(f"WARNING: reference `fold` missing or not donor-disjoint (donors with >1 fold: {int((nf > 1).sum())}); "
            f"re-assigned 2 donor-level folds (seed 12345)")
        R["fold_reassigned"] = True
        fold[r] = [dmap[d] for d in O.indiv.values[r]]
    fold[~r] = "nan"
    return fold


def build(gs):
    flag, ynpz = load_flags()
    isg_genes = gs.get(ARGS.isg_key) or gs.get("ISG_typeI") or []
    state_genes = sorted(set(g for s in STATES for g in s["up"] + s.get("down", [])))
    isg_use = [g for g in isg_genes if g not in state_genes]
    genes = state_genes + isg_use
    O, present, E = load_expression(genes)
    gi = {g: i for i, g in enumerate(present)}
    O["role"] = O.role.str.lower()
    O["is_ref"] = O.role.eq("ref")
    O["is_query"] = O.role.eq("query")
    O["is_sle"] = O.cond.map(is_sle_str).astype(int)
    # ---- npz <-> h5ad join (FIX: verify instead of silently producing all-NaN flags)
    qidx = O.index[O.is_query.values]
    hit = float(pd.Index(qidx).isin(flag.index).mean()) if len(qidx) else 0.0
    R["join_name_match_frac"] = hit
    if hit < 0.99 and len(flag) == len(qidx):
        log(f"WARNING: only {hit:.3f} of h5ad query cells found by name in npz; npz has the same length as the "
            f"query cells, so joining by POSITION (CONTEXT: same order)")
        O["flag"] = np.nan
        O.loc[O.is_query.values, "flag"] = flag.values
        if ynpz is not None:
            ynpz = pd.Series(ynpz.values, index=qidx)
        R["join_mode"] = "position"
    else:
        if hit < 0.99:
            log(f"WARNING: only {hit:.3f} of h5ad query cells matched npz names; unmatched cells have flag NA")
        O["flag"] = flag.reindex(O.index).values
        R["join_mode"] = "name"
    O.loc[~O.is_query, "flag"] = np.nan
    if ynpz is not None:
        yq = ynpz.reindex(qidx)
        m = yq.notna().values
        agree = float((yq.values[m] == O.label.values[O.is_query.values][m]).mean()) if m.any() else np.nan
        R["join_label_agreement"] = agree
        log(f"join check: npz y == h5ad label for {agree:.4f} of joined query cells")
        if np.isfinite(agree) and agree < 0.9:
            log("!!! WARNING: npz labels disagree with h5ad labels: the cell join is probably WRONG")
            R["errors"]["join"] = f"label agreement {agree:.3f}"
    nq = int(O.is_query.sum())
    R["n_query_cells"] = nq
    R["n_query_with_flag"] = int(O.flag.notna().sum())
    if R["n_query_with_flag"] < 0.9 * nq:
        log(f"WARNING: {nq - R['n_query_with_flag']} query cells without a flag (excluded from (d))")
    R["flag_rate"] = {"SLE": float(O.flag[O.is_query & (O.is_sle == 1)].mean()),
                      "healthy_query": float(O.flag[O.is_query & (O.is_sle == 0)].mean())}
    log("cells:", len(O), "ref", int(O.is_ref.sum()), "query", nq, "with flag", R["n_query_with_flag"], R["flag_rate"])
    if O.is_ref.sum() == 0:
        raise ValueError("no role=='ref' cells")
    if (O.is_ref & (O.is_sle == 1)).any():
        log("WARNING: reference cells labelled SLE")
    # FIX: individuals present in both roles would have their query cells tested against thresholds fit on themselves
    both = O.groupby("indiv").role.agg(lambda s: s.eq("ref").any() and s.eq("query").any())
    both = both[both].index
    R["indiv_in_both_roles"] = [str(x) for x in both]
    if len(both):
        log(f"WARNING: {len(both)} individuals have both ref and query cells; their query cells are excluded "
            f"from donor-level tests: {list(both)[:10]}")
    O["mixed_role"] = O.indiv.isin(both).values
    O["fold"] = _fold_ids(O)
    labs = O.label.values
    isref = O.is_ref.values
    ok_lab = pd.Series(labs[isref]).value_counts().to_dict()
    ok_lab = {l: int(ok_lab.get(l, 0)) for l in np.unique(labs)}
    R["ref_cells_per_label"] = ok_lab
    lmap = json.load(open(ARGS.label_map)) if ARGS.label_map else {}
    all_labels = sorted(set(labs))
    R["labels"] = all_labels
    states, specs = [], []
    for st in STATES:
        up = [gi[g] for g in st["up"] if g in gi]
        dn = [gi[g] for g in st.get("down", []) if g in gi]
        rx = lmap.get(st["name"], st["labels"])
        L = [l for l in all_labels if re.search(rx, l, re.I) and not re.search(EXCL_LAB, l, re.I)
             and ok_lab.get(l, 0) >= ARGS.min_ref_cells]
        info = dict(name=st["name"], labels=L, genes_up=[g for g in st["up"] if g in gi],
                    genes_down=[g for g in st.get("down", []) if g in gi], control=st.get("control", False))
        states.append(info)
        if len(up) < 2 or not L or (st.get("down") and len(dn) < 2):
            log(f"state {st['name']}: skipped (genes up {len(up)}, down {len(dn)}, labels {L})")
            info["skipped"] = True
            continue
        specs.append((info, up, dn))
        log(f"state {st['name']}: labels {L}, genes {info['genes_up']} - {info['genes_down']}")

    def score(Zs, up, dn):
        s = np.nanmean(Zs[:, up], axis=1)
        if dn:
            s = s - np.nanmean(Zs[:, dn], axis=1)
        return s.astype(np.float32)

    # per label reference z parameters (full reference for query cells; out-of-fold for reference cells)
    Z = np.zeros_like(E)
    for info, _, _ in specs:
        OOF[info["name"]] = {}
    folds = O.fold.values
    for l in np.unique(labs):
        m = labs == l
        r = m & isref
        if r.sum() < ARGS.min_ref_cells:
            Z[m] = np.nan
            continue
        mu = E[r].mean(0)
        sd = np.maximum(E[r].std(0), ARGS.sd_floor)
        Z[m] = np.clip((E[m] - mu) / sd, -ARGS.zclip, ARGS.zclip)
        # FIX (leakage): reference cells of fold f are scored with z parameters fit on the other folds only
        for f in np.unique(folds[r]):
            tr = r & (folds != f)
            if tr.sum() < ARGS.min_ref_cells:
                continue
            mu_f = E[tr].mean(0)
            sd_f = np.maximum(E[tr].std(0), ARGS.sd_floor)
            Zr = np.clip((E[r] - mu_f) / sd_f, -ARGS.zclip, ARGS.zclip)
            for info, up, dn in specs:
                if l not in info["labels"]:
                    continue
                arr = OOF[info["name"]].setdefault(f, np.full(len(O), np.nan, np.float32))
                arr[r] = score(Zr, up, dn)
    del E
    ii = [gi[g] for g in isg_use if g in gi]
    O["isg"] = np.nanmean(Z[:, ii], axis=1) if len(ii) >= 3 else np.nan
    R["isg_genes_used"] = [g for g in isg_use if g in gi]
    for info, up, dn in specs:
        O[f"s_{info['name']}"] = score(Z, up, dn)
    del Z
    return O, states


def threshold_positive(O, st):
    """Positive calls for cells of the state's labels.  Query cells: threshold = q-th percentile of all reference
    cells of the label.  Reference cells of fold f: z parameters AND threshold fit on the other folds (donor-disjoint),
    so the reference positive rate is an honest calibration check."""
    sc = O[f"s_{st['name']}"].values
    lab = O.label.values
    isref = O.is_ref.values
    folds = O.fold.values
    oof = OOF.get(st["name"], {})
    pos = np.zeros(len(O), bool)
    inlab = np.isin(lab, st["labels"])
    thr_info = {}
    for l in st["labels"]:
        m = lab == l
        r = m & isref
        thr = np.nanpercentile(sc[r], ARGS.q)
        q = m & ~isref
        pos[q] = sc[q] > thr
        uf = list(np.unique(folds[r]))
        oof_ok = len(uf) >= 2
        if oof_ok:
            for f in uf:
                tr = r & (folds != f)
                te = r & (folds == f)
                s_f = oof.get(f)
                if tr.sum() < ARGS.min_ref_cells or s_f is None:
                    oof_ok = False
                    break
                pos[te] = s_f[te] > np.nanpercentile(s_f[tr], ARGS.q)
        if not oof_ok:
            pos[r] = sc[r] > thr
        thr_info[l] = dict(thr=float(thr), oof=bool(oof_ok), ref_pos_rate=float(pos[r].mean()))
    st["thresholds"] = thr_info
    st["oof_all"] = all(v["oof"] for v in thr_info.values())
    # FIX: query cells of individuals that are also reference donors are dropped from the state's cell set
    inlab = inlab & ~(O.mixed_role.values & ~isref)
    return pos, inlab


# ------------------------------------------------------------------------------------------------
# donor tables and tests
# ------------------------------------------------------------------------------------------------
def donor_meta(O):
    g = O.groupby("indiv")
    meta = pd.DataFrame(dict(
        is_sle=g.is_sle.max(), is_ref=g.is_ref.max(), is_query=g.is_query.max(),
        cohort=g.cohort.agg(lambda s: s.value_counts().index[0]),
        eth=g.eth.agg(lambda s: s.value_counts().index[0]),
        sex=g.sex.agg(lambda s: s.value_counts().index[0]),
        n_cells=g.size()))
    meta["group"] = np.where(meta.is_ref == 1, "ref", np.where(meta.is_sle == 1, "SLE", "healthy_query"))
    # donor ISG: mean over labels of donor x label mean cell ISG z (labels with >= 10 cells)
    dl = O.groupby(["indiv", "label"]).isg.agg(["mean", "size"])
    dl = dl[dl["size"] >= 10]
    meta["isg"] = dl["mean"].groupby(level=0).mean().reindex(meta.index)
    meta["flag_rate"] = O[O.is_query].groupby("indiv").flag.mean().reindex(meta.index)
    meta["log_umi"] = O.groupby("indiv").log_umi.median().reindex(meta.index)
    meta["n_cohorts"] = g.cohort.nunique()
    if (meta.n_cohorts > 1).any():
        log(f"NOTE: {int((meta.n_cohorts > 1).sum())} individuals span >1 processing cohort (modal cohort used)")
    return meta


def isg_matched_flags(O, labels):
    """ISG-only detector: per label, flag the top-n query cells by ISG score, n = ARCANE flags in that label."""
    out = pd.Series(np.nan, index=O.index)
    for l in labels:
        m = (O.label.values == l) & O.flag.notna().values
        if m.sum() == 0:
            continue
        nfl = int(O.flag.values[m].sum())
        idx = np.where(m)[0]
        isg = np.nan_to_num(O.isg.values[idx], nan=-np.inf)
        f = np.zeros(len(idx))
        if nfl > 0:
            f[np.argsort(-isg, kind="stable")[:nfl]] = 1
        out.iloc[idx] = f
    return out


def analyse_state(O, meta, st):
    name = st["name"]
    pos, inlab = threshold_positive(O, st)
    C = O.loc[inlab, ["indiv", "dstate", "label", "is_ref", "is_query", "is_sle", "flag", "cohort"]].copy()
    C["pos"] = pos[inlab]
    C["isgflag"] = isg_matched_flags(O, st["labels"]).values[inlab]
    res = dict(name=name, labels=st["labels"], control=st["control"],
               ref_pos_rate_oof=float(C.pos[C.is_ref].mean()) if C.is_ref.any() else np.nan,
               pos_rate_hq=float(C.pos[C.is_query & (C.is_sle == 0)].mean()),
               pos_rate_sle=float(C.pos[C.is_query & (C.is_sle == 1)].mean()))
    # donor fractions
    D = C.groupby("indiv").agg(n=("pos", "size"), k=("pos", "sum"))
    D = D.join(meta[["group", "is_sle", "cohort", "eth", "sex", "isg", "log_umi"]])
    D["frac"] = D.k / D.n
    D["logit"] = logit_frac(D.k, D.n)
    Dv = D[D.n >= ARGS.min_cells].copy()
    Q = Dv[Dv.group != "ref"].copy()
    sle, hq = Q[Q.is_sle == 1], Q[Q.is_sle == 0]
    res.update(n_sle=int(len(sle)), n_hq=int(len(hq)), n_ref=int((Dv.group == "ref").sum()),
               med_sle=float(sle.frac.median()) if len(sle) else np.nan,
               med_hq=float(hq.frac.median()) if len(hq) else np.nan,
               med_ref=float(Dv.frac[Dv.group == "ref"].median()) if (Dv.group == "ref").any() else np.nan)
    # (a)
    hl, lo, hi = hl_boot(sle.frac.values, hq.frac.values, ARGS.nboot)
    res["a_HL"], res["a_HL_lo"], res["a_HL_hi"] = hl, lo, hi
    sp_ = strat_perm(Q.logit.values, Q.is_sle.values, Q.cohort.values, ARGS.nperm,
                     seed=sum(map(ord, name)))
    res["a_perm_p"], res["a_perm_stat_logit"], res["a_strata"] = sp_["p"], sp_["stat"], sp_["strata"]
    # sensitivity: strata = cohort x ancestry (ancestry differs between SLE and healthy in Perez)
    ce = Q.cohort.astype(str) + "|" + collapse_levels(Q.eth.values).values
    sp2 = strat_perm(Q.logit.values, Q.is_sle.values, ce.values, ARGS.nperm, seed=sum(map(ord, name)) + 1)
    res["a_perm_p_cohort_eth"] = sp2["p"]
    res["a_perm_cohort_eth_n"] = int(sum(x["n1"] + x["n0"] for x in sp2["strata"]))
    res["a_perm_n"] = int(sum(x["n1"] + x["n0"] for x in sp_["strata"]))
    res["a_mwu_p_pooled"] = float(stats.mannwhitneyu(sle.frac, hq.frac, alternative="two-sided")[1]) \
        if len(sle) > 2 and len(hq) > 2 else np.nan
    # (b)
    sc = spearman_ci(sle.isg.values, sle.frac.values, ARGS.nboot)
    res["b_rho"], res["b_rho_lo"], res["b_rho_hi"], res["b_rho_p"] = sc["rho"], sc["lo"], sc["hi"], sc["p"]
    Qm = Q[np.isfinite(Q.isg)].copy()
    Qm["isg_z"] = (Qm.isg - Qm.isg.mean()) / (Qm.isg.std() + 1e-12)
    Qm["SLE"] = Qm.is_sle.astype(float)
    if len(Qm) >= 15 and Qm.SLE.nunique() == 2:
        X, nm = design(Qm, ["SLE", "isg_z"], ["cohort", "eth", "sex"])
        f = ols_hc3(Qm.logit.values, X, nm)
        res["b_ols_SLE"], res["b_ols_SLE_lo"], res["b_ols_SLE_hi"], res["b_ols_SLE_p"] = (
            f["SLE"]["coef"], f["SLE"]["lo"], f["SLE"]["hi"], f["SLE"]["p"])
        res["b_ols_ISG"], res["b_ols_ISG_lo"], res["b_ols_ISG_hi"] = f["isg_z"]["coef"], f["isg_z"]["lo"], f["isg_z"]["hi"]
        X0, nm0 = design(Qm, ["SLE"], ["cohort", "eth", "sex"])
        f0 = ols_hc3(Qm.logit.values, X0, nm0)
        res["b_ols_SLE_noISG"], res["b_ols_SLE_noISG_lo"], res["b_ols_SLE_noISG_hi"] = (
            f0["SLE"]["coef"], f0["SLE"]["lo"], f0["SLE"]["hi"])
        res["b_ols_n"] = int(len(Qm))
        # sensitivity: + donor median log10 UMI (sparse-gene positivity depends on depth)
        if Qm.log_umi.notna().sum() >= 15:
            Qd = Qm[Qm.log_umi.notna()].copy()
            Qd["umi_z"] = (Qd.log_umi - Qd.log_umi.mean()) / (Qd.log_umi.std() + 1e-12)
            Xd, nmd = design(Qd, ["SLE", "isg_z", "umi_z"], ["cohort", "eth", "sex"])
            fd = ols_hc3(Qd.logit.values, Xd, nmd)
            res["b_ols_SLE_depth"], res["b_ols_SLE_depth_lo"], res["b_ols_SLE_depth_hi"] = (
                fd["SLE"]["coef"], fd["SLE"]["lo"], fd["SLE"]["hi"])
    # (c) disease_state within SLE, samples = individual x disease_state
    S = C[C.is_query & (C.is_sle == 1)].groupby(["indiv", "dstate"]).agg(n=("pos", "size"), k=("pos", "sum"),
                                                                             cohort=("cohort", "first")).reset_index()
    S = S[S.n >= ARGS.min_cells]
    S["frac"] = S.k / S.n
    S["logit"] = logit_frac(S.k, S.n)
    S["logit_cc"] = S.logit - S.groupby("cohort").logit.transform("median")
    S["state"] = name
    st_lv = [l for l in ["managed", "flare", "treated"] if (S.dstate == l).sum() >= 3]
    res["c_n"] = {l: int((S.dstate == l).sum()) for l in ["managed", "flare", "treated"]}
    res["c_med"] = {l: float(S.frac[S.dstate == l].median()) for l in ["managed", "flare", "treated"]
                    if (S.dstate == l).any()}
    # FIX (pseudoreplication): Kruskal uses ONE sample per individual (random, seeded) among the tested levels
    Sk = S[S.dstate.isin(st_lv)]
    Sk = Sk.sample(frac=1.0, random_state=7).drop_duplicates("indiv")
    st_lv = [l for l in st_lv if (Sk.dstate == l).sum() >= 3]
    res["c_kw_n_indiv"] = int(len(Sk))
    if len(st_lv) >= 2:
        try:
            res["c_kw_p"] = float(stats.kruskal(*[Sk.logit[Sk.dstate == l] for l in st_lv])[1])
            res["c_kw_p_cohort"] = float(stats.kruskal(*[Sk.logit_cc[Sk.dstate == l] for l in st_lv])[1])
        except ValueError as e:  # all values identical
            log(f"  kruskal skipped: {e}")
    pv = S.pivot_table(index="indiv", columns="dstate", values="frac")
    if {"flare", "treated"} <= set(pv.columns):
        dd = (pv["flare"] - pv["treated"]).dropna()
        res["c_paired_n"] = int(len(dd))
        if len(dd) >= 5:
            res["c_paired_med"] = float(dd.median())
            try:
                res["c_paired_p"] = float(stats.wilcoxon(dd)[1]) if (dd != 0).any() else 1.0
            except ValueError:
                res["c_paired_p"] = np.nan
    # (d) flagged odds
    Cq = C[C.is_query & C.flag.notna()]
    for grp, gsel in [("sle", Cq.is_sle == 1), ("hq", Cq.is_sle == 0)]:
        Cg = Cq[gsel]
        for det in ["flag", "isgflag"]:
            t = Cg.groupby("indiv").apply(lambda x: (
                int((x.pos & (x[det] == 1)).sum()), int((x.pos & (x[det] == 0)).sum()),
                int((~x.pos & (x[det] == 1)).sum()), int((~x.pos & (x[det] == 0)).sum())))
            mh = mantel_haenszel(list(t.values))
            res[f"d_MH_{det}_{grp}"], res[f"d_MH_{det}_{grp}_lo"], res[f"d_MH_{det}_{grp}_hi"] = (
                mh["OR"], mh["lo"], mh["hi"])
            res[f"d_MH_{det}_{grp}_n"] = mh["n_strata"]
            if det == "flag" and grp == "sle":
                T = np.array(list(t.values), float)
                if len(T):
                    ors = ((T[:, 0] + .5) * (T[:, 3] + .5)) / ((T[:, 1] + .5) * (T[:, 2] + .5))
                    ok = (T[:, 0] + T[:, 1]) > 0
                    res["d_donorOR_median_sle"] = float(np.median(ors[ok])) if ok.any() else np.nan
        P = Cg[Cg.pos]
        res[f"d_npos_{grp}"] = int(len(P))
        res[f"d_pos_flag_{grp}"] = float(P.flag.mean()) if len(P) else np.nan
        res[f"d_pos_isgflag_{grp}"] = float(P.isgflag.mean()) if len(P) else np.nan
        res[f"d_pos_flag_not_isg_{grp}"] = float(((P.flag == 1) & (P.isgflag == 0)).mean()) if len(P) else np.nan
        res[f"d_neg_flag_{grp}"] = float(Cg.flag[~Cg.pos].mean()) if (~Cg.pos).any() else np.nan
    # (e) donors with >= min_pos positive cells
    for grp in ["ref", "healthy_query", "SLE"]:
        x = Dv[Dv.group == grp]
        p_, l_, h_ = wilson(int((x.k >= ARGS.min_pos).sum()), len(x))
        res[f"e_{grp}"], res[f"e_{grp}_lo"], res[f"e_{grp}_hi"] = p_, l_, h_
    # per-label (a) breakdown
    pl = []
    if len(st["labels"]) > 1:
        for l in st["labels"]:
            Cl = C[C.label == l]
            Dl = Cl.groupby("indiv").agg(n=("pos", "size"), k=("pos", "sum")).join(meta[["group", "is_sle", "cohort"]])
            Dl = Dl[(Dl.n >= ARGS.min_cells) & (Dl.group != "ref")]
            if Dl.is_sle.nunique() < 2:
                continue
            Dl["frac"] = Dl.k / Dl.n
            s1, s0 = Dl[Dl.is_sle == 1], Dl[Dl.is_sle == 0]
            pp = strat_perm(logit_frac(Dl.k, Dl.n), Dl.is_sle.values, Dl.cohort.values, min(ARGS.nperm, 2000))
            pl.append(dict(state=name, label=l, n_sle=len(s1), n_hq=len(s0), med_sle=s1.frac.median(),
                           med_hq=s0.frac.median(), HL=hl_diff(s1.frac, s0.frac), perm_p=pp["p"]))
    # verdict
    crit = dict(p=bool(np.isfinite(res.get("a_perm_p", np.nan)) and res["a_perm_p"] < 0.01 and res["a_HL"] > 0),
                ols=bool(np.isfinite(res.get("b_ols_SLE_lo", np.nan)) and res["b_ols_SLE_lo"] > 0),
                rho=bool(np.isfinite(res.get("b_rho", np.nan)) and abs(res["b_rho"]) < 0.5))
    res["crit"] = crit
    res["verdict"] = "yes" if all(crit.values()) else "no"
    res["thresholds"] = st.get("thresholds")
    res["ref_oof"] = bool(st.get("oof_all", False))
    D = D.reset_index()
    D["state"] = name
    return res, D, S, pl


# ------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------
GCOL = {"ref": "#8c8c8c", "healthy_query": "#3b75af", "SLE": "#c8423b"}


def fig_fractions(DF, res):
    names = [r["name"] for r in res]
    nc = 4
    nr = int(np.ceil(len(names) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.6 * nr), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, nm in zip(axes.ravel(), names):
        d = DF[(DF.state == nm) & (DF.n >= ARGS.min_cells)]
        for i, g in enumerate(["ref", "healthy_query", "SLE"]):
            v = 100 * d.frac[d.group == g].values
            if len(v) == 0:
                continue
            ax.boxplot([v], positions=[i], widths=0.6, showfliers=False, medianprops=dict(color="k", lw=2))
            ax.scatter(i + rng.uniform(-.2, .2, len(v)), v, s=12, color=GCOL[g], alpha=.7)
        ax.axhline(100 - ARGS.q, ls="--", color="grey", lw=1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Ref", "HC", "SLE"])
        ax.set_ylabel(f"{nm}\n% positive", fontsize=15)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    savefig(fig, "fig_state_fractions.png")


def fig_isg(DF, res):
    names = [r["name"] for r in res]
    nc = 4
    nr = int(np.ceil(len(names) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.6 * nr), squeeze=False)
    for ax, r in zip(axes.ravel(), res):
        d = DF[(DF.state == r["name"]) & (DF.n >= ARGS.min_cells) & (DF.group != "ref")]
        for g in ["healthy_query", "SLE"]:
            x = d[d.group == g]
            ax.scatter(x.isg, 100 * x.frac, s=14, color=GCOL[g], alpha=.7, label="HC" if g != "SLE" else "SLE")
        ax.set_xlabel("donor ISG z")
        ax.set_ylabel(f"{r['name']}\n% positive", fontsize=15)
        ax.text(0.03, 0.95, f"rho {fmt(r.get('b_rho'))}", transform=ax.transAxes, va="top", fontsize=14)
    for ax in axes.ravel()[len(res):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="upper right", frameon=False)
    savefig(fig, "fig_state_vs_isg.png")


def fig_effects(res):
    names = [r["name"] for r in res]
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(20, 0.55 * len(names) + 2.5), sharey=True)
    specs = [("a_HL", "a_HL_lo", "a_HL_hi", 100, "SLE - HC (pp, HL)", 0),
             ("b_ols_SLE", "b_ols_SLE_lo", "b_ols_SLE_hi", 1, "SLE coef | ISG (logit)", 0),
             ("b_rho", "b_rho_lo", "b_rho_hi", 1, "Spearman rho with ISG (SLE)", None)]
    for ax, (k, kl, kh, sc, xl, ref) in zip(axes, specs):
        for yi, r in zip(y, res):
            e, lo, hi = [np.nan if r.get(kk) is None else float(r[kk]) for kk in (k, kl, kh)]
            col = "#c8423b" if r["verdict"] == "yes" else ("#8c8c8c" if r["control"] else "#3b75af")
            ax.plot([sc * lo, sc * hi], [yi, yi], color=col, lw=2.5)
            ax.plot(sc * e, yi, "o", color=col, ms=8)
        if ref is not None:
            ax.axvline(ref, color="k", lw=1)
        else:
            for v in (-0.5, 0.5):
                ax.axvline(v, color="grey", ls="--", lw=1)
        ax.set_xlabel(xl)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    savefig(fig, "fig_state_effects.png")


def fig_or(res):
    names = [r["name"] for r in res]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 2.5))
    for off, det, col, lab in [(0.15, "flag", "#c8423b", "ARCANE flag"), (-0.15, "isgflag", "#8c8c8c", "ISG top-n")]:
        for yi, r in zip(y, res):
            e, lo, hi = [r.get(f"d_MH_{det}_sle{s}", np.nan) for s in ("", "_lo", "_hi")]
            if e is None or not np.isfinite(e) or e <= 0:
                continue
            lo = lo if lo is not None and np.isfinite(lo) and lo > 0 else e
            hi = hi if hi is not None and np.isfinite(hi) else e
            ax.plot([lo, hi], [yi + off] * 2, color=col, lw=2.5)
            ax.plot(e, yi + off, "o", color=col, ms=7, label=lab if yi == y[0] else None)
    ax.set_xscale("log")
    ax.axvline(1, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("MH odds ratio, state+ vs state- (SLE)")
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, "fig_flag_odds.png")


# ------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------
def write_md(res, meta):
    L = ["# Candidate SLE states: targeted donor-level tests", ""]
    nd = meta.group.value_counts().to_dict()
    L.append(f"Donors: SLE {nd.get('SLE', 0)}, healthy query {nd.get('healthy_query', 0)}, ref {nd.get('ref', 0)}. "
             f"ARCANE flag p_unknown_ctrl<={ARGS.alpha} (majority of {R.get('n_seeds')} seeds): SLE "
             f"{fmt(100 * R['flag_rate']['SLE'], 1)}%, HC {fmt(100 * R['flag_rate']['healthy_query'], 1)}% of cells. "
             f"State+ = score > ref q{ARGS.q:g} of the label (score = mean z vs ref cells); donor ISG = type-I ISG z "
             f"({len(R.get('isg_genes_used', []))} genes, state genes removed), averaged over labels. "
             f"Donors need >= {ARGS.min_cells} cells of the state's labels. CIs: {ARGS.nboot} donor bootstraps; "
             f"perm p: {ARGS.nperm} within-cohort permutations.")
    L += ["", "## (a,b,e) Expansion and ISG dependence", "",
          "| state | labels | n S/H | med % S / H / ref | HL pp [CI] | perm p (BH q) / cohort x anc p | rho ISG [CI] "
          "| SLE coef adj ISG [CI] | no ISG / +depth | donors>=3 pos ref / H / S | expanded beyond ISG |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        crit = "".join(k[0].upper() if v else "-" for k, v in r["crit"].items())
        L.append(f"| {r['name']}{' (ctrl)' if r['control'] else ''} | {','.join(r['labels'])[:28]} | "
                 f"{r['n_sle']}/{r['n_hq']} | {fmt(100 * r['med_sle'], 1)} / {fmt(100 * r['med_hq'], 1)} / "
                 f"{fmt(100 * r['med_ref'], 1)} | {eci(100 * r['a_HL'], 100 * r['a_HL_lo'], 100 * r['a_HL_hi'], 1)} | "
                 f"{fmt(r['a_perm_p'], 4)} ({fmt(r.get('a_perm_q'), 3)}) / {fmt(r.get('a_perm_p_cohort_eth'), 4)} | "
                 f"{eci(r['b_rho'], r['b_rho_lo'], r['b_rho_hi'])} | "
                 f"{eci(r.get('b_ols_SLE'), r.get('b_ols_SLE_lo'), r.get('b_ols_SLE_hi'))} | "
                 f"{fmt(r.get('b_ols_SLE_noISG'))} / {fmt(r.get('b_ols_SLE_depth'))} | "
                 f"{fmt(100 * r['e_ref'], 0)} / {fmt(100 * r['e_healthy_query'], 0)} / {fmt(100 * r['e_SLE'], 0)}% | "
                 f"**{r['verdict']}** ({crit}) |")
    L += ["", "Criteria letters: P = within-cohort p<0.01 and HL>0, O = ISG-adjusted SLE coef CI>0, R = |rho|<0.5. "
          "BH q over non-control states. Caveat: state gene lists were chosen from DE on these same donors, so p-values "
          "are post-selection; (d) ORs are inflated by construction (states were found in flagged cells). "
          "(c) Kruskal uses one sample per individual.",
          "", "## (c,d) Disease state and ARCANE flags (SLE donors)", "",
          "| state | med % managed/flare/treated (n) | KW p raw / cohort-centred | flare-treated paired (n, p) "
          "| MH OR ARCANE [CI] | MH OR ISG top-n [CI] | state+ cells: % ARCANE / % ISG / % ARCANE-not-ISG | "
          "HC MH OR ARCANE |", "|---|---|---|---|---|---|---|---|"]
    for r in res:
        cm = "/".join(fmt(100 * r.get("c_med", {}).get(l, np.nan), 1) for l in ["managed", "flare", "treated"])
        cn = "/".join(str(r.get("c_n", {}).get(l, 0)) for l in ["managed", "flare", "treated"])
        L.append(f"| {r['name']} | {cm} ({cn}) | {fmt(r.get('c_kw_p'), 3)} / {fmt(r.get('c_kw_p_cohort'), 3)} | "
                 f"{fmt(100 * r.get('c_paired_med', np.nan), 1)}pp ({r.get('c_paired_n', 0)}, "
                 f"{fmt(r.get('c_paired_p'), 3)}) | "
                 f"{eci(r.get('d_MH_flag_sle'), r.get('d_MH_flag_sle_lo'), r.get('d_MH_flag_sle_hi'), 1)} | "
                 f"{eci(r.get('d_MH_isgflag_sle'), r.get('d_MH_isgflag_sle_lo'), r.get('d_MH_isgflag_sle_hi'), 1)} | "
                 f"{fmt(100 * r.get('d_pos_flag_sle', np.nan), 0)} / {fmt(100 * r.get('d_pos_isgflag_sle', np.nan), 0)}"
                 f" / {fmt(100 * r.get('d_pos_flag_not_isg_sle', np.nan), 0)} (n {r.get('d_npos_sle', 0)}) | "
                 f"{fmt(r.get('d_MH_flag_hq'), 1)} |")
    yes = [r["name"] for r in res if r["verdict"] == "yes"]
    L += ["", f"**SLE-expanded beyond ISG: {', '.join(yes) if yes else 'none'}** "
          f"({len(yes)}/{len(res)} states tested).",
          f"Ref out-of-fold positive rate (calibration, target {100 - ARGS.q:g}%): " +
          ", ".join(f"{r['name']} {fmt(100 * r['ref_pos_rate_oof'], 1)}{'' if r.get('ref_oof') else '(in-sample)'}"
                    for r in res) + ".",
          f"Join: {R.get('join_mode')} (name match {fmt(R.get('join_name_match_frac'), 3)}, label agreement "
          f"{fmt(R.get('join_label_agreement'), 3)}); query cells with flag {R.get('n_query_with_flag')}/"
          f"{R.get('n_query_cells')}; individuals in both roles {len(R.get('indiv_in_both_roles', []))}"
          f"{'; ref folds re-assigned by donor' if R.get('fold_reassigned') else ''}.", ""]
    if R["errors"]:
        L.append("Errors: " + "; ".join(f"{k}: {v[:120]}" for k, v in R["errors"].items()))
    L.append("Files: state_summary.csv, donor_state_fractions.csv, sample_state_fractions.csv, per_label_tests.csv, "
             "fig_state_fractions.png, fig_state_vs_isg.png, fig_state_effects.png, fig_flag_odds.png.")
    txt = "\n".join(L) + "\n"
    if len(txt.encode()) > 8000:
        txt = txt[:7900] + "\n(truncated; see state_summary.csv)\n"
    open(os.path.join(OUT, "RESULTS.md"), "w").write(txt)


def main(argv=None):
    global ARGS, OUT, LOGF
    ARGS = parse_args(argv)
    OUT = ARGS.out if os.path.isabs(ARGS.out) else os.path.join(ARGS.root, ARGS.out)
    os.makedirs(OUT, exist_ok=True)
    LOGF = open(os.path.join(OUT, "log.txt"), "w")
    R["args"] = vars(ARGS)
    gs = load_genesets()
    O, states = build(gs)
    meta = donor_meta(O)
    wcsv(meta.reset_index(), "donor_table.csv")
    res, DFs, Ss, PL = [], [], [], []
    for st in states:
        if st.get("skipped"):
            R.setdefault("skipped", []).append(st["name"])
            continue
        log(f"\n=== [{time.time() - T0:6.0f}s] {st['name']}")
        out = guarded(st["name"], analyse_state, O, meta, st)
        if out is None:
            continue
        r, D, S, pl = out
        res.append(r)
        DFs.append(D)
        Ss.append(S)
        PL += pl
        log({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
             if k.startswith(("a_HL", "a_perm_p", "b_rho", "b_ols_SLE", "d_MH_flag_sle", "verdict", "med_"))})
        save_results()
    # BH across non-control states for the primary within-cohort permutation p
    ps = [(i, r["a_perm_p"]) for i, r in enumerate(res) if not r["control"] and np.isfinite(r.get("a_perm_p", np.nan))]
    if ps:
        idx, pv = zip(*ps)
        pv = np.asarray(pv)
        o = np.argsort(pv)
        q = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        qq = np.empty_like(q)
        qq[o] = np.minimum(q, 1)
        for i, v in zip(idx, qq):
            res[i]["a_perm_q"] = float(v)
    R["states"] = res
    save_results()
    if not res:
        log("no state analysed")
        return
    DF = pd.concat(DFs, ignore_index=True)
    wcsv(DF, "donor_state_fractions.csv")
    wcsv(pd.concat(Ss, ignore_index=True), "sample_state_fractions.csv")
    wcsv(pd.DataFrame(PL), "per_label_tests.csv")
    flat = pd.DataFrame([{k: (json.dumps(clean(v)) if isinstance(v, (dict, list)) else v) for k, v in r.items()
                          if k not in ("thresholds", "a_strata")} for r in res])
    wcsv(flat, "state_summary.csv")
    guarded("fig_fractions", fig_fractions, DF, res)
    guarded("fig_isg", fig_isg, DF, res)
    guarded("fig_effects", fig_effects, res)
    guarded("fig_or", fig_or, res)
    guarded("write_md", write_md, res, meta)
    save_results()
    log(f"done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
