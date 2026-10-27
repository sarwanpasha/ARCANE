#!/usr/bin/env python
"""ARCANE unmatched-cell flags in COVID-19 blood (Stephenson et al. 2021), tag covid_stephenson.

Question: does the ARCANE "unknown" flag carry disease biology that a type-I interferon (ISG) score alone could not
produce? In COVID-19 the ISG response peaks in mild/moderate disease and falls in severe/critical disease, whereas
known severity biology (emergency myelopoiesis, HLA-DR-low/MS1 monocytes, plasmablasts, platelet aggregates,
glucocorticoid exposure) rises with severity. This dissociation is the test.

Sections (each wrapped in try/except; failures are logged in results.json["errors"] and RESULTS.md):
  Q0  load: cells npz for every seed, obs (ALL columns and levels printed to obs_columns.txt), column detection,
      severity mapping (healthy query / asymptomatic / mild / moderate / severe / critical; non-COVID respiratory
      separately; LPS-stimulated samples excluded), sample and donor tables (primary sample = earliest per donor)
  Q1  donor flag rate (p_unknown_ctrl<=alpha; unknown_bonf; composition-standardized ctrl) by severity: group means
      with donor-bootstrap CIs, Spearman over ordered severity (bootstrap CI), Jonckheere-Terpstra (permutation p),
      OLS HC3 adjusted for site, age, sex (+ancestry if present), leave-one-site-out and within-site trends,
      non-COVID respiratory controls vs healthy and vs COVID
  Q2  donor ISG score (type-adjusted z vs healthy query cells): ISG by severity (peak), flag ~ severity + ISG
      (+ GC treatment proxy + IEG dissociation + depth), paired bootstrap of rho(flag, severity) - rho(ISG, severity),
      an ISG-only detector calibrated like p_unknown_ctrl, flag rate among ISG-low cells
  Q3  per fine type flag rates by severity, per-type trend (BH), decomposition of the severe-minus-healthy excess
      into composition and within-type rate parts per type
  Q4  severe/critical flagged vs kept cells of the same type: donor-paired pseudobulk DE, hypergeometric enrichment,
      per-donor gene-set score deltas (flagged - kept) in severe vs mild/moderate vs healthy (MS1, HLA class II,
      plasmablast, platelet, ISG, glucocorticoid, cell cycle, IEG, ...), ISG-low flagged cells
  Q5  cross-disease: SLE-derived programs (optional <root>/summary/lupusprog/programs.json {name: [genes]}) scored in
      COVID flagged vs kept cells; COVID flagged programs derived here (written to covid_programs.json) scored in
      SLE flagged vs kept cells (if the SLE tag is available); gene overlap of the program sets
  Q6  repeat samples: per-donor trajectories of flag rate and ISG over Days_from_onset; paired first-to-last change,
      within-donor association, persistence of flags after ISG resolution
  Gates G1-G7 and a verdict are written to RESULTS.md.

Run from the project directory (~/arcane):
  python covid_deep.py [--root .] [--tag covid_stephenson] [--seeds 0,1,2] [--out summary/coviddeep]
                       [--lupus_programs summary/lupusprog/programs.json] [--sle_tag lupusfull] [--no_sle]
Inputs:  <root>/data/std/<tag>.h5ad (raw counts; falls back to <tag>.expr.npz + <tag>.obs.csv stand-in),
         <root>/res2/<tag>_seed<s>_cells.npz, <root>/data/std/<tag>_donors.csv (optional), <root>/genesets.json
Outputs: <root>/<out>/ : RESULTS.md (<8 KB), results.json, obs_columns.txt, *.csv, *.png, covid_programs.json, log.txt
Dependencies: numpy, pandas, scipy, matplotlib (+ anndata for .h5ad). OLS with HC3 is implemented here.
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
ap.add_argument("--tag", default="covid_stephenson")
ap.add_argument("--res_dir", default="res2")
ap.add_argument("--seeds", default=None, help="comma list; default all found")
ap.add_argument("--h5ad", default=None)
ap.add_argument("--donors_csv", default=None, help="default <root>/data/std/<tag>_donors.csv")
ap.add_argument("--genesets", default="genesets.json")
ap.add_argument("--out", default="summary/coviddeep")
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--nperm", type=int, default=2000)
ap.add_argument("--min_cells_type", type=int, default=10, help="min cells of a type in a donor for its rate")
ap.add_argument("--min_cells_pb", type=int, default=10, help="min flagged and kept cells per donor x type (pseudobulk)")
ap.add_argument("--min_cells_delta", type=int, default=5, help="min flagged and kept cells per donor x type (score deltas)")
ap.add_argument("--min_donors_pb", type=int, default=5)
ap.add_argument("--max_per_group", type=int, default=300)
ap.add_argument("--max_expr_per_donor", type=int, default=2500, help="cells per sample for expression (0=all)")
ap.add_argument("--chunk", type=int, default=50000)
ap.add_argument("--lupus_programs", default="summary/lupusprog/programs.json")
ap.add_argument("--sle_tag", default="lupusfull")
ap.add_argument("--sle_h5ad", default=None)
ap.add_argument("--sle_max_per_donor", type=int, default=800)
ap.add_argument("--no_sle", action="store_true")
ap.add_argument("--isg_key", default=None)
ap.add_argument("--gc_key", default=None)
ap.add_argument("--ieg_key", default=None)
ap.add_argument("--isg_low_q", type=float, default=0.9, help="ISG-low cell: isg_z <= healthy q of its type")
args = ap.parse_args()

ROOT = args.root
OUT = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
os.makedirs(OUT, exist_ok=True)
LOGF = open(os.path.join(OUT, "log.txt"), "w")
T0 = time.time()
R = {"args": vars(args), "errors": {}, "timing_s": {}}
ISG_DOMINATED = set()
SEV_ORDER = ["healthy", "asymptomatic", "mild", "moderate", "severe", "critical"]
SEV_ORD = {s: i for i, s in enumerate(SEV_ORDER)}
TECH_GENE = r"^(MT-|RPL|RPS|MRPL|MRPS|HB[ABDEGMQZ]\d?$|IG[HKL][VDJC]|IGH[GAMDE]|JCHAIN$|MALAT1$|NEAT1$|XIST$|RPS4Y|DDX3Y|UTY$|KDM5D$|EIF1AY$)"
TECH_SET = r"ISG|IFN|interferon|SIGLEC1|dissoc|IEG|heat|mitochond|ribosom|erythro|hypoxia"
KNOWN_COVID = (r"MS1|HLA_class|plasmablast|platelet|glucocort|GC_induced|cell_cycle|LDG|neutrophil|ISG|IFN|SIGLEC1|dissoc|"
               r"IEG|heat|DC3|CD14pos|NFkB|erythro|ribosom|mitochond|hypoxia")  # known COVID states + technical sets
KEY_SETS = ["MS1_immunosuppressive_monocyte", "HLA_classII", "plasmablast", "platelet", "ISG_typeI",
            "glucocorticoid_core", "GC_induced_monocyte", "cell_cycle_S", "cell_cycle_G2M", "dissociation_IEG",
            "IEG_core_specific", "heat_shock", "LDG_immature_neutrophil", "NFkB_TNF_noIEG", "hypoxia"]
# minimal fallbacks if genesets.json lacks a key needed by the questions
FALLBACK_SETS = {
    "MS1_immunosuppressive_monocyte": ["RETN", "ALOX5AP", "IL1R2", "CD163", "VCAN", "S100A12", "CYP1B1", "PLBD1",
                                      "MCEMP1", "CLU", "SLC39A8", "HP", "ARG1", "MMP9"],
    "HLA_classII": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "HLA-DMA", "HLA-DMB", "CD74"],
    "plasmablast": ["MZB1", "JCHAIN", "XBP1", "PRDM1", "TNFRSF17", "SDC1", "IGHG1", "TXNDC5", "DERL3", "CD38"],
    "platelet": ["PPBP", "PF4", "GP9", "ITGA2B", "GNG11", "CAVIN2", "SDPR", "TUBB1", "CLU", "SPARC"],
    "ISG_typeI": ["IFI27", "IFI44L", "IFIT1", "IFIT3", "ISG15", "MX1", "RSAD2", "OAS1", "OAS3", "SIGLEC1", "IFI6",
                  "HERC5", "EPSTI1", "LY6E", "XAF1"],
    "glucocorticoid_core": ["TSC22D3", "FKBP5", "ZBTB16", "DDIT4", "PER1", "KLF9", "IRS2", "CD163", "ADORA3", "VSIG4"],
    "cell_cycle_S": ["MCM5", "PCNA", "TYMS", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7"],
    "cell_cycle_G2M": ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "MKI67", "CENPF", "CCNB2"],
    "dissociation_IEG": ["FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "IER2", "IER3", "DUSP1", "ZFP36", "ATF3",
                         "HSPA1A", "HSPA1B", "NR4A1"],
}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n")
    LOGF.flush()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
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
    with open(os.path.join(OUT, "results.json"), "w") as f:
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


plt.rcParams.update({"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200})
SEV_COL = {"healthy": "#4A7FB0", "asymptomatic": "#8FB3C9", "mild": "#E3B04B", "moderate": "#E08A3C",
           "severe": "#C8553D", "critical": "#7A1F1F", "noncovid": "#6E9E6E"}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def wcsv(df, name, index=False):
    df.to_csv(os.path.join(OUT, name), index=index)


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


def rank_rho(x, y):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


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
        bs.append(rank_rho(x[i], y[i]))
    lo, hi = ci(bs)
    return dict(n=int(n), rho=float(rho), p=float(p), ci_lo=lo, ci_hi=hi)


def spearman_cluster_boot(x, y, groups, nboot=None, seed=13):
    """Spearman rho with a cluster (donor) bootstrap CI: rows are repeated samples nested in donors, so resampling
    rows would treat samples of one donor as independent (pseudoreplication)."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups).astype(str)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[m], y[m], groups[m]
    ug = np.unique(groups)
    if len(x) < 5 or len(ug) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(n=int(len(x)), n_donors=int(len(ug)), rho=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot or args.nboot):
        i = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        bs.append(rank_rho(x[i], y[i]))
    lo, hi = ci(bs)
    return dict(n=int(len(x)), n_donors=int(len(ug)), rho=rank_rho(x, y), ci_lo=lo, ci_hi=hi)


def rho_diff_boot(x, y1, y2, nboot=None, seed=3):
    """paired donor bootstrap of rho(x, y1) - rho(x, y2)"""
    x, y1, y2 = (np.asarray(v, float) for v in (x, y1, y2))
    m = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    x, y1, y2 = x[m], y1[m], y2[m]
    n = len(x)
    if n < 6:
        return dict(n=int(n), diff=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot or args.nboot):
        i = rng.integers(0, n, n)
        bs.append(rank_rho(x[i], y1[i]) - rank_rho(x[i], y2[i]))
    lo, hi = ci(bs)
    r1, r2 = rank_rho(x, y1), rank_rho(x, y2)
    return dict(n=int(n), rho1=r1, rho2=r2, diff=r1 - r2, ci_lo=lo, ci_hi=hi,
                p_boot=float(2 * min(np.mean(np.asarray(bs) <= 0), np.mean(np.asarray(bs) >= 0))))


def boot_mean_ci(v, nboot=None, seed=2):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return dict(n=int(len(v)), mean=float(np.mean(v)) if len(v) else np.nan, median=np.nan, ci_lo=np.nan,
                    ci_hi=np.nan, p_wilcoxon=np.nan, frac_pos=np.nan)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (nboot or args.nboot, len(v)))].mean(1)
    lo, hi = ci(bs)
    try:
        pw = float(stats.wilcoxon(v).pvalue)
    except Exception:  # noqa: BLE001
        pw = np.nan
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), ci_lo=lo, ci_hi=hi,
                p_wilcoxon=pw, frac_pos=float((v > 0).mean()))


def diff_boot(va, vb, nboot=None, seed=7):
    """mean(a) - mean(b), stratified donor bootstrap"""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
    if len(va) < 3 or len(vb) < 3:
        return dict(n_a=int(len(va)), n_b=int(len(vb)), mean_a=np.nan, mean_b=np.nan, diff=np.nan, ci_lo=np.nan,
                    ci_hi=np.nan, p_mwu=np.nan)
    rng = np.random.default_rng(seed)
    B = nboot or args.nboot
    bs = va[rng.integers(0, len(va), (B, len(va)))].mean(1) - vb[rng.integers(0, len(vb), (B, len(vb)))].mean(1)
    lo, hi = ci(bs)
    return dict(n_a=int(len(va)), n_b=int(len(vb)), mean_a=float(va.mean()), mean_b=float(vb.mean()),
                diff=float(va.mean() - vb.mean()), ci_lo=lo, ci_hi=hi, p_mwu=mwu_p(va, vb))


def jt_stat(x, g):
    """Jonckheere-Terpstra statistic for ordered integer groups g"""
    lev = np.unique(g)
    S = 0.0
    for i, a in enumerate(lev):
        xa = x[g == a]
        for b in lev[i + 1:]:
            xb = x[g == b]
            S += (xb[:, None] > xa[None, :]).sum() + 0.5 * (xb[:, None] == xa[None, :]).sum()
    return S


def jonckheere(x, g, nperm=None, seed=5):
    x, g = np.asarray(x, float), np.asarray(g, float)
    m = np.isfinite(x) & np.isfinite(g)
    x, g = x[m], g[m].astype(int)
    if len(np.unique(g)) < 2 or len(x) < 6:
        return dict(n=int(len(x)), J=np.nan, z=np.nan, p_perm=np.nan)
    J = jt_stat(x, g)
    rng = np.random.default_rng(seed)
    B = nperm or args.nperm
    null = np.array([jt_stat(x, rng.permutation(g)) for _ in range(B)])
    mu, sd = null.mean(), null.std()
    p = (1 + np.sum(np.abs(null - mu) >= abs(J - mu))) / (B + 1)
    return dict(n=int(len(x)), J=float(J), z=float((J - mu) / sd) if sd > 0 else np.nan, p_perm=float(p))


def collapse_levels(s, min_n=3, na="unknown"):
    s = s.astype(object).where(s.notna(), na).astype(str)
    s = s.where(~s.str.lower().isin(["nan", "none", "", "na", "unknown", "<na>"]), na)
    vc = s.value_counts()
    rare = vc[vc < min_n].index
    return s.where(~s.isin(rare), "other")


def zs(v):
    v = np.asarray(v, float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / (sd if sd > 0 else 1.0)


def ols_robust(df, y, cats=(), nums=(), raw=(), refs=None, cluster=None, min_level=3):
    """OLS with HC3 (or CR1 cluster-robust). nums are z-scored, raw are used as is (e.g. ordinal severity, 0/1).
    Rows with missing y / nums / raw are dropped; missing categorical values become their own level."""
    refs = refs or {}
    cols = [y] + list(cats) + list(nums) + list(raw) + ([cluster] if cluster else [])
    d = df[[c for c in dict.fromkeys(cols) if c in df]].copy()
    d = d.dropna(subset=[c for c in [y] + list(nums) + list(raw) if c in d])
    Xc = [pd.Series(1.0, index=d.index, name="Intercept")]
    used = []
    for c in raw:
        if c in d and np.nanstd(d[c].astype(float)) > 0:
            Xc.append(pd.Series(d[c].astype(float).values, index=d.index, name=c))
            used.append(c)
    for c in nums:
        if c in d and np.nanstd(d[c].astype(float)) > 0:
            Xc.append(pd.Series(zs(d[c].astype(float).values), index=d.index, name=c))
            used.append(c)
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
    X = pd.concat(Xc, axis=1)
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


def term_row(T, term):
    r = T[T.term == term]
    if not len(r):
        return dict(coef=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    r = r.iloc[0]
    return dict(coef=float(r.coef), ci_lo=float(r.ci_lo), ci_hi=float(r.ci_hi), p=float(r.p))


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
    """numbers, '54-year-old human stage', '(50, 59]', '50-59', '80+' -> float"""
    if s is None:
        return np.nan
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    t = str(s)
    dec = re.search(r"(second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+decade", t, re.I)
    if dec:
        return {"second": 15, "third": 25, "fourth": 35, "fifth": 45, "sixth": 55, "seventh": 65, "eighth": 75,
                "ninth": 85, "tenth": 95}[dec.group(1).lower()] * 1.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*year", t)
    if m:
        return float(m.group(1))
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    if not nums:
        return np.nan
    if len(nums) >= 2:
        return float(np.mean(nums[:2]))
    return nums[0] + (5.0 if "+" in t else 0.0)


def parse_num(s):
    try:
        v = float(str(s).strip())
        return v if np.isfinite(v) else np.nan
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", str(s))
        return float(m.group(0)) if m else np.nan


def norm_sex(s):
    t = str(s).strip().lower()
    if t in ("f", "female", "woman", "w"):
        return "female"
    if t in ("m", "male", "man"):
        return "male"
    return "unknown"


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


def load_cells(tag, seeds=None):
    rd = os.path.join(ROOT, args.res_dir)
    files = sorted(glob.glob(os.path.join(rd, f"{tag}_seed*_cells.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(ROOT, "res", f"{tag}_seed*_cells.npz")))
    if not files:
        raise FileNotFoundError(f"no cells npz for {tag} in {rd} or res/")
    base, used = None, []
    for f in files:
        m = re.search(r"_seed(\d+)_cells\.npz$", f)
        if not m:
            continue
        s = int(m.group(1))
        if seeds is not None and s not in seeds:
            continue
        C = np.load(f, allow_pickle=True)
        names = np.asarray(C["names"]).astype(str)
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
        for k, v in fl.items():
            df[f"f_{k}_s{s}"] = v.astype(np.float32)
        log(f"{tag} seed {s}: {len(df)} cells; " + ", ".join(f"{k}={v.mean():.4f}" for k, v in fl.items()))
        base = df if base is None else base.join(df, how="inner")
        used.append(s)
    if base is None:
        raise FileNotFoundError(f"no requested seeds for {tag}")
    if base.index.duplicated().any():
        log(f"WARNING: {base.index.duplicated().sum()} duplicated cell names; keeping first")
        base = base[~base.index.duplicated()]
    for k in ["ctrl", "bonf", "pool"]:
        cs = [f"f_{k}_s{s}" for s in used if f"f_{k}_s{s}" in base]
        if cs:
            base[f"f_{k}"] = base[cs].mean(1)
    base["novelty"] = base[[f"novelty_s{s}" for s in used]].mean(1)
    return base, sorted(used)


class Matrix:
    """row access to raw counts in an .h5ad (anndata, backed when possible) or the numpy stand-in"""

    def __init__(self, tag, h5=None):
        h5 = h5 or os.path.join(ROOT, "data", "std", f"{tag}.h5ad")
        stand = re.sub(r"\.h5ad$", "", h5) + ".expr.npz"
        self.A, self.Xa = None, None
        if os.path.exists(h5):
            import anndata as ad
            try:
                self.A = ad.read_h5ad(h5, backed="r")
                self.backed = True
            except Exception as e:  # noqa: BLE001
                log(f"backed read failed ({e!r}); loading fully")
                self.A = ad.read_h5ad(h5)
                self.backed = False
            self.h5 = h5
            fmt = getattr(self.A.X, "format", None) or getattr(self.A.X, "format_str", None)
            if self.backed and fmt == "csc":
                log("backed X is CSC (row access would read the whole matrix per chunk); loading into memory as CSR")
                self.A = ad.read_h5ad(h5)
                self.backed = False
                if sp.issparse(self.A.X) and not sp.isspmatrix_csr(self.A.X):
                    self.A.X = sp.csr_matrix(self.A.X)
            obs = self.A.obs.copy()
            self.names = pd.Index(self.A.obs_names.astype(str))
            self.var = pd.Index(self.A.var_names.astype(str))
            log(f"opened {h5}: {len(self.names)} cells x {len(self.var)} genes")
        elif os.path.exists(stand):
            log("h5ad not found; using numpy stand-in", stand)
            Z = np.load(stand, allow_pickle=True)
            self.Xa = sp.csr_matrix((Z["data"], Z["indices"], Z["indptr"]), shape=tuple(Z["shape"]))
            self.names = pd.Index(np.asarray(Z["obs_names"]).astype(str))
            self.var = pd.Index(np.asarray(Z["var_names"]).astype(str))
            obs = pd.read_csv(stand.replace(".expr.npz", ".obs.csv"), index_col=0, dtype=str, keep_default_na=False)
        else:
            raise FileNotFoundError(f"neither {h5} nor {stand}")
        for c in obs.columns:
            obs[c] = obs[c].astype(str)
        obs.index = self.names
        self.obs = obs

    def rows(self, cells):
        pos = self.names.get_indexer(cells)
        if (pos < 0).any():
            raise KeyError(f"{(pos < 0).sum()} cells not in matrix, e.g. {np.asarray(cells)[pos < 0][:3]}")
        if self.Xa is not None:
            X, tot = normalize_rows(self.Xa[pos])
            return X, tot
        order = np.argsort(pos)
        sp_pos = pos[order]
        chunks, tots = [], []
        for i in range(0, len(sp_pos), args.chunk):
            idx = sp_pos[i:i + args.chunk]
            try:
                Xi = self.A.X[idx]
            except Exception:  # noqa: BLE001
                if self.backed:
                    import anndata as ad
                    log("backed row indexing failed; loading h5ad into memory")
                    self.A = ad.read_h5ad(self.h5)
                    self.backed = False
                Xi = self.A.X[idx]
            Xi, tot = normalize_rows(Xi)
            chunks.append(Xi)
            tots.append(tot)
            log(f"  expression rows {i + len(idx)}/{len(sp_pos)}")
        X = sp.vstack(chunks).tocsr()
        tot = np.concatenate(tots)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return X[inv], tot[inv]


def normalize_rows(X):
    X = sp.csr_matrix(X, dtype=np.float32)
    X.eliminate_zeros()
    samp = X.data[:200000]
    if len(samp) and not np.allclose(samp, np.round(samp)):
        log("WARNING: X is not integer counts; assuming already log-normalized")
        return X, np.asarray(np.expm1(X).sum(1)).ravel()
    tot = np.asarray(X.sum(1)).ravel()
    X.data *= np.repeat((1e4 / np.maximum(tot, 1)).astype(np.float32), np.diff(X.indptr))
    np.log1p(X.data, out=X.data)
    return X, tot


def load_genesets():
    cands = [args.genesets if os.path.isabs(args.genesets) else os.path.join(ROOT, args.genesets),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "genesets.json")]
    gs = {}
    for p in cands:
        if os.path.exists(p):
            gs = json.load(open(p))
            gs = {str(k): [str(g) for g in v] for k, v in gs.items() if isinstance(v, (list, tuple))}
            log("gene sets:", p, len(gs))
            break
    else:
        log("WARNING: genesets.json not found; using built-in fallback sets only")
    for k, v in FALLBACK_SETS.items():
        if k not in gs:
            gs[k] = v
            log(f"  fallback set added: {k}")
    return gs


def detect_key(gs, user, pats):
    if user:
        return user if user in gs else None
    for p in pats:
        for k in gs:
            if re.search(p, k, re.I):
                return k
    return None


def score_sets(X, var, sets, rng, n_bins=25, ctrl_size=50, min_det=20):
    """scanpy score_genes algorithm (expression-binned control genes) for many sets with one sparse product."""
    det = np.bincount(X.indices, minlength=X.shape[1])
    ok = det >= min(min_det, max(3, X.shape[0] // 200))
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
    v = pd.Series(np.asarray(v, float))
    y = pd.Series(np.asarray(y))
    h = np.asarray(healthy, bool)
    out = np.full(len(v), np.nan)
    gmu, gsd = np.nanmean(v[h]) if h.any() else np.nanmean(v), np.nanstd(v[h]) if h.any() else np.nanstd(v)
    for t in pd.unique(y):
        m = (y == t).values
        hv = v[m & h].dropna()
        mu, sd = (hv.mean(), hv.std()) if len(hv) >= min_n else (gmu, gsd)
        out[m] = (v[m] - mu) / (sd if sd > 0 else 1.0)
    return out


def lodo_thresholds(v, y, donor, healthy, q, min_n=20):
    """per-cell threshold: q-quantile of healthy cells of the same type (global healthy quantile if < min_n);
    for healthy cells the donor's own cells are left out."""
    v = np.asarray(v, float)
    y, donor, h = np.asarray(y).astype(str), np.asarray(donor).astype(str), np.asarray(healthy, bool)
    out = np.full(len(v), np.nan)
    gq = np.nanquantile(v[h], q) if h.any() else np.nan
    hall, dall = v[h], donor[h]
    for t in np.unique(y):
        m = np.where(y == t)[0]
        vt, ht, dt = v[m], h[m], donor[m]
        hv, hd = vt[ht], dt[ht]
        thr = np.full(len(m), np.nanquantile(hv, q) if np.isfinite(hv).sum() >= min_n else gq)
        for d in np.unique(hd):
            rest = hv[hd != d]
            if np.isfinite(rest).sum() >= min_n:
                thr[ht & (dt == d)] = np.nanquantile(rest, q)
            else:
                rg = hall[dall != d]
                thr[ht & (dt == d)] = np.nanquantile(rg, q) if np.isfinite(rg).sum() else gq
        out[m] = thr
    return out


def std_rates(D, col, w, min_cells, by="sample"):
    d = D[[by, "y", col]].dropna()
    g = d.groupby([by, "y"])[col].agg(["sum", "size"])
    N = g["size"].unstack(fill_value=0)
    S = g["sum"].unstack(fill_value=0.0)
    Rt = S / N.where(N >= min_cells)
    W = pd.Series(w).reindex(Rt.columns).fillna(0).values[None, :]
    valid = Rt.notna().values
    num = np.nansum(Rt.values * W, 1)
    den = (W * valid).sum(1)
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=Rt.index)


def lineage_of(label, parent=""):
    t = f"{parent} {label}".lower()
    if re.search(r"platelet|megak|\bplt\b", t):
        return "platelet"
    if re.search(r"eryth|rbc|\bhsc|progen|cd34", t):
        return "other"
    if re.search(r"plasma|\bpb\b|plasmablast", t):
        return "B"
    if re.search(r"mono|\bcm\b|\bncm|cd14|cd16_?mono|\bdc|cdc|pdc|dendrit|macro|myeloid", t):
        return "myeloid"
    if re.search(r"\bb[_ ]|b_?cell|\bb$|bnaive|bmem|b_naive|b_mem|b_atyp|b_switch|b_exh", t):
        return "B"
    if re.search(r"\bnk|nk_|nkt|ilc", t):
        return "NK"
    if re.search(r"cd4|cd8|\bt[_ ]|t_?cell|treg|mait|gdt|gd_t|\bt\d|tfh|th\d", t):
        return "T"
    return "other"


def read_programs(path):
    if not os.path.exists(path):
        return None
    P = json.load(open(path))
    if isinstance(P, dict) and "programs" in P:
        P = P["programs"]
    out = {}
    if isinstance(P, dict):
        for k, v in P.items():
            if isinstance(v, dict):
                v = v.get("genes") or v.get("top_genes") or []
            if isinstance(v, (list, tuple)):
                out[str(k)] = [str(g) for g in v]
    elif isinstance(P, list):
        for i, v in enumerate(P):
            if isinstance(v, dict):
                out[str(v.get("name", f"prog{i}"))] = [str(g) for g in (v.get("genes") or [])]
    return {k: v for k, v in out.items() if len(v) >= 3}


# ----------------------------------------------------------------------------------------------
# Q0
# ----------------------------------------------------------------------------------------------
@section("Q0_load")
def load_all(gs):
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    D, seeds = load_cells(args.tag, seeds)
    H = Matrix(args.tag, args.h5ad)
    if H.names.duplicated().any():
        raise ValueError(f"{H.names.duplicated().sum()} duplicated obs_names in the h5ad: npz<->h5ad join by name is "
                         "ambiguous (make obs_names unique the same way the npz writer did)")
    inobs = D.index.isin(H.names)
    miss = int((~inobs).sum())
    R["join"] = dict(npz_cells=int(len(D)), matched=int(inobs.sum()), missing=miss)
    if miss:
        log(f"WARNING: {miss}/{len(D)} npz cells not in h5ad obs (dropped)")
        if inobs.mean() < 0.9:
            raise ValueError(f"only {inobs.mean():.1%} of npz cells found in h5ad obs_names: wrong file or name scheme")
        D = D[inobs].copy()
    obs = H.obs.reindex(D.index)
    # alignment sanity: npz donor (batch) must agree with the h5ad donor column for the same cell name
    for bc in ["batch", "donor_id", "patient_id"]:
        if bc in obs.columns:
            agree = float((obs[bc].astype(str).values == D.donor.values).mean())
            R["join"][f"donor_agree_{bc}"] = agree
            log(f"npz batch vs obs[{bc}] agreement: {agree:.4f}")
            if bc == "batch" and agree < 0.99:
                raise ValueError(f"npz batch disagrees with h5ad obs.batch for {1 - agree:.1%} of cells: misaligned join")
            break
    # print every obs column with levels
    lines = []
    for c in obs.columns:
        vc = obs[c].value_counts(dropna=False)
        lev = "; ".join(f"{k}={v}" for k, v in vc.head(15).items())
        lines.append(f"{c} [n_levels={len(vc)}]: {lev}{' ...' if len(vc) > 15 else ''}")
    open(os.path.join(OUT, "obs_columns.txt"), "w").write("\n".join(lines) + "\n")
    log("obs columns (query cells):")
    for ln in lines:
        log("  " + ln[:300])
    cols = list(obs.columns)
    det = dict(
        severity_std=find_col(cols, ["severity_std"]),
        severity_summary=find_col(cols, ["Status_on_day_collection_summary"], [r"status_on_day.*summary"]),
        severity_day=find_col(cols, ["Status_on_day_collection"], [r"status_on_day"]),
        status=find_col(cols, ["Status", "disease", "condition"], [r"^status$"]),
        worst=find_col(cols, ["Worst_Clinical_Status", "worst_status"], [r"worst"]),
        site=find_col(cols, ["site_std", "Site", "site"], [r"^site", r"centre|center"]),
        age=find_col(cols, ["age_std", "Age_interval", "age", "Age"], [r"^age"]),
        dev_stage=find_col(cols, ["development_stage"], [r"development_stage"]),
        sex=find_col(cols, ["sex_std", "Sex", "sex", "gender"], [r"^sex", r"gender"]),
        days=find_col(cols, ["Days_from_onset", "days_from_onset", "days_std"], [r"days.*onset", r"onset"]),
        collection=find_col(cols, ["Collection_Day", "collection_day"], [r"collection_day"]),
        sample=find_col(cols, ["sample_id", "sample", "sample_std"], [r"^sample"]),
        patient=find_col(cols, ["patient_id", "donor_id"], [r"patient"]),
        ancestry=find_col(cols, ["self_reported_ethnicity", "ethnicity", "ancestry"], [r"ethnic", r"ancestr"]),
        smoker=find_col(cols, ["Smoker"], [r"smok"]),
        outcome=find_col(cols, ["Outcome"], [r"^outcome"]),
        lps=find_col(cols, ["time_after_LPS"], [r"lps"]),
        parent=find_col(cols, ["parent"]),
        role=find_col(cols, ["role"]),
    )
    R["columns"] = det
    log("detected columns:", det)
    # severity: pick the candidate column with most cells mapped to the ordered scale
    best, best_n = None, -1
    for k in ["severity_std", "severity_summary", "severity_day", "status"]:
        c = det.get(k)
        if not c:
            continue
        mp = obs[c].map(map_severity)
        n_ok = int(mp.isin(SEV_ORDER + ["noncovid", "LPS"]).sum())
        log(f"severity candidate {c}: mapped {n_ok}/{len(mp)}; {mp.value_counts().to_dict()}")
        if n_ok > best_n:
            best, best_n = c, n_ok
    R["severity_column"] = best
    sev = obs[best].map(map_severity) if best else pd.Series("na", index=obs.index)
    # fill unknown severity from condition (normal -> healthy)
    norm = D.cond_raw.str.lower().str.contains(r"normal|healthy|control")
    sev = sev.where(~(sev.isin(["na", "other"]) & norm), "healthy")
    # LPS samples: any candidate severity/status column saying LPS (severity_std may hide it), or the LPS column
    for k in ["severity_std", "severity_summary", "severity_day", "status", "collection"]:
        c = det.get(k)
        if c:
            sev = sev.where(~obs[c].astype(str).str.contains("lps", case=False).values, "LPS")
    if det.get("lps"):
        lv = obs[det["lps"]].astype(str).str.lower()
        islps = ~lv.isin(["", "nan", "none", "na", "healthy", "0", "unknown", "not applicable", "<na>"])
        islps &= lv.str.contains(r"\d|lps|min|hour")
        sev = sev.where(~islps, "LPS")
    D["sev"] = sev.values
    D["sev_raw"] = obs[best].values if best else "na"
    xt = pd.crosstab(D["sev_raw"], D["sev"])
    wcsv(xt, "Q0_severity_mapping.csv", index=True)
    R["severity_mapping"] = {str(r): {str(c): int(v) for c, v in xt.loc[r].items() if v} for r in xt.index}
    D["sample"] = obs[det["sample"]].values if det["sample"] else D.donor.values
    if det["sample"] and obs[det["sample"]].nunique() < D.donor.nunique():
        log("sample column has fewer levels than donors; using donor x sample")
        D["sample"] = D.donor + "|" + D["sample"].astype(str)
    if det["collection"] and not det["sample"]:
        D["sample"] = D.donor + "|" + obs[det["collection"]].astype(str).values
    for k in ["site", "sex", "ancestry", "worst", "outcome", "smoker", "collection", "patient"]:
        if det.get(k):
            D[k] = obs[det[k]].astype(str).values
    if "sex" in D:
        D["sex"] = D.sex.map(norm_sex)
    src = det.get("age") or det.get("dev_stage")
    D["age_num"] = obs[src].map(parse_age).values if src else np.nan
    if det.get("dev_stage") and np.isnan(D.age_num).mean() > 0.5:
        D["age_num"] = obs[det["dev_stage"]].map(parse_age).values
    D["days"] = obs[det["days"]].map(parse_num).values if det["days"] else np.nan
    D["coll_num"] = obs[det["collection"]].map(parse_num).values if det.get("collection") else np.nan
    par = obs[det["parent"]].astype(str).values if det["parent"] else np.array([""] * len(D))
    D["parent"] = par
    pairs = pd.Series(list(zip(D.y.values, par)))
    lmap = {u: lineage_of(*u) for u in pd.unique(pairs)}
    D["lineage"] = pairs.map(lmap).values
    # donor csv merge (fills missing fields only)
    dcsv = args.donors_csv or os.path.join(ROOT, "data", "std", f"{args.tag}_donors.csv")
    if os.path.exists(dcsv):
        try:
            dd = pd.read_csv(dcsv, dtype=str)
            log(f"donor csv {dcsv}: columns {list(dd.columns)}")
            idc = max(dd.columns, key=lambda c: dd[c].isin(D.donor.unique()).sum())
            if dd[idc].isin(D.donor.unique()).sum() > 0:
                dd = dd.drop_duplicates(idc).set_index(idc)
                for k, pats in [("site", [r"site"]), ("sex", [r"sex|gender"]), ("age_num", [r"age"])]:
                    c = find_col(list(dd.columns), patterns=pats)
                    if not c:
                        continue
                    v = D.donor.map(dd[c])
                    if k == "age_num":
                        v = v.map(parse_age)
                        D[k] = D[k].where(np.isfinite(D[k]), v)
                    elif k == "sex":
                        v = v.map(norm_sex)
                        D[k] = D[k].where(D[k] != "unknown", v) if k in D else v
                    else:
                        D[k] = D[k].where(~D[k].isin(["nan", "", "unknown"]), v) if k in D else v
                R["donor_csv"] = dict(path=dcsv, id_col=idc, n=len(dd))
        except Exception as e:  # noqa: BLE001
            log(f"donor csv merge failed: {e!r}")
    R["n_raw"] = dict(cells=len(D), donors=int(D.donor.nunique()), samples=int(D["sample"].nunique()),
                      seeds=seeds, sev_cells=D.sev.value_counts().to_dict(),
                      sev_donors=D.groupby("sev").donor.nunique().to_dict())
    log(R["n_raw"])
    return D, H, seeds


@section("Q0_expression")
def expression(D, H, gs, keys, extra_sets):
    rng = np.random.default_rng(0)
    keep = []
    for _, g in D[D.sev != "LPS"].groupby("sample"):
        idx = g.index.values
        n = len(idx) if args.max_expr_per_donor <= 0 else min(len(idx), args.max_expr_per_donor)
        keep.append(rng.choice(idx, n, replace=False))
    cells = D.index[D.index.isin(np.concatenate(keep))].values
    log(f"expression cells: {len(cells)}")
    X, tot = H.rows(cells)
    var = H.var
    sets = dict(gs)
    sets.update(extra_sets)
    S, used = score_sets(X, var, sets, rng)
    S.index = pd.Index(cells)
    R["genesets_used_n"] = {k: len(v) for k, v in used.items()}
    E = S.add_prefix("gs_")
    E["log_counts"] = np.log10(np.maximum(tot, 1))
    healthy = (D.loc[cells, "sev"] == "healthy").values
    y = D.loc[cells, "y"].values
    for c in list(E.columns):
        if c.startswith("gs_"):
            E["z_" + c[3:]] = z_by_type(E[c].values, y, healthy)
    for nm, k in keys.items():
        if k and "z_" + k in E:
            E[nm + "_z"] = E["z_" + k]
    if "isg_z" in E:
        donor = D.loc[cells, "donor"].values
        # thresholds from healthy cells of the type; healthy donors get leave-one-donor-out thresholds, matching how
        # p_unknown_ctrl is calibrated (in-sample thresholds would deflate the healthy rate of the ISG detector)
        E["isgdet"] = (E.isg_z.values > lodo_thresholds(E.isg_z.values, y, donor, healthy, 1 - args.alpha)).astype(float)
        E["isg_low"] = (E.isg_z.values <= lodo_thresholds(E.isg_z.values, y, donor, healthy, args.isg_low_q)).astype(float)
    rowpos = pd.Series(np.arange(len(cells)), index=cells)
    return X, var, E, rowpos, used


# ----------------------------------------------------------------------------------------------
# sample / donor tables
# ----------------------------------------------------------------------------------------------
@section("Q0_tables")
def build_tables(D, E):
    D = D[D.sev != "LPS"].copy()
    h = D[(D.sev == "healthy")]
    w = h.y.value_counts(normalize=True).to_dict()
    R["healthy_composition"] = w
    DE = D.join(E, how="left")
    g = DE.groupby("sample")
    ST = pd.DataFrame(dict(donor=g.donor.agg(lambda s: s.value_counts().index[0]),
                           sev=g.sev.agg(lambda s: s.value_counts().index[0]), n_cells=g.size()))
    for k in ["site", "sex", "ancestry", "worst", "outcome", "smoker", "collection"]:
        if k in D:
            ST[k] = g[k].agg(lambda s: s.value_counts().index[0])
    ST["age_num"] = g.age_num.median()
    ST["days"] = g.days.median()
    if "coll_num" in D:
        ST["coll_num"] = g.coll_num.median()
    for k in ["ctrl", "bonf"]:
        if f"f_{k}" in D:
            ST[f"rate_{k}"] = g[f"f_{k}"].mean() * 100
    if "f_ctrl" in D:
        ST["rate_ctrl_std"] = std_rates(D, "f_ctrl", w, args.min_cells_type, by="sample").reindex(ST.index) * 100
        # cell-level flagged fraction among ISG-low cells (expression subsample)
        if "isg_low" in DE:
            m = DE.isg_low == 1
            ST["rate_ctrl_isglow"] = DE[m].groupby("sample").f_ctrl.mean().reindex(ST.index) * 100
            ST["rate_ctrl_isglow_std"] = std_rates(DE[m], "f_ctrl", w, max(3, args.min_cells_type // 2),
                                                   by="sample").reindex(ST.index) * 100
    ST["novelty"] = g.novelty.mean()
    for c in ["isg_z", "gc_z", "ieg_z", "ifng_z", "log_counts", "isgdet"]:
        if c in DE:
            ST[c] = g[c].mean() * (100 if c == "isgdet" else 1)
    if "isg_z" in ST:
        ST["isg_z2"] = ST.isg_z ** 2
    ST["sev_ord"] = ST.sev.map(SEV_ORD).astype(float)
    ST["is_covid"] = ST.sev.isin(SEV_ORDER[1:]).astype(int)
    # primary sample per donor: earliest days, then largest
    S2 = ST.copy()
    S2["_d"] = time_key(S2)
    S2 = S2.sort_values(["donor", "_d", "n_cells"], ascending=[True, True, False])
    prim = S2.groupby("donor").head(1)
    DN = prim.reset_index().set_index("donor")
    DN["n_samples"] = ST.groupby("donor").size().reindex(DN.index)
    if "worst" in DN:
        DN["worst_sev"] = DN.worst.map(map_severity)
    wcsv(ST.reset_index(), "Q0_samples.csv")
    wcsv(DN.reset_index(), "Q0_donors.csv")
    R["n"] = dict(cells=int(len(D)), samples=int(len(ST)), donors=int(len(DN)),
                  donors_by_sev=DN.sev.value_counts().to_dict(),
                  donors_multi_sample=int((DN.n_samples > 1).sum()),
                  sites=DN.site.value_counts().to_dict() if "site" in DN else None)
    log(R["n"])
    try:
        R["design_crosstab_site_sev"] = clean(pd.crosstab(DN.site, DN.sev)) if "site" in DN else None
        if "site" in DN:
            wcsv(pd.crosstab(DN.site, DN.sev), "Q0_site_by_severity.csv", index=True)
    except Exception:  # noqa: BLE001
        pass
    Dp = D[D["sample"].isin(DN["sample"])].copy()
    return ST, DN, Dp, w


def time_key(ST):
    """within-donor sampling order: Collection_Day number when it is known for all of the donor's samples (it is the
    sampling order and is defined for asymptomatic donors), else days from onset; unknown -> last."""
    k = ST.days.astype(float).copy()
    if "coll_num" in ST:
        cn = ST.coll_num.astype(float)
        full = cn.notna().groupby(ST.donor).transform("all")
        k = k.where(~full.values, cn)
    return k.fillna(np.inf)


def group_table(DN, col):
    rows = []
    for s in SEV_ORDER + ["noncovid"]:
        v = DN.loc[DN.sev == s, col] if col in DN else pd.Series(dtype=float)
        b = boot_mean_ci(v.values)
        rows.append(dict(var=col, sev=s, n=b["n"], mean=b["mean"], median=b.get("median", np.nan),
                         ci_lo=b["ci_lo"], ci_hi=b["ci_hi"]))
    return pd.DataFrame(rows)


def covars(DN):
    cats = [c for c in ["site", "sex", "ancestry"] if c in DN and DN[c].nunique() > 1]
    if "ancestry" in cats and (DN.ancestry.str.lower().isin(["unknown", "nan", "na"]).mean() > 0.7):
        cats.remove("ancestry")
    nums = ["age_num"] if "age_num" in DN and DN.age_num.notna().mean() > 0.5 else []
    return cats, nums


# ----------------------------------------------------------------------------------------------
# Q1 severity trend
# ----------------------------------------------------------------------------------------------
@section("Q1_severity_trend")
def q1(DN):
    out = {}
    rates = [c for c in ["rate_ctrl", "rate_ctrl_std", "rate_bonf", "rate_ctrl_isglow", "rate_ctrl_isglow_std"] if c in DN]
    GT = pd.concat([group_table(DN, c) for c in rates + ["isg_z", "isgdet", "gc_z", "ieg_z"] if c in DN])
    wcsv(GT, "Q1_group_means.csv")
    out["groups"] = GT
    ordd = DN[DN.sev.isin(SEV_ORDER)]
    cov = ordd[ordd.sev != "healthy"]
    trend = []
    for c in rates + [x for x in ["isg_z", "isgdet", "gc_z", "ieg_z", "novelty"] if x in DN]:
        for nm, d in [("all", ordd), ("covid_only", cov)]:
            sb = spearman_boot(d.sev_ord, d[c])
            jt = jonckheere(d[c].values, d.sev_ord.values)
            trend.append(dict(var=c, set=nm, **sb, jt_z=jt["z"], jt_p=jt["p_perm"]))
    TT = pd.DataFrame(trend)
    wcsv(TT, "Q1_trend.csv")
    out["trend"] = TT
    cats, nums = covars(DN)
    out["covariates"] = dict(cats=cats, nums=nums)
    mods = []
    for c in rates:
        for nm, d in [("all", ordd), ("covid_only", cov)]:
            try:
                T, info = ols_robust(d, c, cats=cats, nums=nums, raw=["sev_ord"])
                mods.append(dict(var=c, set=nm, model="sev_ord+site+age+sex", **term_row(T, "sev_ord"),
                                 n=info["n"], se=info["se_type"]))
                T.to_csv(os.path.join(OUT, f"Q1_ols_{c}_{nm}.csv"), index=False)
            except Exception as e:  # noqa: BLE001
                log(f"OLS {c} {nm} failed: {e!r}")
        # categorical severity vs healthy
        try:
            d = ordd.copy()
            T, info = ols_robust(d, c, cats=["sev"] + cats, nums=nums, refs={"sev": "healthy"}, min_level=2)
            for s in SEV_ORDER[1:]:
                mods.append(dict(var=c, set="all", model=f"sev[{s}] vs healthy", **term_row(T, f"sev[T.{s}]"),
                                 n=info["n"], se=info["se_type"]))
        except Exception as e:  # noqa: BLE001
            log(f"OLS categorical {c} failed: {e!r}")
    M = pd.DataFrame(mods)
    wcsv(M, "Q1_ols_severity.csv")
    out["ols"] = M
    # leave-one-site-out and within-site trends
    if "site" in DN and DN.site.nunique() > 1:
        lo = []
        for s in sorted(ordd.site.unique()):
            d = ordd[ordd.site != s].dropna(subset=["rate_ctrl", "sev_ord"])
            w_ = ordd[ordd.site == s].dropna(subset=["rate_ctrl", "sev_ord"])
            lo.append(dict(site=s, n_left_out=len(w_), rho_without=rank_rho(d.sev_ord, d.rate_ctrl) if len(d) > 5 else np.nan,
                           rho_within=rank_rho(w_.sev_ord, w_.rate_ctrl) if len(w_) > 5 and w_.sev_ord.nunique() > 1 else np.nan,
                           n_within=len(w_)))
        LO = pd.DataFrame(lo)
        wcsv(LO, "Q1_site_robustness.csv")
        out["loso"] = LO
        wv = LO.dropna(subset=["rho_within"])
        out["within_site_weighted_rho"] = float(np.average(wv.rho_within, weights=wv.n_within)) if len(wv) else np.nan
    # non-COVID respiratory controls
    nc = DN[DN.sev == "noncovid"]
    if len(nc):
        out["noncovid"] = {c: dict(vs_healthy=diff_boot(nc[c], DN.loc[DN.sev == "healthy", c]),
                                   vs_covid_sevcrit=diff_boot(nc[c], DN.loc[DN.sev.isin(["severe", "critical"]), c]),
                                   vs_covid_all=diff_boot(nc[c], cov[c]))
                           for c in rates[:2] + [x for x in ["isg_z"] if x in DN]}
    # figure: flag rate and ISG by severity
    try:
        pv = [c for c in ["rate_ctrl", "isg_z"] if c in DN]
        fig, axs = plt.subplots(1, len(pv), figsize=(7.5 * len(pv), 5.5))
        axs = np.atleast_1d(axs)
        levs = SEV_ORDER + (["noncovid"] if len(nc) else [])
        rng = np.random.default_rng(0)
        for ax, c in zip(axs, pv):
            for i, s in enumerate(levs):
                v = DN.loc[DN.sev == s, c].dropna().values
                if not len(v):
                    continue
                ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=28, color=SEV_COL[s], alpha=.8)
                ax.hlines(np.median(v), i - .3, i + .3, color="k", lw=2.5)
            ax.set_xticks(range(len(levs)))
            ax.set_xticklabels([s[:5] if s != "asymptomatic" else "asym" for s in levs], rotation=30)
            ax.set_ylabel("flagged cells (%)" if c == "rate_ctrl" else "donor ISG (z)")
        savefig(fig, "Q1_rate_isg_by_severity.png")
    except Exception as e:  # noqa: BLE001
        log(f"Q1 figure failed: {e!r}")
    R["Q1"] = out
    return out


# ----------------------------------------------------------------------------------------------
# Q2 beyond ISG
# ----------------------------------------------------------------------------------------------
@section("Q2_beyond_ISG")
def q2(DN):
    out = {}
    if "isg_z" not in DN:
        raise ValueError("no ISG score")
    ordd = DN[DN.sev.isin(SEV_ORDER)]
    cov = ordd[ordd.sev != "healthy"]
    gm = ordd.groupby("sev").isg_z.mean().reindex(SEV_ORDER)
    out["isg_peak_group"] = str(gm.idxmax()) if gm.notna().any() else None
    out["n_ordered_donors"] = int(len(ordd))
    out["isg_by_sev"] = gm.to_dict()
    out["rate_by_sev"] = ordd.groupby("sev").rate_ctrl.mean().reindex(SEV_ORDER).to_dict()
    # rho differences within COVID (mild..critical) and all
    out["rho_diff"] = {}
    for nm, d in [("covid_only", cov), ("all", ordd)]:
        for rc in [c for c in ["rate_ctrl", "rate_ctrl_std", "rate_bonf"] if c in d]:
            out["rho_diff"][f"{rc}_vs_isg|{nm}"] = rho_diff_boot(d.sev_ord, d[rc], d.isg_z)
            if "isgdet" in d:
                out["rho_diff"][f"{rc}_vs_isgdet|{nm}"] = rho_diff_boot(d.sev_ord, d[rc], d.isgdet)
    # severe/critical minus mild contrast for ARCANE rate vs ISG-detector rate (in units of healthy SD)
    try:
        sc = cov[cov.sev.isin(["severe", "critical"])]
        mi = cov[cov.sev.isin(["mild", "moderate"])]
        hh = ordd[ordd.sev == "healthy"]
        rng = np.random.default_rng(11)
        bs = []
        for _ in range(args.nboot):
            a = sc.iloc[rng.integers(0, len(sc), len(sc))]
            b = mi.iloc[rng.integers(0, len(mi), len(mi))]
            h = hh.iloc[rng.integers(0, len(hh), len(hh))]
            sda, sdi = max(h.rate_ctrl.std(), 1e-6), max(h.isgdet.std(), 1e-6)
            bs.append((a.rate_ctrl.mean() - b.rate_ctrl.mean()) / sda - (a.isgdet.mean() - b.isgdet.mean()) / sdi)
        sda, sdi = max(hh.rate_ctrl.std(), 1e-6), max(hh.isgdet.std(), 1e-6)
        est = (sc.rate_ctrl.mean() - mi.rate_ctrl.mean()) / sda - (sc.isgdet.mean() - mi.isgdet.mean()) / sdi
        lo, hi = ci(bs)
        out["sevcrit_minus_mildmod_contrast"] = dict(
            arcane=float((sc.rate_ctrl.mean() - mi.rate_ctrl.mean()) / sda),
            isgdet=float((sc.isgdet.mean() - mi.isgdet.mean()) / sdi), diff=float(est), ci_lo=lo, ci_hi=hi,
            n_sevcrit=len(sc), n_mildmod=len(mi), n_healthy=len(hh))
    except Exception as e:  # noqa: BLE001
        log(f"contrast failed: {e!r}")
    cats, nums = covars(DN)
    mods = []
    # ISG is non-monotone in severity and the flag can depend on it non-linearly (only ISG-high cells cross the
    # threshold), so a linear ISG term under-adjusts; the gated models use ISG + ISG^2 + ISG-detector rate.
    isgx = ["isg_z", "isg_z2", "isgdet"]
    specs = [("sev+ISG(lin)", ["isg_z"]), ("sev+ISG", isgx), ("sev+ISG+GC+IEG+depth", isgx + ["gc_z", "ieg_z", "log_counts"]),
             ("sev+ISG+IFNg+GC+IEG+depth", isgx + ["ifng_z", "gc_z", "ieg_z", "log_counts"])]
    for rc in [c for c in ["rate_ctrl", "rate_ctrl_std", "rate_bonf", "rate_ctrl_isglow_std"] if c in DN]:
        for nm, d in [("all", ordd), ("covid_only", cov)]:
            for mname, extra in specs:
                ex = [x for x in extra if x in d]
                try:
                    T, info = ols_robust(d, rc, cats=cats, nums=nums + ex, raw=["sev_ord"])
                    mods.append(dict(var=rc, set=nm, model=mname, **term_row(T, "sev_ord"),
                                     isg_coef=term_row(T, "isg_z")["coef"], isg_p=term_row(T, "isg_z")["p"],
                                     gc_coef=term_row(T, "gc_z")["coef"], gc_p=term_row(T, "gc_z")["p"],
                                     ieg_coef=term_row(T, "ieg_z")["coef"], ieg_p=term_row(T, "ieg_z")["p"],
                                     n=info["n"], r2=info["r2"]))
                except Exception as e:  # noqa: BLE001
                    log(f"Q2 OLS {rc} {nm} {mname}: {e!r}")
    M = pd.DataFrame(mods)
    wcsv(M, "Q2_ols_beyond_isg.csv")
    out["ols"] = M
    # residual trend: rate residualized on ISG (within COVID), Spearman with severity
    try:
        d = cov.dropna(subset=["rate_ctrl", "isg_z"])
        X = np.column_stack([np.ones(len(d)), d.isg_z, d.isg_z ** 2])
        res = d.rate_ctrl.values - X @ np.linalg.lstsq(X, d.rate_ctrl.values, rcond=None)[0]
        out["resid_on_isg_quadratic_rho_sev"] = spearman_boot(d.sev_ord.values, res)
    except Exception as e:  # noqa: BLE001
        log(f"residual trend failed: {e!r}")
    out["rho_rate_isg_covid"] = spearman_boot(cov.isg_z, cov.rate_ctrl)
    try:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for s in SEV_ORDER:
            d = ordd[ordd.sev == s]
            ax.scatter(d.isg_z, d.rate_ctrl, s=36, color=SEV_COL[s], label=s, alpha=.85)
        ax.set_xlabel("donor ISG (z)")
        ax.set_ylabel("flagged cells (%)")
        ax.legend(frameon=False, fontsize=12)
        savefig(fig, "Q2_rate_vs_isg.png")
    except Exception as e:  # noqa: BLE001
        log(f"Q2 fig failed: {e!r}")
    R["Q2"] = out
    return out


# ----------------------------------------------------------------------------------------------
# Q3 per type
# ----------------------------------------------------------------------------------------------
@section("Q3_types")
def q3(Dp, DN, w):
    out = {}
    d = Dp[Dp.sev.isin(SEV_ORDER)]
    g = d.groupby(["donor", "y"]).f_ctrl.agg(["mean", "size"])
    rate = (g["mean"] * 100).where(g["size"] >= args.min_cells_type).unstack()
    n = g["size"].unstack(fill_value=0)
    frac = n.div(n.sum(1), axis=0)
    sev = DN.sev.reindex(rate.index)
    so = sev.map(SEV_ORD)
    rows = []
    for t in rate.columns:
        v = rate[t]
        row = dict(type=t, lineage=lineage_of(t), n_donors=int(v.notna().sum()))
        for s in SEV_ORDER:
            row[f"mean_{s}"] = float(v[sev == s].mean())
        ex = diff_boot(v[sev.isin(["severe", "critical"])], v[sev == "healthy"])
        row.update(excess_sevcrit=ex["diff"], ex_ci_lo=ex["ci_lo"], ex_ci_hi=ex["ci_hi"], ex_p=ex["p_mwu"])
        sb = spearman_boot(so, v, nboot=500)
        row.update(rho_sev=sb["rho"], rho_lo=sb["ci_lo"], rho_hi=sb["ci_hi"], rho_p=sb["p"])
        rows.append(row)
    T = pd.DataFrame(rows)
    T["rho_q"] = bh(T.rho_p.values)
    T["ex_q"] = bh(T.ex_p.values)
    # decomposition of severe/critical - healthy excess of raw donor rate
    sc, hh = sev.isin(["severe", "critical"]), sev == "healthy"
    P_S, P_H = frac[sc].mean(), frac[hh].mean()
    R_S, R_H = rate[sc].mean().fillna(rate.mean()), rate[hh].mean().fillna(rate.mean())
    comp = (P_S - P_H) * R_H
    within = P_S * (R_S - R_H)
    tot = (comp + within).sum()
    T = T.set_index("type")
    T["frac_healthy"] = P_H
    T["frac_sevcrit"] = P_S
    T["contrib_composition"] = comp
    T["contrib_within_rate"] = within
    T["share_of_excess"] = (comp + within) / tot if tot != 0 else np.nan
    T = T.reset_index().sort_values("contrib_within_rate", ascending=False)
    wcsv(T, "Q3_types.csv")
    out["types"] = T
    out["excess_total_pp"] = float(tot)
    out["excess_composition_pp"] = float(comp.sum())
    out["excess_within_pp"] = float(within.sum())
    out["observed_excess_pp"] = float(DN.loc[DN.sev.isin(["severe", "critical"]), "rate_ctrl"].mean()
                                      - DN.loc[DN.sev == "healthy", "rate_ctrl"].mean())
    # lineage shares of severe/critical flagged cells
    m = Dp.sev.isin(["severe", "critical"])
    out["sevcrit_flags_by_lineage"] = (Dp[m].groupby("lineage").f_ctrl.sum() / max(Dp[m].f_ctrl.sum(), 1e-9)).to_dict()
    try:
        top = T.dropna(subset=["n_donors"]).head(20)
        M = top[[f"mean_{s}" for s in SEV_ORDER]].values
        fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(top) + 2))
        im = ax.imshow(M, aspect="auto", cmap="Reds")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.type)
        ax.set_xticks(range(len(SEV_ORDER)))
        ax.set_xticklabels([s[:5] for s in SEV_ORDER], rotation=30)
        fig.colorbar(im, ax=ax, label="flagged (%)")
        savefig(fig, "Q3_type_rates_heatmap.png")
    except Exception as e:  # noqa: BLE001
        log(f"Q3 fig failed: {e!r}")
    R["Q3"] = {k: v for k, v in out.items() if k != "types"}
    R["Q3"]["top_within"] = T.head(8)[["type", "excess_sevcrit", "ex_ci_lo", "ex_ci_hi", "contrib_within_rate",
                                       "contrib_composition", "rho_sev", "rho_q"]]
    return out


# ----------------------------------------------------------------------------------------------
# Q4 flagged vs kept
# ----------------------------------------------------------------------------------------------
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


def tech_genes(gs, var):
    t = set(g for g in var if re.match(TECH_GENE, g))
    for k, v in gs.items():
        if re.search(TECH_SET, k, re.I):
            t |= set(v)
    return t


def donor_deltas(DE, cols):
    """per donor: mean over flagged cells of (score - mean score of kept cells of the same donor and type).
    Equals the n_flagged-weighted mean over types of (flagged mean - kept mean). Needs >= min_cells_delta kept cells
    in the donor x type and >= min_cells_delta flagged cells in the donor (summed over types), so it is estimable in
    donors with few flagged cells (healthy controls)."""
    d = DE[(DE.flag_cat != "mid")]
    if not len(d):
        return pd.DataFrame(columns=cols)
    kept = d[d.flag_cat == "kept"]
    km = kept.groupby(["donor", "y"])[cols].mean()
    kn = kept.groupby(["donor", "y"]).size()
    km = km[kn.reindex(km.index) >= args.min_cells_delta]
    fl = d[d.flag_cat == "flagged"]
    if not len(fl) or not len(km):
        return pd.DataFrame(columns=cols)
    key = pd.MultiIndex.from_arrays([fl.donor.values, fl.y.values])
    base = km.reindex(key)
    ok = base.notna().all(1).values
    res = pd.DataFrame(fl[cols].values[ok] - base.values[ok], columns=cols)
    res["donor"] = fl.donor.values[ok]
    n = res.groupby("donor").size()
    out = res.groupby("donor")[cols].mean()
    return out[n.reindex(out.index) >= args.min_cells_delta]


@section("Q4_flagged_vs_kept")
def q4(Dp, DN, X, var, E, rowpos, gs, extra_sets):
    out = {}
    rng = np.random.default_rng(4)
    DE = Dp.join(E, how="inner")
    DE["flag_cat"] = np.select([DE.f_ctrl >= 0.5, DE.f_ctrl <= 0], ["flagged", "kept"], "mid")
    DE["grp"] = np.select([DE.sev.isin(["severe", "critical"]), DE.sev.isin(["mild", "moderate"]),
                           DE.sev == "healthy", DE.sev == "asymptomatic", DE.sev == "noncovid"],
                          ["sevcrit", "mildmod", "healthy", "asym", "noncovid"], "other")
    univ = set(var)
    tech = tech_genes(gs, var)
    # (a) pseudobulk DE per type in severe/critical donors (and healthy for comparison)
    pb_rows, enr_rows, up_lists = [], [], {}
    for grp in ["sevcrit", "healthy"]:
        for t, g in DE[DE.grp == grp].groupby("y"):
            ga, gb = (g.flag_cat == "flagged").values, (g.flag_cat == "kept").values
            res, nd = pseudobulk_paired(X, rowpos, g, ga, gb, rng)
            if res is None:
                continue
            res["gene"] = var[res.gi.values]
            res["q"] = bh(res.p.values)
            res["technical"] = res.gene.isin(tech)
            res = res.sort_values("t", ascending=False)
            res.drop(columns="gi").to_csv(os.path.join(OUT, f"Q4_pb_{grp}_{re.sub(r'[^0-9A-Za-z_]', '_', t)}.csv"),
                                          index=False)
            sig_up = res[(res.q < 0.1) & (res.mean_diff > 0) & (res.frac_donors_up >= 0.7)]
            sig_dn = res[(res.q < 0.1) & (res.mean_diff < 0) & (res.frac_donors_up <= 0.3)]
            nontech_up = sig_up[~sig_up.technical]
            pb_rows.append(dict(grp=grp, type=t, lineage=lineage_of(t), n_donors=nd, n_up=len(sig_up),
                                n_down=len(sig_dn), n_up_nontech=len(nontech_up),
                                top_up_nontech=",".join(nontech_up.gene.head(15)),
                                top_down=",".join(sig_dn.sort_values("t").gene.head(10))))
            if grp == "sevcrit":
                up_lists[t] = nontech_up
                # universe = genes actually tested (detected in >=5% of cells), not all var_names: a 30k-gene universe
                # makes any set of expressed genes look enriched among the top hits
                tuniv = set(res.gene)
                upg = sig_up.gene.tolist() if len(sig_up) >= 5 else res.gene.head(100).tolist()
                en = enrich(upg, tuniv, gs)
                if len(en):
                    en["type"] = t
                    en["direction"] = "up" if len(sig_up) >= 5 else "up(top100 by t, not significant)"
                    enr_rows.append(en)
                dng = sig_dn.gene.tolist() if len(sig_dn) >= 5 else res.gene.tail(100).tolist()
                en = enrich(dng, tuniv, gs)
                if len(en):
                    en["type"] = t
                    en["direction"] = "down" if len(sig_dn) >= 5 else "down(top100 by t, not significant)"
                    enr_rows.append(en)
    PB = pd.DataFrame(pb_rows)
    wcsv(PB, "Q4_pseudobulk_summary.csv")
    out["pb"] = PB
    if enr_rows:
        EN = pd.concat(enr_rows)
        wcsv(EN, "Q4_enrichment.csv")
        out["enrich_sig"] = EN[EN.q < 0.05][["type", "direction", "set", "overlap", "size", "q"]].head(40)
    # (b) gene-set score deltas flagged - kept per donor, by group
    zc = [c for c in E.columns if c.startswith("z_")]
    rows = []
    deltas = {}
    for grp in ["sevcrit", "mildmod", "asym", "healthy", "noncovid"]:
        dd = donor_deltas(DE[DE.grp == grp], zc)
        deltas[grp] = dd
        for c in zc:
            if c in dd:
                b = boot_mean_ci(dd[c].values)
                rows.append(dict(grp=grp, set=c[2:], n_donors=b["n"], delta=b["mean"], ci_lo=b["ci_lo"],
                                 ci_hi=b["ci_hi"], p_wilcoxon=b["p_wilcoxon"], frac_pos=b["frac_pos"]))
    GD = pd.DataFrame(rows)
    # severe - healthy difference of deltas
    dif = []
    for c in zc:
        a = deltas.get("sevcrit", pd.DataFrame()).get(c)
        b = deltas.get("healthy", pd.DataFrame()).get(c)
        m = deltas.get("mildmod", pd.DataFrame()).get(c)
        if a is None or b is None:
            continue
        r = diff_boot(a.values, b.values)
        r2 = diff_boot(a.values, m.values) if m is not None else {}
        dif.append(dict(set=c[2:], sev_minus_healthy=r["diff"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p=r["p_mwu"],
                        sev_minus_mildmod=r2.get("diff", np.nan), smm_ci_lo=r2.get("ci_lo", np.nan),
                        smm_ci_hi=r2.get("ci_hi", np.nan),
                        known_or_technical=bool(re.search(KNOWN_COVID, c[2:], re.I)),
                        from_lupus=c[2:].startswith("LUPUS_")))
    DF = pd.DataFrame(dif)
    if len(DF):
        DF["q"] = bh(DF.p.values)
        DF = DF.sort_values("p")
    wcsv(GD, "Q4_geneset_deltas.csv")
    wcsv(DF, "Q4_geneset_delta_sev_vs_healthy.csv")
    out["deltas"] = GD
    out["delta_diff"] = DF
    out["key_sets"] = GD[GD.set.isin(KEY_SETS) & GD.grp.isin(["sevcrit", "healthy"])]
    # (c) ISG-low flagged cells in severe/critical
    if "isg_low" in DE:
        sc = DE[(DE.grp == "sevcrit")]
        fl = sc[sc.flag_cat == "flagged"]
        pd_ = fl.groupby("donor").isg_low.mean()
        out["sevcrit_flagged_isglow_frac"] = boot_mean_ci(pd_.values)
        out["sevcrit_kept_isglow_frac"] = boot_mean_ci(sc[sc.flag_cat == "kept"].groupby("donor").isg_low.mean().values)
        dd = donor_deltas(sc[sc.isg_low == 1], zc)
        rr = []
        for c in zc:
            if c in dd:
                b = boot_mean_ci(dd[c].values)
                rr.append(dict(set=c[2:], n_donors=b["n"], delta=b["mean"], ci_lo=b["ci_lo"], ci_hi=b["ci_hi"]))
        IL = pd.DataFrame(rr)
        wcsv(IL, "Q4_isglow_deltas_sevcrit.csv")
        out["isglow_deltas"] = IL
    # lineage-specific deltas for key sets
    lr = []
    for lin in sorted(DE.lineage.unique()):
        for grp in ["sevcrit", "healthy"]:
            dd = donor_deltas(DE[(DE.grp == grp) & (DE.lineage == lin)], zc)
            for c in [x for x in zc if x[2:] in KEY_SETS or x[2:].startswith("LUPUS_")]:
                if c in dd and dd[c].notna().sum() >= 3:
                    b = boot_mean_ci(dd[c].values, nboot=500)
                    lr.append(dict(lineage=lin, grp=grp, set=c[2:], n=b["n"], delta=b["mean"], ci_lo=b["ci_lo"],
                                   ci_hi=b["ci_hi"]))
    wcsv(pd.DataFrame(lr), "Q4_geneset_deltas_by_lineage.csv")
    out["up_lists"] = up_lists
    try:
        ks = [s for s in KEY_SETS if s in set(GD.set)]
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(ks) + 2))
        for j, (grp, col) in enumerate([("healthy", SEV_COL["healthy"]), ("mildmod", SEV_COL["mild"]),
                                        ("sevcrit", SEV_COL["severe"])]):
            g = GD[GD.grp == grp].set_index("set").reindex(ks)
            yy = np.arange(len(ks)) + (j - 1) * 0.25
            ax.errorbar(g.delta, yy, xerr=[g.delta - g.ci_lo, g.ci_hi - g.delta], fmt="o", color=col, label=grp,
                        ms=7, capsize=2)
        ax.axvline(0, color="grey", lw=1)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks, fontsize=12)
        ax.set_xlabel("flagged - kept score (z)")
        ax.legend(frameon=False)
        savefig(fig, "Q4_geneset_deltas.png")
    except Exception as e:  # noqa: BLE001
        log(f"Q4 fig failed: {e!r}")
    R["Q4"] = {k: v for k, v in out.items() if k not in ("up_lists", "deltas")}
    return out, DE


# ----------------------------------------------------------------------------------------------
# Q5 cross-disease
# ----------------------------------------------------------------------------------------------
def derive_covid_programs(q4o, gs, var):
    progs = {}
    up = q4o.get("up_lists", {}) if q4o else {}
    tech = tech_genes(gs, var)
    by_lin = {}
    for t, df in up.items():
        if df is None or not len(df):
            continue
        by_lin.setdefault(lineage_of(t), []).append(df[["gene", "t"]])
    allg = []
    for lin, dfs in by_lin.items():
        d = pd.concat(dfs).groupby("gene").t.max().sort_values(ascending=False)
        d = d[~d.index.isin(tech)]
        if len(d) >= 8:
            progs[f"COVID_flag_{lin}"] = d.index[:50].tolist()
        allg.append(d)
    if allg:
        d = pd.concat(allg).groupby(level=0).max().sort_values(ascending=False)
        if len(d) >= 8:
            progs["COVID_flag_all"] = d.index[:50].tolist()
    return progs


@section("Q5_cross_disease")
def q5(DE, DN, q4o, gs, var, lupus_progs, used):
    out = {}
    isg_ieg = set()
    for k, v in gs.items():
        if re.search(r"ISG|IFN|SIGLEC1|dissoc|IEG|heat", k, re.I):
            isg_ieg |= set(v)
    # (a) lupus programs in COVID
    if lupus_progs:
        out["lupus_programs"] = {k: dict(n_genes=len(v), n_in_data=len(used.get("LUPUS_" + k, [])))
                                 for k, v in lupus_progs.items()}
        rows = []
        zc = [c for c in DE.columns if c.startswith("z_LUPUS_")]
        cats, nums = covars(DN)
        for grp in ["sevcrit", "mildmod", "healthy", "noncovid"]:
            dd = donor_deltas(DE[DE.grp == grp], zc)
            for c in zc:
                if c in dd:
                    b = boot_mean_ci(dd[c].values)
                    rows.append(dict(program=c[8:], grp=grp, n_donors=b["n"], delta=b["mean"], ci_lo=b["ci_lo"],
                                     ci_hi=b["ci_hi"]))
            if grp == "sevcrit":
                sev_dd = dd
        LP = pd.DataFrame(rows)
        # donor-level: delta vs severity across all ordered donors, adjusted for ISG
        alld = donor_deltas(DE[DE.sev.isin(SEV_ORDER)], zc)
        mods = []
        for c in zc:
            if c not in alld:
                continue
            d = DN.join(alld[[c]], how="inner")
            d = d[d.sev.isin(SEV_ORDER)]
            try:
                T, info = ols_robust(d, c, cats=cats, nums=nums + ["isg_z"], raw=["sev_ord"])
                mods.append(dict(program=c[8:], **term_row(T, "sev_ord"), n=info["n"]))
            except Exception as e:  # noqa: BLE001
                log(f"lupus prog OLS {c}: {e!r}")
        out["lupus_in_covid"] = LP
        out["lupus_in_covid_ols"] = pd.DataFrame(mods)
        wcsv(LP, "Q5_lupus_programs_in_covid.csv")
        wcsv(pd.DataFrame(mods), "Q5_lupus_programs_sev_ols.csv")
    else:
        out["lupus_programs"] = "not found"
        log("no lupus programs file; skipping (a)")
    # (b) COVID programs
    cp = derive_covid_programs(q4o, gs, var)
    cp_noisg = {k: [g for g in v if g not in isg_ieg] for k, v in cp.items()}
    json.dump(cp_noisg, open(os.path.join(OUT, "covid_programs.json"), "w"), indent=1)
    out["covid_programs"] = {k: v[:12] for k, v in cp_noisg.items()}
    # (c) overlap
    if lupus_progs and cp_noisg:
        univ = set(var)
        ov = []
        for a, ga in lupus_progs.items():
            for b, gb in cp_noisg.items():
                A, B = set(ga) & univ, set(gb) & univ
                k = len(A & B)
                p = stats.hypergeom.sf(k - 1, len(univ), len(A), len(B)) if k > 0 else 1.0
                ov.append(dict(lupus=a, covid=b, n_lupus=len(A), n_covid=len(B), overlap=k,
                               jaccard=k / max(len(A | B), 1), p=p, genes=",".join(sorted(A & B))))
        OV = pd.DataFrame(ov)
        OV["q"] = bh(OV.p.values)
        wcsv(OV, "Q5_program_overlap.csv")
        out["overlap"] = OV.sort_values("p").head(10)
    # (d) COVID programs in SLE
    if not args.no_sle and cp_noisg:
        try:
            out["covid_in_sle"] = covid_in_sle(cp_noisg, gs)
        except Exception as e:  # noqa: BLE001
            log(f"COVID programs in SLE failed: {e!r}")
            log(traceback.format_exc())
            out["covid_in_sle"] = f"failed: {e!r}"
    R["Q5"] = out
    return out


def covid_in_sle(progs, gs):
    D, seeds = load_cells(args.sle_tag)
    H = Matrix(args.sle_tag, args.sle_h5ad)
    D["is_sle"] = D.cond_raw.str.contains(r"lupus|\bsle\b", case=False, regex=True).astype(int)
    rng = np.random.default_rng(9)
    keep = []
    for _, g in D.groupby("donor"):
        keep.append(rng.choice(g.index.values, min(len(g), args.sle_max_per_donor), replace=False))
    cells = D.index[D.index.isin(np.concatenate(keep))].values
    X, tot = H.rows(cells)
    isgk = detect_key(gs, args.isg_key, [r"^ISG_typeI$", r"ISG"])
    sets = {("COVID_" + k if not k.startswith("COVID") else k): v for k, v in progs.items()}
    if isgk:
        sets["ISG"] = gs[isgk]
    S, used = score_sets(X, H.var, sets, rng)
    S.index = pd.Index(cells)
    d = D.loc[cells].copy()
    healthy = (d.is_sle == 0).values
    zc = []
    for c in S.columns:
        d["z_" + c] = z_by_type(S[c].values, d.y.values, healthy)
        zc.append("z_" + c)
    d["flag_cat"] = np.select([d.f_ctrl >= 0.5, d.f_ctrl <= 0], ["flagged", "kept"], "mid")
    rows = []
    dS = donor_deltas(d[d.is_sle == 1], [c for c in zc if c != "z_ISG"])
    dH = donor_deltas(d[d.is_sle == 0], [c for c in zc if c != "z_ISG"])
    disg = d.groupby("donor")["z_ISG"].mean() if "z_ISG" in d else None
    for c in dS.columns:
        hv = dH[c].dropna().values if c in dH else np.array([])
        if len(hv) < 3:  # too few healthy donors with enough flagged cells: compare SLE delta to 0
            hv = np.zeros(0)
        e = diff_boot(dS[c].values, hv)
        bS = boot_mean_ci(dS[c].values)
        r = dict(program=c[2:], n_sle=bS["n"], n_healthy=int(len(hv)), delta_sle=bS["mean"], sle_ci_lo=bS["ci_lo"],
                 sle_ci_hi=bS["ci_hi"], delta_healthy=e["mean_b"], sle_minus_healthy=e["diff"], ci_lo=e["ci_lo"],
                 ci_hi=e["ci_hi"], p=e["p_mwu"])
        if disg is not None:
            sb = spearman_boot(disg.reindex(dS.index).values, dS[c].values, nboot=500)
            r.update(rho_isg_sle=sb["rho"], rho_lo=sb["ci_lo"], rho_hi=sb["ci_hi"])
        rows.append(r)
    T = pd.DataFrame(rows)
    wcsv(T, "Q5_covid_programs_in_sle.csv")
    log(f"COVID programs in SLE ({args.sle_tag}, seeds {seeds}): {len(cells)} cells")
    return T


# ----------------------------------------------------------------------------------------------
# Q6 repeat samples
# ----------------------------------------------------------------------------------------------
@section("Q6_repeat_samples")
def q6(ST, DN):
    out = {}
    S = ST[ST.sev.isin(SEV_ORDER + ["noncovid"])]
    multi = S.groupby("donor").size()  # count AFTER filtering, else donors left with 1 sample give first==last (delta 0)
    multi = multi[multi > 1].index
    S = S[S.donor.isin(multi)].copy()
    out["n_donors_multi"] = int(len(multi))
    if len(multi) < 2:
        R["Q6"] = out
        log("fewer than 2 donors with repeat samples")
        return out
    S["_d"] = time_key(S)
    S = S.sort_values(["donor", "_d"])
    wcsv(S.reset_index().drop(columns="_d"), "Q6_trajectories.csv")
    first, last = S.groupby("donor").head(1).set_index("donor"), S.groupby("donor").tail(1).set_index("donor")
    cols = [c for c in ["rate_ctrl", "rate_ctrl_std", "rate_ctrl_isglow", "isg_z", "isgdet", "gc_z", "ieg_z",
                        "days", "sev_ord"] if c in S]
    dlt = (last[cols] - first[cols])
    out["paired_first_to_last"] = {c: boot_mean_ci(dlt[c].values) for c in cols}
    hq95 = np.nanquantile(DN.loc[DN.sev == "healthy", "rate_ctrl"], 0.95) if (DN.sev == "healthy").sum() >= 5 else np.nan
    isg_res = dlt.isg_z < 0 if "isg_z" in dlt else pd.Series(False, index=dlt.index)
    out["healthy_q95_rate"] = hq95
    out["n_isg_falls"] = int(isg_res.sum())
    # persistence is judged on the ISG-low-cell flag rate against the healthy q95 of the same quantity (the overall rate
    # can stay high just because residual ISG-high cells are flagged)
    pc = "rate_ctrl_isglow" if "rate_ctrl_isglow" in S and "rate_ctrl_isglow" in DN else "rate_ctrl"
    hv = DN.loc[DN.sev == "healthy", pc].dropna()
    hq95i = float(np.nanquantile(hv, 0.95)) if len(hv) >= 5 else np.nan
    out["persistence_var"] = pc
    out["healthy_q95_persistence_var"] = hq95i
    if isg_res.sum():
        ix = isg_res[isg_res].index
        out["frac_still_flag_high_when_isg_falls"] = float((last.loc[ix, pc] > hq95i).mean())
        out["frac_first_flag_high_when_isg_falls"] = float((first.loc[ix, pc] > hq95i).mean())
        out["frac_still_rate_ctrl_high_when_isg_falls"] = float((last.loc[ix, "rate_ctrl"] > hq95).mean())
    # within-donor association (donor-demeaned)
    wd = S.copy()
    for c in cols:
        wd[c + "_w"] = wd[c] - wd.groupby("donor")[c].transform("mean")
    out["within_rho_rate_days"] = spearman_cluster_boot(wd.days_w, wd.rate_ctrl_w, wd.donor) if "days" in cols else None
    out["within_rho_isg_days"] = spearman_cluster_boot(wd.days_w, wd.isg_z_w, wd.donor) if "isg_z" in cols else None
    out["within_rho_rate_isg"] = spearman_cluster_boot(wd.isg_z_w, wd.rate_ctrl_w, wd.donor) if "isg_z" in cols else None
    try:
        T, info = ols_robust(wd, "rate_ctrl_w", nums=[c + "_w" for c in ["isg_z", "days", "gc_z", "ieg_z"] if c in cols],
                             cluster="donor")
        out["within_ols"] = T
        out["within_ols_info"] = info
        T.to_csv(os.path.join(OUT, "Q6_within_ols.csv"), index=False)
    except Exception as e:  # noqa: BLE001
        log(f"Q6 within OLS: {e!r}")
    try:
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
        for dn, g in S.groupby("donor"):
            g = g[np.isfinite(g.days)]
            if len(g) < 2:
                continue
            col = SEV_COL.get(g.sev.iloc[0], "grey")
            axs[0].plot(g.days, g.rate_ctrl, "-o", color=col, alpha=.7)
            if "isg_z" in g:
                axs[1].plot(g.days, g.isg_z, "-o", color=col, alpha=.7)
        if np.isfinite(hq95):
            axs[0].axhline(hq95, color="grey", ls="--")
        axs[0].set_xlabel("days from onset")
        axs[1].set_xlabel("days from onset")
        axs[0].set_ylabel("flagged cells (%)")
        axs[1].set_ylabel("ISG (z)")
        savefig(fig, "Q6_trajectories.png")
    except Exception as e:  # noqa: BLE001
        log(f"Q6 fig failed: {e!r}")
    R["Q6"] = out
    return out


# ----------------------------------------------------------------------------------------------
# gates and report
# ----------------------------------------------------------------------------------------------
def f3(x, d=2):
    try:
        return "NA" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def eci(e, k="coef", d=2):
    if not e:
        return "NA"
    return f"{f3(e.get(k), d)} [{f3(e.get('ci_lo'), d)}, {f3(e.get('ci_hi'), d)}]"


def pick(M, **kw):
    if M is None or not len(M):
        return {}
    m = np.ones(len(M), bool)
    for k, v in kw.items():
        m &= (M[k] == v).values
    return M[m].iloc[0].to_dict() if m.any() else {}


def gates(q1o, q2o, q3o, q4o, q5o, q6o):
    G = {}
    if q1o:
        tr = pick(q1o["trend"], var="rate_ctrl", set="all")
        ol = pick(q1o["ols"], var="rate_ctrl", set="all", model="sev_ord+site+age+sex")
        G["G1 severity trend"] = (
            (tr.get("ci_lo", np.nan) > 0) and (ol.get("ci_lo", np.nan) > 0),
            f"rho {eci(tr, 'rho')}; slope {eci(ol)} pp/step")
        olS = pick(q1o["ols"], var="rate_ctrl_std", set="all", model="sev_ord+site+age+sex")
        G["G2 composition-standardized"] = (olS.get("ci_lo", np.nan) > 0, f"slope {eci(olS)}")
        lo = q1o.get("loso")
        if lo is None:
            G["G4b leave-one-site-out"] = (True, "single site")
        else:
            G["G4b leave-one-site-out"] = (bool(len(lo) and (lo.rho_without > 0).all()),
                                           f"min {f3(lo.rho_without.min()) if len(lo) else 'NA'}")
    if q2o:
        o1 = pick(q2o["ols"], var="rate_ctrl", set="all", model="sev+ISG")
        rd = q2o["rho_diff"].get("rate_ctrl_vs_isg|covid_only", {})
        G["G3 beyond ISG"] = (
            (o1.get("ci_lo", np.nan) > 0) and (rd.get("ci_lo", np.nan) > 0),
            f"slope|ISG {eci(o1)}; drho {eci(rd, 'diff')}")
        o3 = pick(q2o["ols"], var="rate_ctrl", set="all", model="sev+ISG+GC+IEG+depth")
        il = pick(q1o["trend"], var="rate_ctrl_isglow_std", set="all") if q1o else {}
        if not il or il.get("n", 0) < 0.6 * q2o.get("n_ordered_donors", 0) or not np.isfinite(il.get("ci_lo", np.nan)):
            il = pick(q1o["trend"], var="rate_ctrl_isglow", set="all") if q1o else {}
        G["G4 beyond ISG+GC+IEG; ISG-low cells"] = (
            (o3.get("ci_lo", np.nan) > 0) and (il.get("ci_lo", np.nan) > 0),
            f"slope {eci(o3)}; ISG-low rho {eci(il, 'rho')}")
    if q4o:
        DF = q4o.get("delta_diff")
        nov = DF[(~DF.known_or_technical) & (DF.ci_lo > 0) & (DF.q < 0.1)] if DF is not None and len(DF) else []
        kn = DF[(DF.known_or_technical) & (DF.ci_lo > 0)] if DF is not None and len(DF) else []
        pb = q4o.get("pb")
        npb = int((pb[(pb.grp == "sevcrit")].n_up_nontech >= 10).sum()) if pb is not None and len(pb) else 0
        G["G5 non-canonical flagged content"] = (
            npb > 0 and len(nov) > 0,
            f"types {npb}; non-known sets {list(nov.set)[:4] if len(nov) else []}; known/positive-control sets "
            f"{list(kn.set)[:5] if len(kn) else []}")
    if q5o:
        ok_a, va = False, "no lupus programs"
        LP = q5o.get("lupus_in_covid")
        if isinstance(LP, pd.DataFrame) and len(LP):
            a = LP[LP.grp == "sevcrit"].set_index("program")
            h = LP[LP.grp == "healthy"].set_index("program")
            # programs with an ISG/IEG-stripped variant are judged on that variant only
            cand = [p for p in a.index if (p + "__noISG") not in a.index and p not in ISG_DOMINATED]
            hits = [p for p in cand if a.loc[p, "ci_lo"] > 0 and (p not in h.index or a.loc[p, "delta"] > h.loc[p, "ci_hi"])]
            ok_a, va = len(hits) > 0, f"SLE programs up in severe COVID flagged cells: {hits[:4]}"
        ok_b, vb = None, "SLE side not run (no COVID programs or no SLE tag)"
        CS = q5o.get("covid_in_sle")
        if isinstance(CS, pd.DataFrame) and len(CS):
            lo = CS.ci_lo.where(CS.ci_lo.notna(), CS.sle_ci_lo)  # vs healthy if estimable, else vs 0
            hits = CS[lo > 0]
            hits = hits[~(hits.rho_isg_sle > 0.3)] if "rho_isg_sle" in hits else hits
            ok_b, vb = len(hits) > 0, f"COVID programs up in SLE flagged cells, rho ISG<=0.3: {list(hits.program)[:4]}"
        G["G6 cross-disease program"] = (bool(ok_a and (ok_b is not False)), f"{va}; {vb}")
    if q6o and q6o.get("n_donors_multi", 0) >= 5:
        pf = q6o.get("paired_first_to_last", {})
        di, dr, dl = pf.get("isg_z", {}), pf.get("rate_ctrl", {}), pf.get("rate_ctrl_isglow", {})
        ok = (di.get("ci_hi", np.nan) < 0) and not (dl.get("ci_hi", np.nan) < 0) and np.isfinite(dl.get("mean", np.nan)) \
            and (q6o.get("frac_still_flag_high_when_isg_falls") or 0) >= 0.5
        G["G7 persistence after ISG resolves"] = (
            bool(ok), f"dISG {eci(di, 'mean')}; dRate {eci(dr, 'mean')}; dRate ISG-low {eci(dl, 'mean')}; still high "
                      f"{f3(q6o.get('frac_still_flag_high_when_isg_falls'))} (n={q6o.get('n_donors_multi')})")
    else:
        G["G7 persistence after ISG resolves"] = (
            False, "fewer than 5 donors with repeat samples")
    # all core gates must exist (a failed section must not silently drop a gate) and pass, incl. leave-one-site-out
    core_keys = ["G1 severity trend", "G2 composition-standardized", "G3 beyond ISG",
                 "G4 beyond ISG+GC+IEG; ISG-low cells", "G4b leave-one-site-out"]
    if "G4b leave-one-site-out" in G and G["G4b leave-one-site-out"][1] == "single site":
        core_keys.remove("G4b leave-one-site-out")
    core = all(bool(G.get(k, (False,))[0]) for k in core_keys)
    nov = G.get("G5 non-canonical flagged content", (False,))[0]
    cross = G.get("G6 cross-disease program", (False,))[0]
    pers = [v for k, v in G.items() if k.startswith("G7")][0][0]
    if core and (cross or pers) and nov:
        v = "NOVEL candidate: severity signal not reducible to ISG, carried by a non-canonical state that is shared with SLE and/or persists after ISG resolution"
    elif core and (cross or pers):
        v = "BEYOND-ISG (strong): severity signal not reducible to ISG plus cross-disease or persistence evidence; flagged-cell content is known COVID biology"
    elif core:
        v = "BEYOND-ISG (methodological): flags track severity independently of ISG, but flagged-cell content matches known severity states (positive control)"
    else:
        v = "NOT SHOWN: the flag-severity relation is not separable from ISG/composition/site/confounders on this data"
    return G, v


def write_results(q1o, q2o, q3o, q4o, q5o, q6o):
    L = []
    n = R.get("n", {})
    L.append("# ARCANE unmatched cells in COVID-19 (Stephenson) vs severity and ISG")
    L.append(f"Tag `{args.tag}`, seeds {R.get('n_raw', {}).get('seeds')}; severity column `{R.get('severity_column')}`. "
             f"Donors {n.get('donors')} (primary = earliest sample; {n.get('donors_multi_sample')} with repeats): "
             f"{n.get('donors_by_sev')}. LPS samples excluded. Flag = p_unknown_ctrl<={args.alpha} (mean over seeds), "
             f"% of cells; `std` = standardized to healthy-query type mix; CIs = 95% donor bootstrap ({args.nboot}); "
             "OLS = HC3 with site, sex, age (+ancestry if present); severity ordinal healthy=0..critical=5.")
    if R["errors"]:
        L.append(f"Errors: {', '.join(R['errors'])} (see log.txt).")
    try:
        G, verdict = gates(q1o, q2o, q3o, q4o, q5o, q6o)
        L.append(f"\n## Verdict: {verdict}")
        L.append("| gate | pass | value |\n|---|---|---|")
        for k, (ok, v) in G.items():
            L.append(f"| {k} | {'yes' if ok else 'no'} | {str(v)[:170]} |")
        L.append("Gates: G1 rho and site/age/sex-adjusted slope CI>0; G2 same on composition-standardized rate; G3 slope "
                 "| ISG+ISG^2+ISG-detector rate CI>0 and rho(sev,flag)-rho(sev,ISG) CI>0 in COVID; G4 slope | ISG+GC+IEG+depth CI>0 and trend in "
                 "ISG-low cells; G4b leave-one-site-out; G5 >=10 non-technical up genes in a type and a severe>healthy "
                 "flagged-kept delta for a set outside the known COVID list (MS1, HLA-II, plasmablast, platelet, GC, "
                 "cell cycle, LDG, ISG, IEG); G6 SLE program (ISG-stripped) up in severe COVID flags and COVID program up "
                 "in SLE flags (not ISG-tracking); G7 ISG falls but ISG-low-cell flags persist "
                 "(no significant drop, and >=50% of ISG-falling donors still above the healthy q95 of the ISG-low rate). Novel = G1-G4 + G5 + "
                 "(G6 or G7). Known-state flags alone are positive controls.")
        R["gates"] = {k: dict(pass_=v[0], value=v[1]) for k, v in G.items()}
        R["verdict"] = verdict
    except Exception as e:  # noqa: BLE001
        L.append(f"gates failed: {e!r}")
    try:
        if q1o:
            L.append("\n## Q1 flag rate by severity (donor mean % [CI])")
            GT = q1o["groups"]
            vs = [v for v in ["rate_ctrl", "rate_ctrl_std", "isg_z"] if v in set(GT["var"])]
            L.append("| severity | n | " + " | ".join(vs) + " |\n|---|---|" + "---|" * len(vs))
            for s in SEV_ORDER + ["noncovid"]:
                cells = []
                nn = 0
                for v in vs:
                    r = GT[(GT["var"] == v) & (GT.sev == s)]
                    if len(r):
                        r = r.iloc[0]
                        nn = r.n
                        cells.append(f"{f3(r['mean'])} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]")
                    else:
                        cells.append("NA")
                if nn:
                    L.append(f"| {s} | {nn} | " + " | ".join(cells) + " |")
            L.append("Trend (Spearman rho [CI]; JT z, perm p):")
            for _, r in q1o["trend"].iterrows():
                if r["var"] in ("rate_ctrl", "isg_z") or (r["var"] in ("rate_ctrl_std", "rate_ctrl_isglow") and r.set == "all"):
                    L.append(f"- {r['var']} | {r.set}: {f3(r.rho)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}] (n={r.n}); JT z {f3(r.jt_z)} p {f3(r.jt_p, 4)}")
            M = q1o["ols"]
            L.append("Adjusted slope (pp per severity step):")
            for _, r in M[(M.model == "sev_ord+site+age+sex") & ((M["var"] == "rate_ctrl") | ((M["var"] == "rate_ctrl_std") & (M.set == "all")))].iterrows():
                L.append(f"- {r['var']} | {r.set}: {f3(r.coef)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}] p={f3(r.p, 4)} (n={r.n})")
            if "within_site_weighted_rho" in q1o:
                L.append(f"- within-site weighted rho {f3(q1o['within_site_weighted_rho'])}; LOSO min rho "
                         f"{f3(q1o['loso'].rho_without.min())}")
            if "noncovid" in q1o:
                nc = q1o["noncovid"].get("rate_ctrl", {})
                L.append(f"- non-COVID resp. minus healthy {eci(nc.get('vs_healthy'), 'diff')}; minus severe/critical "
                         f"COVID {eci(nc.get('vs_covid_sevcrit'), 'diff')}")
    except Exception as e:  # noqa: BLE001
        L.append(f"Q1 report failed: {e!r}")
    try:
        if q2o:
            L.append(f"\n## Q2 beyond ISG (ISG peaks in: {q2o.get('isg_peak_group')})")
            for k in ["rate_ctrl_vs_isg|covid_only", "rate_ctrl_vs_isgdet|covid_only"]:
                r = q2o["rho_diff"].get(k)
                if r:
                    L.append(f"- rho(sev, {k.split('_vs_')[0]}) - rho(sev, {k.split('_vs_')[1]}): {f3(r.get('rho1'))} - "
                             f"{f3(r.get('rho2'))} = {eci(r, 'diff')}")
            c = q2o.get("sevcrit_minus_mildmod_contrast")
            if c:
                L.append(f"- severe/critical minus mild/moderate (healthy-SD units): ARCANE {f3(c['arcane'])}, ISG detector "
                         f"{f3(c['isgdet'])}, difference {eci(c, 'diff')}")
            M = q2o["ols"]
            for _, r in M[(M["var"] == "rate_ctrl") & (M.set == "all") & (M.model.isin(["sev+ISG", "sev+ISG+GC+IEG+depth"]))].iterrows():
                L.append(f"- {r['var']} ~ {r.model} (all donors): sev slope {f3(r.coef)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]; "
                         f"ISG {f3(r.isg_coef)} (p {f3(r.isg_p, 3)}); GC {f3(r.gc_coef)}; IEG {f3(r.ieg_coef)}")
            rr = q2o.get("resid_on_isg_quadratic_rho_sev")
            if rr:
                L.append(f"- COVID: flag residual on ISG+ISG^2 vs severity rho {eci(rr, 'rho')}; "
                         f"rho(flag, ISG) {eci(q2o.get('rho_rate_isg_covid'), 'rho')}")
    except Exception as e:  # noqa: BLE001
        L.append(f"Q2 report failed: {e!r}")
    try:
        if q3o:
            L.append(f"\n## Q3 types carrying the severe/critical excess (observed {f3(q3o['observed_excess_pp'])} pp; "
                     f"decomposition: composition {f3(q3o['excess_composition_pp'])}, within-type "
                     f"{f3(q3o['excess_within_pp'])})")
            L.append("| type | excess pp [CI] | within-type contrib | composition contrib | rho sev (q) |\n|---|---|---|---|---|")
            for _, r in q3o["types"].head(6).iterrows():
                L.append(f"| {r.type} | {f3(r.excess_sevcrit)} [{f3(r.ex_ci_lo)}, {f3(r.ex_ci_hi)}] | "
                         f"{f3(r.contrib_within_rate)} | {f3(r.contrib_composition)} | {f3(r.rho_sev)} ({f3(r.rho_q, 3)}) |")
            L.append(f"- severe/critical flags by lineage: "
                     + ", ".join(f"{k} {v:.2f}" for k, v in sorted(q3o["sevcrit_flags_by_lineage"].items(), key=lambda x: -x[1])))
    except Exception as e:  # noqa: BLE001
        L.append(f"Q3 report failed: {e!r}")
    try:
        if q4o:
            L.append("\n## Q4 severe/critical flagged vs kept (same type, donor-paired)")
            pb = q4o.get("pb")
            if pb is not None and len(pb):
                for _, r in pb[pb.grp == "sevcrit"].sort_values("n_up_nontech", ascending=False).head(4).iterrows():
                    L.append(f"- {r.type} (n={r.n_donors}): up {r.n_up} (non-technical {r.n_up_nontech}: "
                             f"{r.top_up_nontech[:70]}); down {r.n_down} ({r.top_down[:50]})")
            DF = q4o.get("delta_diff")
            GD = q4o.get("key_sets")
            if GD is not None and len(GD):
                L.append("| set | severe delta [CI] | healthy delta [CI] | severe - healthy [CI] |\n|---|---|---|---|")
                for s in [x for x in KEY_SETS if x not in ("IEG_core_specific", "heat_shock", "GC_induced_monocyte",
                                                           "LDG_immature_neutrophil", "NFkB_TNF_noIEG", "hypoxia")]:
                    a = GD[(GD.set == s) & (GD.grp == "sevcrit")]
                    b = GD[(GD.set == s) & (GD.grp == "healthy")]
                    dd = DF[DF.set == s] if DF is not None and len(DF) else []
                    if not len(a):
                        continue
                    a = a.iloc[0]
                    bt = f"{f3(b.iloc[0].delta)} [{f3(b.iloc[0].ci_lo)}, {f3(b.iloc[0].ci_hi)}]" if len(b) else "NA"
                    dt = f"{f3(dd.iloc[0].sev_minus_healthy)} [{f3(dd.iloc[0].ci_lo)}, {f3(dd.iloc[0].ci_hi)}]" if len(dd) else "NA"
                    L.append(f"| {s} | {f3(a.delta)} [{f3(a.ci_lo)}, {f3(a.ci_hi)}] | {bt} | {dt} |")
            if DF is not None and len(DF):
                nk = DF[(~DF.known_or_technical) & (DF.ci_lo > 0)].head(5)
                L.append("- other sets with severe > healthy delta (CI>0): "
                         + (", ".join(f"{r.set} {f3(r.sev_minus_healthy)} (q {f3(r.q, 3)})" for _, r in nk.iterrows()) or "none"))
            if "sevcrit_flagged_isglow_frac" in q4o:
                L.append(f"- ISG-low fraction: flagged {eci(q4o['sevcrit_flagged_isglow_frac'], 'mean')}, kept "
                         f"{eci(q4o['sevcrit_kept_isglow_frac'], 'mean')}")
    except Exception as e:  # noqa: BLE001
        L.append(f"Q4 report failed: {e!r}")
    try:
        if q5o:
            L.append("\n## Q5 cross-disease")
            LP = q5o.get("lupus_in_covid")
            if isinstance(LP, pd.DataFrame) and len(LP):
                for p in LP.program.unique()[:3]:
                    a = LP[(LP.program == p) & (LP.grp == "sevcrit")]
                    h = LP[(LP.program == p) & (LP.grp == "healthy")]
                    o = q5o["lupus_in_covid_ols"]
                    o = o[o.program == p] if len(o) else o
                    L.append(f"- SLE `{p}` in COVID flagged-kept: severe {eci(a.iloc[0].to_dict() if len(a) else {}, 'delta')}, "
                             f"healthy {eci(h.iloc[0].to_dict() if len(h) else {}, 'delta')}; severity slope | ISG "
                             f"{eci(o.iloc[0].to_dict() if len(o) else {})}")
            else:
                L.append(f"- SLE programs: {q5o.get('lupus_programs')}")
            L.append(f"- COVID programs written to covid_programs.json: "
                     + "; ".join(f"{k}: {','.join(v[:6])}" for k, v in q5o.get("covid_programs", {}).items()))
            OV = q5o.get("overlap")
            if isinstance(OV, pd.DataFrame) and len(OV):
                r = OV.iloc[0]
                L.append(f"- best gene overlap {r.lupus} x {r.covid}: {r.overlap} genes (p {f3(r.p, 4)}) {r.genes[:50]}")
            CS = q5o.get("covid_in_sle")
            if isinstance(CS, pd.DataFrame) and len(CS):
                for _, r in CS.iterrows():
                    L.append(f"- `{r.program}` in SLE flagged-kept: SLE {f3(r.delta_sle)} [{f3(r.sle_ci_lo)}, "
                             f"{f3(r.sle_ci_hi)}] (n={r.n_sle}), SLE - healthy {f3(r.sle_minus_healthy)} [{f3(r.ci_lo)}, "
                             f"{f3(r.ci_hi)}] (n healthy={r.n_healthy}); rho with ISG {f3(r.get('rho_isg_sle'))}")
            elif CS is not None:
                L.append(f"- COVID programs in SLE: {str(CS)[:150]}")
    except Exception as e:  # noqa: BLE001
        L.append(f"Q5 report failed: {e!r}")
    try:
        if q6o:
            L.append(f"\n## Q6 repeat samples (donors with >1 sample: {q6o.get('n_donors_multi')})")
            pf = q6o.get("paired_first_to_last", {})
            L.append("- first->last change: " + "; ".join(f"{c} {eci(pf[c], 'mean')}" for c in
                                                         ["days", "rate_ctrl", "rate_ctrl_isglow", "isg_z"] if c in pf))
            if "frac_still_flag_high_when_isg_falls" in q6o:
                L.append(f"- donors whose ISG falls: {q6o['n_isg_falls']}; flag rate > healthy q95 at last sample "
                         f"{f3(q6o['frac_still_flag_high_when_isg_falls'])} (first {f3(q6o['frac_first_flag_high_when_isg_falls'])})")
            for k in ["within_rho_rate_days", "within_rho_isg_days", "within_rho_rate_isg"]:
                if q6o.get(k):
                    L.append(f"- {k}: {eci(q6o[k], 'rho')}")
    except Exception as e:  # noqa: BLE001
        L.append(f"Q6 report failed: {e!r}")
    L.append("\nDetails: Q0-Q6 csv/png, obs_columns.txt, covid_programs.json, results.json.")
    txt = "\n".join(L) + "\n"
    # keep RESULTS.md < 8 KB: drop lower-priority lines first (all are in results.json / csv)
    for pat in [r"^- within_rho_rate_days", r"^- within_rho_isg_days", r"^- rho\(sev, rate_ctrl\) - rho\(sev, isgdet",
                r"^- severe/critical flags by lineage", r"^- rate_ctrl_std \| all", r"^- rate_ctrl_isglow \| all",
                r"^- non-COVID", r"^- COVID programs written"]:
        if len(txt.encode()) <= 8000:
            break
        L = [x for x in L if not re.match(pat, x)]
        txt = "\n".join(L) + "\n"
    if len(txt.encode()) > 8000:
        log(f"RESULTS.md is {len(txt.encode())} bytes (> 8 KB); details in results.json")
    open(os.path.join(OUT, "RESULTS.md"), "w").write(txt)
    save_results()


def main():
    gs = load_genesets()
    keys = dict(isg=detect_key(gs, args.isg_key, [r"^ISG_typeI$", r"ISG_typeI", r"ISG"]),
                gc=detect_key(gs, args.gc_key, [r"^glucocorticoid_core$", r"glucocort"]),
                ieg=detect_key(gs, args.ieg_key, [r"^dissociation_IEG$", r"dissoc", r"IEG"]),
                ifng=detect_key(gs, None, [r"IFN_gamma"]))
    R["keys"] = keys
    lp_path = args.lupus_programs if os.path.isabs(args.lupus_programs) else os.path.join(ROOT, args.lupus_programs)
    lupus_progs = None
    try:
        lupus_progs = read_programs(lp_path)
        log(f"lupus programs {lp_path}: {None if lupus_progs is None else list(lupus_progs)}")
    except Exception as e:  # noqa: BLE001
        log(f"reading lupus programs failed: {e!r}")
    extra = {f"LUPUS_{k}": v for k, v in (lupus_progs or {}).items()}
    isg_ieg = set()
    for k, v in gs.items():
        if re.search(r"ISG|IFN|SIGLEC1|dissoc|IEG|heat", k, re.I):
            isg_ieg |= set(v)
    for k, v in list(extra.items()):
        nv = [g for g in v if g not in isg_ieg]
        if 3 <= len(nv) < len(v):
            extra[k + "__noISG"] = nv
        elif len(nv) < 3:  # essentially an ISG/IEG program: cannot count as ISG-independent cross-disease evidence
            ISG_DOMINATED.add(k[6:])
    R["lupus_isg_dominated"] = sorted(ISG_DOMINATED)
    if lupus_progs:
        lupus_progs = {k[6:]: v for k, v in extra.items()}
    r0 = load_all(gs)
    if r0 is None:
        write_results(None, None, None, None, None, None)
        return
    D, H, seeds = r0
    r1 = expression(D, H, gs, keys, extra)
    X, var, E, rowpos, used = r1 if r1 else (None, H.var, pd.DataFrame(index=[]), None, {})
    r2 = build_tables(D, E)
    if r2 is None:
        write_results(None, None, None, None, None, None)
        return
    ST, DN, Dp, w = r2
    q1o = q1(DN)
    q2o = q2(DN)
    q3o = q3(Dp, DN, w)
    r4 = q4(Dp, DN, X, var, E, rowpos, gs, extra) if X is not None else None
    q4o, DE = r4 if r4 else (None, None)
    q5o = q5(DE, DN, q4o, gs, var, lupus_progs, used) if DE is not None else None
    q6o = q6(ST, DN)
    write_results(q1o, q2o, q3o, q4o, q5o, q6o)
    log(f"done in {time.time() - T0:.0f}s; errors: {list(R['errors'])}")


if __name__ == "__main__":
    main()
