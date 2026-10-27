#!/usr/bin/env python
"""ARCANE deep biological analysis of the full Perez et al. (2022) lupus cohort.

Run from the project directory (~/arcane):   python lupus_deep.py [--tag lupusfull] [--genesets genesets.json]

Inputs (relative to --root, default "."):
  data/std/<tag>.h5ad                 raw counts, all CELLxGENE obs columns + label/parent/batch/role/condition
  res/<tag>_seed{s}_cells.npz         per-cell ARCANE outputs for the query cells (seeds 0..2)
  emb/<tag>_seed{s}_full.npz          query/reference latent embeddings (U = scVI, S = scANVI)
  genesets.json (optional)            {"set name": [gene symbols]}; built-in defaults otherwise

Outputs: <out> (default summary/lupusdeep/): lupusdeep.json, many CSVs, PNG figures, RESULTS.md, log.txt.
Every analysis section is wrapped in try/except; failures are logged in lupusdeep.json["errors"].
Only numpy/pandas/scipy/scanpy/statsmodels/sklearn/matplotlib (+ leidenalg/igraph) are used.
"""
import argparse
import glob
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

# ----------------------------------------------------------------------------------------------
# arguments, logging, result store
# ----------------------------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--tag", default="lupusfull")
ap.add_argument("--h5ad", default=None, help="default <root>/data/std/<tag>.h5ad")
ap.add_argument("--genesets", default="genesets.json")
ap.add_argument("--out", default="summary/lupusdeep")
ap.add_argument("--seed0", type=int, default=0, help="seed used for cell-level detail")
ap.add_argument("--isg_key", default=None, help="gene-set name of the type I ISG set (auto-detected)")
ap.add_argument("--diss_key", default=None, help="gene-set name of the dissociation/IEG set (auto-detected)")
ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--nperm", type=int, default=2000)
ap.add_argument("--max_de", type=int, default=20000, help="max cells per DE comparison (both groups)")
ap.add_argument("--glm_cells", type=int, default=250000, help="cells subsampled for cell-level GLMs")
ap.add_argument("--min_flag_sub", type=int, default=300, help="min flagged cells of a fine type to sub-cluster")
ap.add_argument("--max_sub_cells", type=int, default=30000)
ap.add_argument("--max_sub_types", type=int, default=14)
ap.add_argument("--leiden_res", type=float, default=0.6)
ap.add_argument("--min_donors_state", type=int, default=5)
ap.add_argument("--min_cells_per_donor_state", type=int, default=3)
ap.add_argument("--fast_scores", action="store_true", help="own chunk-free implementation of score_genes")
ap.add_argument("--skip_de", action="store_true")
ap.add_argument("--scrublet", action="store_true",
                help="run scanpy scrublet per lane/sample on raw counts (doublet control; slower)")
ap.add_argument("--skip_controls", action="store_true", help="skip control sections 10-16")
args = ap.parse_args()

ROOT = args.root
OUT = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
os.makedirs(OUT, exist_ok=True)
LOGF = open(os.path.join(OUT, "log.txt"), "w")
RNG = np.random.default_rng(0)
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n")
    LOGF.flush()


R = {"args": vars(args), "errors": {}, "timing_s": {}}


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Series,)):
        return o.to_dict()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def clean(o):
    """Recursively replace NaN/inf floats by None so the JSON is valid."""
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
    return o


def save_results():
    with open(os.path.join(OUT, "lupusdeep.json"), "w") as f:
        json.dump(clean(R), f, indent=1, default=jsonable)


def section(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            log(f"\n=== [{time.time() - T0:7.0f}s] {name}")
            try:
                out = fn(*a, **k)
            except Exception as e:  # noqa: BLE001 - keep going, log everything
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
                     "figure.dpi": 100, "savefig.dpi": 200})
COL = {"healthy": "#4A7FB0", "SLE": "#C8553D"}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def wcsv(df, name, index=False):
    df.to_csv(os.path.join(OUT, name), index=index)


# ----------------------------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------------------------
def spearman_boot(x, y, nboot=None, rng=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 4:
        return dict(n=n, rho=None, p=None, ci_lo=None, ci_hi=None)
    rho, p = stats.spearmanr(x, y)
    nboot = nboot or args.nboot
    rng = rng or np.random.default_rng(1)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        bs.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if len(bs) > 10 else (np.nan, np.nan))
    return dict(n=int(n), rho=float(rho), p=float(p), ci_lo=float(lo), ci_hi=float(hi))


def boot_mean_ci(v, nboot=None, rng=None):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return dict(n=len(v), mean=float(np.mean(v)) if len(v) else None, ci_lo=None, ci_hi=None)
    rng = rng or np.random.default_rng(2)
    nboot = nboot or args.nboot
    bs = rng.choice(v, (nboot, len(v)), replace=True).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    try:
        pw = float(stats.wilcoxon(v).pvalue)
    except Exception:
        pw = None
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), ci_lo=float(lo), ci_hi=float(hi),
                wilcoxon_p_vs0=pw)


def mwu(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(n_a=len(a), n_b=len(b), p=None)
    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    auc = stats.mannwhitneyu(a, b).statistic / (len(a) * len(b))
    return dict(n_a=int(len(a)), n_b=int(len(b)), mean_a=float(a.mean()), mean_b=float(b.mean()),
                median_a=float(np.median(a)), median_b=float(np.median(b)), auc_a_gt_b=float(auc), p=float(p))


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


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def logit_frac(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    return np.log((k + 0.5) / (n - k + 0.5))


def zs(v):
    v = np.asarray(v, float)
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 0 else 1.0)


def coef_table(fit, keep=None):
    rows = []
    ci = fit.conf_int()
    for t in fit.params.index:
        if keep is not None and not any(k in t for k in keep):
            continue
        rows.append(dict(term=t, coef=float(fit.params[t]), se=float(fit.bse[t]), p=float(fit.pvalues[t]),
                         ci_lo=float(ci.loc[t, 0]), ci_hi=float(ci.loc[t, 1])))
    return rows


def collapse_levels(s, min_n=3, na="unknown"):
    s = s.astype(object).where(s.notna(), na).astype(str)
    vc = s.value_counts()
    rare = vc[vc < min_n].index
    return s.where(~s.isin(rare), "other")


# ----------------------------------------------------------------------------------------------
# gene sets
# ----------------------------------------------------------------------------------------------
DEFAULT_SETS = {
    "ISG_typeI": ["ISG15", "IFI6", "IFI27", "IFIT1", "IFIT2", "IFIT3", "IFIT5", "MX1", "MX2", "OAS1", "OAS2",
                  "OAS3", "OASL", "IFI44L", "IFI44", "RSAD2", "LY6E", "IFITM3", "XAF1", "HERC5", "HERC6",
                  "EPSTI1", "IRF7", "ISG20", "SAMD9L", "CMPK2", "USP18", "SIGLEC1", "RTP4", "PLSCR1", "SPATS2L"],
    "IFN_gamma": ["GBP1", "GBP2", "GBP4", "GBP5", "CXCL9", "CXCL10", "IDO1", "IRF1", "STAT1", "CIITA", "CD274",
                  "SOCS1", "WARS1", "WARS", "TAP1", "PSMB9", "PSMB8", "APOL6", "SERPING1", "FCGR1A", "BATF2"],
    "Dissociation_IEG": ["FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "EGR2", "EGR3", "ATF3", "IER2", "IER3",
                         "IER5", "ZFP36", "ZFP36L1", "ZFP36L2", "DUSP1", "NR4A1", "NR4A2", "NR4A3", "KLF2", "KLF4",
                         "KLF6", "PPP1R15A", "CCNL1", "BTG2", "SOCS3", "CYR61", "CCN1", "MYC", "NFKBIA", "HSPA1A",
                         "HSPA1B", "DNAJB1", "HSPA8", "GADD45B", "RHOB", "SRF", "MCL1"],
    "Heat_shock": ["HSPA1A", "HSPA1B", "HSPA6", "HSPH1", "HSP90AA1", "HSP90AB1", "DNAJB1", "DNAJA1", "HSPD1",
                   "HSPE1", "BAG3", "HSPB1", "CRYAB"],
    "Glucocorticoid": ["TSC22D3", "FKBP5", "ZBTB16", "PER1", "DDIT4", "KLF9", "CXCR4", "IRS2", "CD163", "VSIG4",
                       "ADAMTS2", "MAOA", "SESN1", "DUSP1", "SAP30", "CPM", "AREG", "IL1R2", "ADORA3"],
    "NFkB_TNF_inflammatory": ["NFKBIA", "NFKBIZ", "TNFAIP3", "NFKB1", "NFKB2", "REL", "RELB", "IL1B", "CCL3",
                              "CCL4", "CXCL8", "TNF", "PTGS2", "ICAM1", "CD83", "IL6", "CCL20", "IRAK3", "SOD2"],
    "Neutrophil_LDG": ["ELANE", "MPO", "DEFA1", "DEFA3", "DEFA4", "LTF", "CAMP", "LCN2", "CEACAM8", "MMP8", "MMP9",
                       "CTSG", "AZU1", "PRTN3", "BPI", "OLFM4", "CEACAM6", "ARG1"],
    "Platelet": ["PPBP", "PF4", "GP9", "ITGA2B", "TUBB1", "GNG11", "CLU", "SPARC", "NRGN", "CAVIN2", "TREML1"],
    "Erythroid": ["HBB", "HBA1", "HBA2", "HBD", "ALAS2", "CA1", "SLC4A1"],
    "Proliferation": ["MKI67", "TOP2A", "TYMS", "STMN1", "CDK1", "PCNA", "MCM2", "MCM7", "CCNB1", "CCNB2", "BIRC5",
                      "TUBB", "RRM2", "UBE2C", "CENPF", "NUSAP1", "H2AZ1", "TK1"],
    "Plasmablast": ["JCHAIN", "MZB1", "XBP1", "PRDM1", "TNFRSF17", "SDC1", "IGHG1", "IGHA1", "TXNDC5", "DERL3"],
    "Cytotoxic": ["GZMB", "GZMA", "GZMH", "GZMK", "PRF1", "NKG7", "GNLY", "CCL5", "CST7", "KLRD1"],
    "T_exhaustion_activation": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "CD38", "ICOS", "CXCL13",
                                "MAF", "IL21", "CD27"],
    "Atypical_ABC_B": ["ITGAX", "TBX21", "FCRL5", "FCRL3", "ZEB2", "CD86", "FCRL4", "SOX5", "CXCR3", "FGR",
                       "ENC1", "AICDA"],
    "MHC_II": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "CD74", "HLA-DMA", "HLA-DMB"],
    "Hypoxia_glycolysis": ["VEGFA", "SLC2A1", "SLC2A3", "PGK1", "BNIP3", "LDHA", "ENO1", "ADM", "PFKFB3", "NDRG1",
                           "ANKRD37", "P4HA1", "MXI1", "BHLHE40"],
    "Monocyte_S100_alarmin": ["S100A8", "S100A9", "S100A12", "VCAN", "LYZ", "FCN1", "CD14", "MNDA", "CSF3R"],
    "Apoptosis_stress": ["BAX", "BBC3", "PMAIP1", "CDKN1A", "MDM2", "GADD45A", "TP53INP1", "SESN2", "TNFRSF10B",
                         "FAS", "DDIT3", "XBP1", "ATF4", "HERPUD1"],
}
IEG_TARGETS = ["JUN", "FOS", "ZFP36", "JUNB", "FOSB", "EGR1", "DUSP1", "IER2", "KLF6", "NR4A1"]


def load_genesets():
    path = args.genesets if os.path.isabs(args.genesets) else os.path.join(ROOT, args.genesets)
    if os.path.exists(path):
        gs = json.load(open(path))
        gs = {str(k): [str(g) for g in v] for k, v in gs.items() if isinstance(v, (list, tuple))}
        src = path
    else:
        gs, src = dict(DEFAULT_SETS), "built-in defaults"
    log("gene sets from", src, ":", len(gs), "sets")

    def detect(user, pats, default):
        if user:
            if user not in gs:
                raise KeyError(f"--key {user} not in gene sets")
            return user
        for k in gs:
            if any(re.search(p, k, re.I) for p in pats):
                return k
        gs[default] = DEFAULT_SETS[default]
        log(f"  no set matching {pats}; adding default {default}")
        return default
    isg = detect(args.isg_key, [r"^isg", r"type.?i.?(ifn|interferon)", r"ifn.?(alpha|a|i)\b", r"interferon.?alpha"],
                 "ISG_typeI")
    diss = detect(args.diss_key, [r"dissoc", r"\bieg", r"immediate.?early"], "Dissociation_IEG")
    # leave-out "core" dissociation score: dissociation set minus the IEG target genes tested individually
    gs["DISS_core"] = [g for g in gs[diss] if g not in IEG_TARGETS]
    # split-half ISG scores (alternating genes) for the residual-confounding test of section 13
    gs["ISG_halfA"] = gs[isg][0::2]
    gs["ISG_halfB"] = gs[isg][1::2]
    return gs, isg, diss, src


def safe_name(s):
    return re.sub(r"[^0-9A-Za-z_]", "_", s)


# ----------------------------------------------------------------------------------------------
# 0. load ARCANE outputs
# ----------------------------------------------------------------------------------------------
def load_res():
    pat = os.path.join(ROOT, "res2", f"{args.tag}_seed*_cells.npz")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no files matching {pat}")
    per = {}
    for f in files:
        m = re.search(r"seed(\d+)_cells\.npz$", f)
        if not m:
            continue
        s = int(m.group(1))
        C = np.load(f, allow_pickle=True)
        need = ["names", "y", "cond", "batch", "novelty", "unknown"]
        miss = [k for k in need if k not in C.files]
        if miss:
            raise KeyError(f"{f} lacks {miss}")
        classes = np.asarray(C["classes"]).astype(str)

        def lab(v):
            v = np.asarray(v)
            if v.dtype.kind in "iu":
                return np.where(v >= 0, classes[np.clip(v, 0, len(classes) - 1)], "none")
            return v.astype(str)
        df = pd.DataFrame(index=pd.Index(np.asarray(C["names"]).astype(str), name="cell"))
        df["y"] = lab(C["y"])
        df["cond_raw"] = np.asarray(C["cond"]).astype(str)
        df["donor"] = np.asarray(C["batch"]).astype(str)
        df["novelty"] = np.asarray(C["novelty"], float)
        df["flag"] = np.asarray(C["unknown"]).astype(bool)
        for k in ["set_size", "scanvi_maxprob", "knn_dist_U", "knn_dist_S", "pmax_S", "pmax_U"]:
            if k in C.files:
                df[k] = np.asarray(C[k])
        for k in C.files:
            if k.startswith("pred_"):
                df[k] = lab(C[k])
        if "pvals" in C.files:
            pv = np.asarray(C["pvals"], dtype=np.float32)
            df["conf_pmax"] = pv.max(1)
            df["closest_label"] = classes[pv.argmax(1)]
        per[s] = (df, classes)
        log(f"seed {s}: {len(df)} query cells, flagged {df.flag.mean():.4f}, keys {C.files}")
    if args.seed0 not in per:
        args.seed0 = sorted(per)[0]
        log("seed0 not found; using", args.seed0)
    return per


# ----------------------------------------------------------------------------------------------
# 1. obs column detection
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


@section("1_obs_columns")
def detect_columns(obs):
    info = {}
    log(f"obs: {obs.shape[0]} cells x {obs.shape[1]} columns")
    for c in obs.columns:
        vc = obs[c].astype(str).value_counts(dropna=False)
        info[c] = dict(n_unique=int(len(vc)), top=vc.head(25).to_dict())
        log(f"--- {c}  ({len(vc)} unique)")
        log(vc.head(25).to_string())
    json.dump(info, open(os.path.join(OUT, "obs_columns.json"), "w"), indent=1, default=jsonable)
    cols = list(obs.columns)
    det = dict(
        donor=find_col(cols, ["donor_id", "ind_cov", "batch", "donor"], [r"donor", r"^ind"]),
        disease=find_col(cols, ["condition", "disease"], [r"disease", r"condition"]),
        cohort=find_col(cols, ["Processing_Cohort", "processing_cohort", "cohort"], [r"process.*cohort", r"cohort"]),
        lane=find_col(cols, ["batch_cov", "pool", "pool_id"], [r"batch_cov", r"^pool", r"lane"]),
        ancestry=find_col(cols, ["self_reported_ethnicity", "pop_cov", "ancestry", "ethnicity", "race"],
                          [r"ethnic", r"ancestr", r"pop_cov", r"race"]),
        sex=find_col(cols, ["sex", "Sex", "sex_cov", "gender"], [r"^sex", r"gender"]),
        age=find_col(cols, ["Age", "age", "age_cov", "age_years"], [r"^age"]),
        dev_stage=find_col(cols, ["development_stage"], [r"development_stage"]),
        sample=find_col(cols, ["sample_uuid", "sample_id", "sample"], [r"sample"]),
        label=find_col(cols, ["label"]),
        parent=find_col(cols, ["parent"]),
        role=find_col(cols, ["role"]),
    )
    # clinical / status / treatment fields
    clin = [c for c in cols if re.search(r"status|sledai|flare|activity|treat|medic|drug|steroid|predni|hcq|"
                                         r"hydroxy|nephrit|dsdna|complement|c3|c4|severity", c, re.I)
            and not re.search(r"ontology", c, re.I) and obs[c].nunique() < 5000]
    # prefer a 'Status'-like field with > 2 levels (Managed / Flare / Treated)
    status = None
    for c in clin:
        if re.search(r"^status$|disease_state|flare", c, re.I) and obs[c].nunique() > 2:
            status = c
            break
    if status is None:
        for c in clin:
            if obs[c].nunique() > 2 and obs[c].nunique() < 20:
                status = c
                break
    det["status"] = status
    det["clinical_candidates"] = clin
    log("detected columns:", json.dumps(det, indent=1))
    R["columns"] = det
    return det


def donor_table_meta(obs, det, donors):
    """One row per donor of donor-level covariates (mode); flags non-constant columns."""
    o = obs.copy()
    dcol = "batch" if "batch" in o.columns else det["donor"]
    o["_donor"] = o[dcol].astype(str)
    o = o[o._donor.isin(donors)]
    meta = pd.DataFrame(index=pd.Index(sorted(set(donors)), name="donor"))
    nonconst = {}
    for key in ["cohort", "lane", "ancestry", "sex", "age", "dev_stage", "status", "sample"] + \
               [c for c in (det.get("clinical_candidates") or []) if c != det.get("status")]:
        c = det.get(key, key if key in o.columns else None)
        if c is None or c not in o.columns:
            continue
        g = o.groupby("_donor")[c]
        nun = g.nunique()
        nonconst[c] = int((nun > 1).sum())
        mode = g.agg(lambda s: s.astype(str).value_counts().index[0] if s.notna().any() else np.nan)
        name = key if key in ["cohort", "lane", "ancestry", "sex", "age", "dev_stage", "status", "sample"] else \
            "clin_" + safe_name(c)
        meta[name] = mode.reindex(meta.index)
        if key == "sample":
            meta["n_samples"] = nun.reindex(meta.index)
        if key == "status":
            meta["n_status"] = nun.reindex(meta.index)
    if "age" in meta:
        meta["age_num"] = meta["age"].map(parse_age)
    elif "dev_stage" in meta:
        meta["age_num"] = meta["dev_stage"].map(parse_age)
    R["donor_nonconstant_columns"] = nonconst
    return meta


# ----------------------------------------------------------------------------------------------
# 2. expression: QC, normalization, module scores
# ----------------------------------------------------------------------------------------------
def score_fast(A, genes, name, rng, n_bins=25, ctrl_size=50, gene_means=None):
    """Same algorithm as scanpy.tl.score_genes (expression-binned control genes), computed directly."""
    X = A.X
    gi = np.where(A.var_names.isin(genes))[0]
    if gene_means is None:
        gene_means = np.asarray(X.mean(0)).ravel()
    gm = pd.Series(gene_means)
    n_items = int(np.round(len(gm) / (n_bins - 1)))
    cut = (gm.rank(method="min") // n_items).astype(int).values
    ctrl = set()
    for b in np.unique(cut[gi]):
        pool = np.where(cut == b)[0]
        pool = pool[~np.isin(pool, gi)]
        rng.shuffle(pool)
        ctrl.update(pool[:ctrl_size].tolist())
    ctrl = np.array(sorted(ctrl))
    s1 = np.asarray(X[:, gi].mean(1)).ravel()
    s0 = np.asarray(X[:, ctrl].mean(1)).ravel() if len(ctrl) else 0
    A.obs[name] = (s1 - s0).astype(np.float32)


@section("2_expression_scores")
def expression_scores(A, gs):
    import scanpy as sc
    X = A.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()
    # use counts layer if X is not integer counts
    samp = X.data[: min(len(X.data), 200000)]
    if len(samp) and not np.allclose(samp, np.round(samp)):
        if "counts" in A.layers:
            log("X is not integer; using layers['counts']")
            X = sp.csr_matrix(A.layers["counts"]).tocsr()
        else:
            raise ValueError("X does not contain raw counts and no 'counts' layer")
    X = X.astype(np.float32)
    X.eliminate_zeros()
    if args.scrublet:
        try:
            import anndata as ad
            bk = next((c for c in [DET.get("lane"), DET.get("sample"), "batch"] if c and c in A.obs), None)
            tmp = ad.AnnData(X=X.copy(), obs=A.obs[[bk]].copy() if bk else pd.DataFrame(index=A.obs_names))
            sc.pp.scrublet(tmp, batch_key=bk, random_state=0)
            A.obs["doublet_score"] = tmp.obs["doublet_score"].values
            A.obs["predicted_doublet"] = tmp.obs["predicted_doublet"].values
            log(f"scrublet (batch_key={bk}): predicted doublets {A.obs.predicted_doublet.mean():.4f}")
            del tmp
        except Exception as e:
            log(f"scrublet failed ({e!r}); continuing with the marker-based doublet proxy only")
    var = pd.Index(A.var_names.astype(str))
    tot = np.asarray(X.sum(1)).ravel()
    ngen = np.diff(X.indptr)
    mt = np.asarray(var.str.upper().str.startswith("MT-"))
    rb = np.asarray(var.str.upper().str.match(r"^RP[SL]\d"))
    A.obs["total_counts"] = tot
    A.obs["n_genes"] = ngen
    A.obs["pct_mito"] = 100 * np.asarray(X[:, mt].sum(1)).ravel() / np.maximum(tot, 1)
    A.obs["pct_ribo"] = 100 * np.asarray(X[:, rb].sum(1)).ravel() / np.maximum(tot, 1)
    log("QC medians:", A.obs[["total_counts", "n_genes", "pct_mito", "pct_ribo"]].median().round(2).to_dict())
    # in-place log-normalization (target 1e4)
    scale = (1e4 / np.maximum(tot, 1)).astype(np.float32)
    X.data *= np.repeat(scale, np.diff(X.indptr))
    np.log1p(X.data, out=X.data)
    A.X = X
    # drop rarely detected genes
    det = np.bincount(X.indices, minlength=X.shape[1])
    keep = det >= 20
    log(f"genes kept (detected in >=20 cells): {keep.sum()} / {len(keep)}")
    A._inplace_subset_var(keep)
    if A.layers:
        for k in list(A.layers.keys()):
            del A.layers[k]
    used = {}
    gm = np.asarray(A.X.mean(0)).ravel()
    rng = np.random.default_rng(0)
    for name, genes in gs.items():
        g = [x for x in genes if x in A.var_names]
        used[name] = g
        if len(g) < 3:
            log(f"  gene set {name}: only {len(g)} genes present; skipped")
            continue
        sn = "gs_" + safe_name(name)
        done = False
        if not args.fast_scores:
            try:
                sc.tl.score_genes(A, g, score_name=sn, ctrl_size=50, random_state=0, use_raw=False)
                done = True
            except Exception as e:
                log(f"  scanpy score_genes failed for {name} ({e!r}); using fast implementation")
        if not done:
            score_fast(A, g, sn, rng, gene_means=gm)
        log(f"  scored {name}: {len(g)} genes; mean {A.obs[sn].mean():.4f}")
    # individual IEG target genes (log-normalized expression)
    tg = [g for g in IEG_TARGETS if g in A.var_names]
    if tg:
        M = A[:, tg].X
        M = M.toarray() if sp.issparse(M) else np.asarray(M)
        for j, g in enumerate(tg):
            A.obs["g_" + safe_name(g)] = M[:, j].astype(np.float32)
    try:
        control_genes(A)
    except Exception as e:
        log(f"control genes failed: {e!r}")
    R["genesets_used"] = used
    return used


# ----------------------------------------------------------------------------------------------
# build analysis tables
# ----------------------------------------------------------------------------------------------
def is_sle_str(s):
    return bool(re.search(r"lupus|\bsle\b", str(s), re.I))


def build_cells(per, A, det, meta):
    s0 = args.seed0
    D = per[s0][0].copy()
    common = D.index
    for s, (df, _) in per.items():
        common = common.intersection(df.index)
    if len(common) < len(D):
        log(f"WARNING: {len(D) - len(common)} seed-{s0} cells missing from other seeds; using intersection")
    D = D.loc[common]
    fl = np.vstack([per[s][0].loc[common, "flag"].values for s in sorted(per)])
    nv = np.vstack([per[s][0].loc[common, "novelty"].values for s in sorted(per)])
    D["flag_mean"] = fl.mean(0)
    D["novelty_mean"] = nv.mean(0)
    for s in sorted(per):
        D[f"flag_s{s}"] = fl[sorted(per).index(s)]
    D["is_sle"] = D.cond_raw.map(is_sle_str).astype(int)
    D["cond"] = np.where(D.is_sle == 1, "SLE", "healthy")
    missing = ~D.index.isin(A.obs_names)
    if missing.any():
        raise KeyError(f"{missing.sum()} query cells not in h5ad obs_names (e.g. {D.index[missing][:3].tolist()})")
    ob = A.obs.loc[D.index]
    for c in ob.columns:
        if c.startswith("gs_") or c.startswith("g_") or c in ["total_counts", "n_genes", "pct_mito", "pct_ribo"]:
            D[c] = ob[c].values
    if det.get("parent") and det["parent"] in ob:
        D["parent"] = ob[det["parent"]].astype(str).values
    else:
        D["parent"] = D["y"]
    D = D.join(meta, on="donor")
    D["log_counts"] = np.log10(np.maximum(D.total_counts.astype(float), 1))
    D["log_genes"] = np.log10(np.maximum(D.n_genes.astype(float), 1))
    D["log_novelty"] = np.log(np.maximum(D.novelty_mean.astype(float), 1e-6))
    return D


# ----------------------------------------------------------------------------------------------
# 3. donor-level and cell-level association models
# ----------------------------------------------------------------------------------------------
def covariate_frame(DN):
    """Standardized covariates for formulas; returns (frame, list of terms)."""
    X = pd.DataFrame(index=DN.index)
    terms = []
    for k in ["cohort", "ancestry", "sex"]:
        if k in DN:
            v = collapse_levels(DN[k], 3)
            if v.nunique() > 1:
                X[k] = v
                terms.append(f"C({k})")
    if "age_num" in DN and DN.age_num.notna().sum() > 10:
        a = DN.age_num.astype(float)
        X["age_z"] = zs(a.fillna(a.median()))
        terms.append("age_z")
    return X, terms


@section("3_donor_and_cell_models")
def donor_models(D, DN, isg_col, diss_col):
    out = {}
    R["S3"] = out
    # ---- within-group donor correlations (flag fraction vs ISG); healthy here = held-out healthy query donors
    corr = {}
    for grp, sub in [("SLE", DN[DN.is_sle == 1]), ("healthy_query", DN[DN.is_sle == 0]), ("all", DN)]:
        corr[grp] = {
            "flag_vs_isg_typeadj": spearman_boot(sub.flag_frac, sub.isg),
            "flag_vs_isg_raw": spearman_boot(sub.flag_frac, sub.isg_raw),
            "novelty_vs_isg_typeadj": spearman_boot(sub.novelty_mean, sub.isg),
            "flag_vs_diss": spearman_boot(sub.flag_frac, sub.diss),
        }
        for s in [c for c in DN.columns if re.match(r"flag_frac_s\d+$", c)]:
            corr[grp][f"{s}_vs_isg"] = spearman_boot(sub[s], sub.isg, nboot=300)
    out["donor_spearman"] = corr
    log("donor Spearman (flag vs type-adjusted ISG):",
        {g: (round(v["flag_vs_isg_typeadj"]["rho"] or np.nan, 3), v["flag_vs_isg_typeadj"]["n"]) for g, v in corr.items()})
    # ---- within-patient (cell-level within donor x fine type) correlation novelty vs ISG
    rows = []
    for (d, t), g in D.groupby(["donor", "y"]):
        if len(g) < 30 or g[isg_col].std() == 0:
            continue
        r = stats.spearmanr(g.novelty_mean, g[isg_col])[0]
        auc = np.nan
        if g.flag.sum() >= 5 and (~g.flag).sum() >= 5:
            auc = stats.mannwhitneyu(g.loc[g.flag, isg_col], g.loc[~g.flag, isg_col]).statistic / (
                g.flag.sum() * (~g.flag).sum())
        rows.append(dict(donor=d, type=t, n=len(g), cond=g.cond.iloc[0], rho=r, auc_isg_flag=auc))
    W = pd.DataFrame(rows)
    wcsv(W, "within_donor_type_novelty_isg.csv")
    wd = W.groupby(["donor", "cond"]).apply(
        lambda g: pd.Series(dict(rho=np.average(g.rho.fillna(0), weights=g.n),
                                 auc=np.nanmean(g.auc_isg_flag) if g.auc_isg_flag.notna().any() else np.nan))
    ).reset_index()
    out["within_patient"] = {c: dict(rho_novelty_isg=boot_mean_ci(g.rho), auc_isg_flagged=boot_mean_ci(g.auc.dropna()))
                             for c, g in wd.groupby("cond")}
    out["within_patient"]["SLE_vs_healthy_rho"] = mwu(wd[wd.cond == "SLE"].rho, wd[wd.cond == "healthy"].rho)
    wcsv(wd, "within_patient_donor_summary.csv")
    # per fine type, donor-level correlation in SLE
    pt = []
    for t, g in D[D.is_sle == 1].groupby("y"):
        dd = g.groupby("donor").agg(f=("flag_mean", "mean"), i=(isg_col, "mean"), n=("flag", "size"))
        dd = dd[dd.n >= 20]
        if len(dd) >= 10:
            r = spearman_boot(dd.f, dd.i, nboot=500)
            r["type"] = t
            pt.append(r)
    if pt:
        pt = pd.DataFrame(pt)
        pt["q"] = bh(pt.p)
        wcsv(pt, "donor_spearman_by_type_SLE.csv")
        out["donor_spearman_by_type_SLE"] = pt.to_dict(orient="records")
    R["S3"] = out
    save_results()
    # figure: donor flag fraction vs ISG
    try:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for c, gg in DN.groupby("cond"):
            r = corr["SLE" if c == "SLE" else "healthy_query"]["flag_vs_isg_typeadj"]["rho"]
            ax.scatter(gg.isg, 100 * gg.flag_frac, s=40, alpha=0.8, c=COL[c], label=f"{c} (rho={fmt(r, 2)})")
        ax.set_xlabel("Donor ISG score (type-adjusted z)")
        ax.set_ylabel("Unknown flags (% cells)")
        ax.legend(frameon=False)
        savefig(fig, "fig_donor_flag_vs_isg.png")
    except Exception as e:
        log("donor figure failed", e)
    # ---- donor OLS
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    X, terms = covariate_frame(DN)
    F = pd.concat([DN[["flag_logit", "is_sle", "isg", "isg_raw", "diss", "n_cells"]], X], axis=1)
    F["isg_z"] = zs(F.isg)
    F["diss_z"] = zs(F.diss)
    models = {
        "flag ~ SLE + covars": "flag_logit ~ is_sle" + "".join(" + " + t for t in terms),
        "flag ~ SLE + ISG + covars": "flag_logit ~ is_sle + isg_z" + "".join(" + " + t for t in terms),
        "flag ~ SLE*ISG + diss + covars": "flag_logit ~ is_sle * isg_z + diss_z" + "".join(" + " + t for t in terms),
    }
    ols = {}
    for nm, fml in models.items():
        try:
            fit = smf.ols(fml, data=F).fit(cov_type="HC3")
            ols[nm] = dict(formula=fml, n=int(fit.nobs), r2=float(fit.rsquared), terms=coef_table(fit))
        except Exception as e:
            ols[nm] = dict(formula=fml, error=repr(e))
    # within-SLE donor models incl. status / clinical fields
    S = F[F.is_sle == 1].copy()
    st_terms = []
    if "status" in DN:
        st = collapse_levels(DN.loc[S.index, "status"], 3)
        if st.nunique() > 1:
            S["status"] = st
            st_terms.append("C(status)")
    for c in [c for c in DN.columns if c.startswith("clin_")]:
        v = DN.loc[S.index, c]
        num = pd.to_numeric(v, errors="coerce")
        if num.notna().mean() > 0.7 and num.nunique() > 5:
            S[c] = zs(num.fillna(num.median()))
            st_terms.append(c)
        elif 1 < v.nunique() <= 8:
            S[c] = collapse_levels(v, 3)
            if S[c].nunique() > 1:
                st_terms.append(f"C({c})")
    st_terms = list(dict.fromkeys(st_terms))
    t_sle = [t for t in terms]
    for nm, fml in {"SLE only: flag ~ ISG + covars": "flag_logit ~ isg_z" + "".join(" + " + t for t in t_sle),
                    "SLE only: flag ~ ISG + diss + covars + status/clinical":
                        "flag_logit ~ isg_z + diss_z" + "".join(" + " + t for t in t_sle + st_terms)}.items():
        try:
            fit = smf.ols(fml, data=S).fit(cov_type="HC3")
            ols[nm] = dict(formula=fml, n=int(fit.nobs), r2=float(fit.rsquared), terms=coef_table(fit))
        except Exception as e:
            ols[nm] = dict(formula=fml, error=repr(e))
    # healthy-only
    H = F[F.is_sle == 0]
    try:
        fml = "flag_logit ~ isg_z + diss_z" + "".join(" + " + t for t in terms)
        fit = smf.ols(fml, data=H).fit(cov_type="HC3")
        ols["healthy only: flag ~ ISG + diss + covars"] = dict(formula=fml, n=int(fit.nobs), r2=float(fit.rsquared),
                                                               terms=coef_table(fit))
    except Exception as e:
        ols["healthy only: flag ~ ISG + diss + covars"] = dict(error=repr(e))
    out["donor_ols"] = ols
    # mixed model: random intercept for processing cohort
    if "cohort" in X:
        try:
            other = [t for t in terms if t != "C(cohort)"]
            fml = "flag_logit ~ is_sle + isg_z + diss_z" + "".join(" + " + t for t in other)
            fit = smf.mixedlm(fml, data=F, groups=F["cohort"]).fit(reml=True)
            out["donor_mixedlm_cohort_RE"] = dict(formula=fml, n=int(fit.nobs), terms=[
                dict(term=t, coef=float(fit.params[t]), se=float(fit.bse[t]), p=float(fit.pvalues[t]))
                for t in fit.params.index], cohort_var=float(fit.cov_re.iloc[0, 0]))
        except Exception as e:
            out["donor_mixedlm_cohort_RE"] = dict(error=repr(e))
    # ---- cell-level logistic GLM with donor-clustered SEs
    glm = {}
    Q = D.copy()
    if len(Q) > args.glm_cells:
        Q = Q.iloc[RNG.choice(len(Q), args.glm_cells, replace=False)]
    Q = Q.drop(columns=[c for c in X.columns if c in Q.columns]).join(X, on="donor")
    Q["flag_i"] = Q.flag.astype(int)
    Q["isg_z"] = zs(Q[isg_col])
    Q["isgt_z"] = zs(Q["isg_tz"])
    Q["diss_z"] = zs(Q[diss_col])
    Q["lc_z"] = zs(Q.log_counts)
    Q["lg_z"] = zs(Q.log_genes)
    Q["mito_z"] = zs(Q.pct_mito)
    Q["ftype"] = collapse_levels(Q.y, 200)
    cov_terms = [t for t in terms]
    base = "flag_i ~ is_sle + isgt_z + diss_z + lc_z + lg_z + mito_z + C(ftype)"
    for nm, data, fml in [
        ("all query cells", Q, base + "".join(" + " + t for t in cov_terms)),
        ("all query cells, no ISG", Q, base.replace(" + isgt_z", "") + "".join(" + " + t for t in cov_terms)),
        ("SLE cells", Q[Q.is_sle == 1], base.replace("is_sle + ", "") + "".join(" + " + t for t in cov_terms)),
        ("healthy query cells", Q[Q.is_sle == 0], base.replace("is_sle + ", "") + "".join(" + " + t for t in cov_terms)),
    ]:
        try:
            grp = pd.factorize(data.donor)[0]
            fit = smf.glm(fml, data=data, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": grp})
            glm[nm] = dict(formula=fml, n_cells=int(fit.nobs), n_donors=int(len(np.unique(grp))),
                           terms=coef_table(fit, keep=["is_sle", "isgt_z", "diss_z", "lc_z", "lg_z", "mito_z",
                                                       "cohort", "ancestry", "sex", "age_z"]))
        except Exception as e:
            glm[nm] = dict(formula=fml, error=repr(e))
    out["cell_glm_donor_clustered"] = glm
    R["S3"] = out
    return out


# ----------------------------------------------------------------------------------------------
# 3b. longitudinal (several samples per donor)
# ----------------------------------------------------------------------------------------------
@section("3b_within_patient_longitudinal")
def longitudinal(D, A_obs, det, isg_col):
    scol = det.get("sample")
    if not scol or scol not in A_obs:
        log("no sample column; skipped")
        return None
    D = D.copy()
    D["sample_id"] = A_obs.loc[D.index, scol].astype(str).values
    if det.get("status") and det["status"] in A_obs:
        D["sample_status"] = A_obs.loc[D.index, det["status"]].astype(str).values
    ns = D.groupby("donor").sample_id.nunique()
    multi = ns[ns >= 2].index
    log(f"donors with >=2 samples: {len(multi)}")
    if len(multi) < 3:
        R["S3b"] = dict(n_donors_multi=int(len(multi)))
        return None
    agg = {"flag": ("flag_mean", "mean"), "isg": ("isg_tz", "mean"), "n": ("flag", "size")}
    S = D[D.donor.isin(multi)].groupby(["donor", "sample_id"]).agg(**agg).reset_index()
    if "sample_status" in D:
        S = S.merge(D.groupby("sample_id").sample_status.first().reset_index(), on="sample_id", how="left")
    S = S[S.n >= 100]
    S["flag_c"] = S.flag - S.groupby("donor").flag.transform("mean")
    S["isg_c"] = S.isg - S.groupby("donor").isg.transform("mean")
    S = S[S.groupby("donor").sample_id.transform("count") >= 2]
    wcsv(S, "longitudinal_samples.csv")
    res = dict(n_donors=int(S.donor.nunique()), n_samples=int(len(S)),
               within_donor_spearman_centered=spearman_boot(S.flag_c, S.isg_c))
    R["S3b"] = res
    log(res)
    return res


# ----------------------------------------------------------------------------------------------
# 4. technical control: flagged rate by processing cohort
# ----------------------------------------------------------------------------------------------
@section("4_technical_cohort_control")
def cohort_control(DN, obs, det):
    out = {}
    for key in ["cohort", "lane"]:
        if key not in DN:
            continue
        rows = []
        for c, g in DN.groupby(key):
            h, s = g[g.is_sle == 0].flag_frac, g[g.is_sle == 1].flag_frac
            r = dict(level=c, n_healthy_query=int(len(h)), n_sle=int(len(s)),
                     healthy_median=float(h.median()) if len(h) else None,
                     sle_median=float(s.median()) if len(s) else None)
            if len(h) >= 2 and len(s) >= 2:
                r.update({"mwu_p": mwu(s, h)["p"], "auc_sle_gt_healthy": mwu(s, h)["auc_a_gt_b"]})
            rows.append(r)
        T = pd.DataFrame(rows)
        for c in ["healthy_median", "sle_median"]:
            T[c] = pd.to_numeric(T[c], errors="coerce")
        if key == "cohort" and det.get("cohort") and det.get("role") and det["role"] in obs:
            o = obs[[det["cohort"], det["role"], "batch" if "batch" in obs else det["donor"]]].drop_duplicates()
            o.columns = ["cohort", "role", "donor"]
            nref = o[o.role == "ref"].groupby("cohort").donor.nunique()
            T["n_ref_donors"] = T.level.map(nref).fillna(0).astype(int)
        wcsv(T, f"flag_by_{key}.csv")
        hq = DN[DN.is_sle == 0]
        groups = [g.flag_frac.values for _, g in hq.groupby(key) if len(g) >= 2]
        kw = stats.kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
        out[key] = dict(table=T.to_dict(orient="records"), kruskal_healthy_query_p=float(kw),
                        healthy_query_range_of_medians=[float(np.nanmin(T.healthy_median)), float(np.nanmax(T.healthy_median))]
                        if T.healthy_median.notna().any() else None)
        log(key, T.to_string())
    # donor crosstab (confounding check)
    if "cohort" in DN:
        ct = pd.crosstab(DN.cohort, DN.cond)
        out["cohort_by_condition_donors"] = ct.to_dict()
        if "ancestry" in DN:
            out["ancestry_by_condition_donors"] = pd.crosstab(DN.ancestry, DN.cond).to_dict()
        # figure
        lev = sorted(DN.cohort.astype(str).unique())
        fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(lev)), 5.5))
        for i, c in enumerate(lev):
            for j, (cn, off) in enumerate([("healthy", -0.18), ("SLE", 0.18)]):
                v = 100 * DN[(DN.cohort.astype(str) == c) & (DN.cond == cn)].flag_frac
                x = i + off + RNG.uniform(-0.07, 0.07, len(v))
                ax.scatter(x, v, s=28, c=COL[cn], alpha=0.75, label=cn if i == 0 else None)
                if len(v):
                    ax.plot([i + off - 0.13, i + off + 0.13], [v.median()] * 2, c="k", lw=2.5)
        ax.set_xticks(range(len(lev)))
        ax.set_xticklabels(lev)
        ax.set_xlabel("Processing cohort")
        ax.set_ylabel("Unknown flags (% cells)")
        ax.legend(frameon=False)
        savefig(fig, "fig_flag_by_cohort.png")
    R["S4"] = out
    return out


# ----------------------------------------------------------------------------------------------
# DE + enrichment helpers
# ----------------------------------------------------------------------------------------------
def run_de(A, cells_a, cells_b, max_n=None, n_top=40):
    import scanpy as sc
    max_n = max_n or args.max_de
    rng = np.random.default_rng(3)
    ha = max_n // 2
    cells_a = np.asarray(cells_a)
    cells_b = np.asarray(cells_b)
    if len(cells_a) > ha:
        cells_a = rng.choice(cells_a, ha, replace=False)
    if len(cells_b) > max_n - len(cells_a):
        cells_b = rng.choice(cells_b, max_n - len(cells_a), replace=False)
    sub = A[np.concatenate([cells_a, cells_b])].copy()
    sub.obs["grp"] = pd.Categorical(["A"] * len(cells_a) + ["B"] * len(cells_b))
    detn = np.asarray((sub.X > 0).sum(0)).ravel()
    sub = sub[:, detn >= max(10, 0.01 * sub.n_obs)].copy()
    sc.tl.rank_genes_groups(sub, "grp", groups=["A"], reference="B", method="wilcoxon", use_raw=False, pts=True)
    df = sc.get.rank_genes_groups_df(sub, "A")
    df = df.rename(columns={"names": "gene", "logfoldchanges": "lfc", "pvals_adj": "padj"})
    universe = set(sub.var_names)
    return df, universe, len(cells_a), len(cells_b)


def enrich(genes_up, universe, gs, min_size=3):
    rows = []
    M = len(universe)
    N = len(genes_up)
    up = set(genes_up)
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
        E["q"] = bh(E.p)
        E = E.sort_values("p")
    return E


def de_up_list(df, lfc=0.5, padj=0.05, n=150):
    d = df[(df.lfc > lfc) & (df.padj < padj)].sort_values("scores", ascending=False)
    return d.gene.head(n).tolist()


# ----------------------------------------------------------------------------------------------
# 5. beyond ISG: residual novelty and ISG-independent unmatched cells
# ----------------------------------------------------------------------------------------------
@section("5_beyond_ISG")
def beyond_isg(D, DN, A, gs, isg_col, diss_col, ref_q75):
    out = {}
    D["nov_resid"] = np.nan
    D["IIU_resid"] = False
    covs = [isg_col, diss_col, "log_genes", "log_counts", "pct_mito"]
    fits = []
    for t, idx in D.groupby("y").groups.items():
        g = D.loc[idx]
        if len(g) < 200:
            continue
        Xm = np.column_stack([np.ones(len(g))] + [zs(g[c].astype(float).values) for c in covs])
        yv = g.log_novelty.values
        beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
        res = yv - Xm @ beta
        r2 = 1 - res.var() / yv.var()
        # partial R2 of ISG (drop-one)
        Xd = np.delete(Xm, 1, axis=1)
        b2, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
        r2_noisg = 1 - (yv - Xd @ b2).var() / yv.var()
        D.loc[idx, "nov_resid"] = res
        D.loc[idx, "IIU_resid"] = res > np.quantile(res, 0.9)
        fits.append(dict(type=t, n=len(g), r2_full=r2, r2_isg_partial=r2 - r2_noisg,
                         **{f"beta_{c}": b for c, b in zip(covs, beta[1:])}))
    F = pd.DataFrame(fits)
    wcsv(F, "novelty_residualization_by_type.csv")
    out["residualization"] = dict(median_r2_full=float(F.r2_full.median()),
                                  median_r2_isg_partial=float(F.r2_isg_partial.median()))
    D["isg_low"] = D[isg_col].values < D.y.map(ref_q75).astype(float).fillna(np.inf).values
    D["IIU_flag_isglow"] = D.flag & D.isg_low
    D["IIU"] = D.IIU_resid | D.IIU_flag_isglow
    rates = D.groupby("cond")[["flag", "IIU_resid", "IIU_flag_isglow", "IIU"]].mean()
    out["cell_rates"] = rates.to_dict()
    sle_fl = D[(D.is_sle == 1) & D.flag]
    out["frac_SLE_flagged_isg_low"] = float(sle_fl.isg_low.mean()) if len(sle_fl) else None
    hq_fl = D[(D.is_sle == 0) & D.flag]
    out["frac_healthy_flagged_isg_low"] = float(hq_fl.isg_low.mean()) if len(hq_fl) else None
    bytype = D.groupby(["y", "cond"]).agg(n=("flag", "size"), flag=("flag", "mean"), IIU=("IIU", "mean"),
                                          IIU_resid=("IIU_resid", "mean"),
                                          IIU_flag_isglow=("IIU_flag_isglow", "mean")).unstack("cond")
    bytype.columns = [f"{a}_{b}" for a, b in bytype.columns]
    wcsv(bytype.reset_index(), "IIU_rates_by_type.csv")
    # donor-level IIU fraction
    dd = D.groupby("donor").agg(iiu=("IIU", "mean"), iiu_resid=("IIU_resid", "mean"),
                                iiu_flag_isglow=("IIU_flag_isglow", "mean"))
    for c in dd:
        DN[c] = dd[c].reindex(DN.index)
    DN["iiu_logit"] = logit_frac(DN.iiu * DN.n_cells, DN.n_cells)
    out["donor_IIU_SLE_vs_healthy"] = {c: mwu(DN[DN.is_sle == 1][c], DN[DN.is_sle == 0][c])
                                       for c in ["iiu", "iiu_resid", "iiu_flag_isglow"]}
    # associations among SLE donors
    S = DN[DN.is_sle == 1]
    assoc = {"spearman_iiu_vs_isg_SLE": spearman_boot(S.iiu, S.isg),
             "spearman_iiu_vs_isg_healthy": spearman_boot(DN[DN.is_sle == 0].iiu, DN[DN.is_sle == 0].isg)}
    for k in ["status", "ancestry", "cohort", "sex"] + [c for c in DN.columns if c.startswith("clin_")]:
        if k not in S:
            continue
        v = S[k]
        num = pd.to_numeric(v, errors="coerce")
        if num.notna().mean() > 0.7 and num.nunique() > 5:
            assoc[f"spearman_iiu_vs_{k}"] = spearman_boot(S.iiu, num, nboot=500)
            continue
        groups = {str(l): g.iiu.values for l, g in S.groupby(v.astype(str)) if len(g) >= 3}
        if len(groups) >= 2:
            assoc[f"kruskal_iiu_by_{k}"] = dict(p=float(stats.kruskal(*groups.values()).pvalue),
                                                medians={l: float(np.median(x)) for l, x in groups.items()},
                                                n={l: int(len(x)) for l, x in groups.items()})
    if "age_num" in S:
        assoc["spearman_iiu_vs_age_SLE"] = spearman_boot(S.iiu, S.age_num, nboot=500)
    try:
        import statsmodels.formula.api as smf
        X, terms = covariate_frame(S)
        Fm = pd.concat([S[["iiu_logit", "isg"]], X], axis=1)
        Fm["isg_z"] = zs(Fm.isg)
        st = []
        if "status" in S and collapse_levels(S.status, 3).nunique() > 1:
            Fm["status"] = collapse_levels(S.status, 3)
            st = ["C(status)"]
        fml = "iiu_logit ~ isg_z" + "".join(" + " + t for t in terms + st)
        fit = smf.ols(fml, data=Fm).fit(cov_type="HC3")
        assoc["ols_SLE"] = dict(formula=fml, n=int(fit.nobs), r2=float(fit.rsquared), terms=coef_table(fit))
    except Exception as e:
        assoc["ols_SLE"] = dict(error=repr(e))
    out["donor_IIU_associations"] = assoc
    # DE per fine type: SLE IIU vs SLE non-flagged/non-IIU; same in healthy as artifact control
    de_out = {}
    if not args.skip_de:
        cnt = D[(D.is_sle == 1) & D.IIU].y.value_counts()
        types = cnt[cnt >= 50].index[:args.max_sub_types]
        allE = []
        for t in types:
            r = {}
            for cn, sle in [("SLE", 1), ("healthy", 0)]:
                g = D[(D.y == t) & (D.is_sle == sle)]
                a_ = g.index[g.IIU].values
                b_ = g.index[(~g.IIU) & (~g.flag)].values
                if len(a_) < 30 or len(b_) < 30:
                    continue
                df, uni, na, nb = run_de(A, a_, b_)
                up = de_up_list(df)
                df.head(200).to_csv(os.path.join(OUT, f"de_IIU_{safe_name(t)}_{cn}.csv"), index=False)
                E = enrich(up, uni, gs)
                if len(E):
                    E.insert(0, "type", t)
                    E.insert(1, "cond", cn)
                    allE.append(E)
                r[cn] = dict(n_IIU=na, n_ref=nb, top_up=df.sort_values("scores", ascending=False).head(25)[
                    ["gene", "lfc", "padj"]].round(4).values.tolist(), n_up=len(up),
                    top_sets=E.head(6)[["set", "overlap", "q"]].values.tolist() if len(E) else [])
            if "SLE" in r and "healthy" in r:
                a = set(de_up_list(pd.read_csv(os.path.join(OUT, f"de_IIU_{safe_name(t)}_SLE.csv")), n=100))
                b = set(de_up_list(pd.read_csv(os.path.join(OUT, f"de_IIU_{safe_name(t)}_healthy.csv")), n=100))
                r["jaccard_up_SLE_vs_healthy"] = len(a & b) / max(len(a | b), 1)
                r["SLE_specific_up"] = sorted(a - b)[:40]
            de_out[t] = r
            log(f"  DE IIU {t}: SLE top {[x[0] for x in r.get('SLE', {}).get('top_up', [])[:10]]}")
        if allE:
            wcsv(pd.concat(allE), "enrichment_IIU_by_type.csv")
    out["DE_IIU"] = de_out
    R["S5"] = out
    # figure: IIU rates per type
    try:
        bt = bytype.copy()
        if "IIU_SLE" in bt and "IIU_healthy" in bt:
            bt = bt.sort_values("IIU_SLE", ascending=False).head(25)
            fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(bt)), 6))
            x = np.arange(len(bt))
            ax.bar(x - 0.2, 100 * bt.IIU_healthy, 0.4, color=COL["healthy"], label="healthy (query)")
            ax.bar(x + 0.2, 100 * bt.IIU_SLE, 0.4, color=COL["SLE"], label="SLE")
            ax.set_xticks(x)
            ax.set_xticklabels(bt.index, rotation=60, ha="right")
            ax.set_ylabel("ISG-independent unmatched (%)")
            ax.legend(frameon=False)
            savefig(fig, "fig_IIU_by_type.png")
    except Exception as e:
        log("IIU figure failed", e)
    return out


# ----------------------------------------------------------------------------------------------
# 6. sub-clustering of flagged cells in the latent spaces
# ----------------------------------------------------------------------------------------------
def leiden(Z, res, key, seed=0):
    import scanpy as sc
    import anndata as ad
    a = ad.AnnData(np.zeros((Z.shape[0], 1), dtype=np.float32))
    a.obsm["Z"] = Z.astype(np.float32)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="Z", random_state=seed)
    try:
        sc.tl.leiden(a, resolution=res, key_added=key, flavor="igraph", n_iterations=2, directed=False,
                     random_state=seed)
    except (TypeError, ValueError, ImportError):
        sc.tl.leiden(a, resolution=res, key_added=key, random_state=seed)
    return a.obs[key].astype(str).values


@section("6_subclustering_flagged")
def subcluster(D, A, gs, isg_col):
    from sklearn.metrics import adjusted_rand_score
    path = os.path.join(ROOT, "emb", f"{args.tag}_seed{args.seed0}_full.npz")
    E = np.load(path, allow_pickle=True)
    tn = pd.Index(np.asarray(E["tgt_names"]).astype(str))
    pos = tn.get_indexer(D.index)
    if (pos < 0).any():
        raise KeyError(f"{(pos < 0).sum()} result cells not in embedding tgt_names")
    U, S = np.asarray(E["tgt_U"], np.float32), np.asarray(E["tgt_S"], np.float32)
    # standardize with reference statistics so blocks have equal weight
    def stdz(Zq, Zr):
        mu, sd = Zr.mean(0), Zr.std(0) + 1e-6
        return (Zq - mu) / sd / np.sqrt(Zq.shape[1])
    Uz, Sz = stdz(U, np.asarray(E["ref_U"])), stdz(S, np.asarray(E["ref_S"]))
    del E
    score_cols = [c for c in D.columns if c.startswith("gs_")]
    fl_counts = D[D.flag].y.value_counts()
    types = fl_counts[fl_counts >= args.min_flag_sub].index[:args.max_sub_types]
    rows, markers = [], {}
    rng = np.random.default_rng(5)
    for t in types:
        g = D[(D.y == t) & D.flag]
        if len(g) > args.max_sub_cells:
            g = g.iloc[rng.choice(len(g), args.max_sub_cells, replace=False)]
        p = tn.get_indexer(g.index)
        Z = np.hstack([Uz[p], Sz[p]])
        cl = leiden(Z, args.leiden_res, "cl")
        try:
            clU = leiden(Uz[p] * np.sqrt(Uz.shape[1]), args.leiden_res, "cU")
            clS = leiden(Sz[p] * np.sqrt(Sz.shape[1]), args.leiden_res, "cS")
            ari = dict(ari_US_vs_U=adjusted_rand_score(cl, clU), ari_US_vs_S=adjusted_rand_score(cl, clS),
                       ari_U_vs_S=adjusted_rand_score(clU, clS))
        except Exception as e:
            ari = dict(error=repr(e))
        g = g.assign(sub=[f"{t}|c{c}" for c in cl])
        D.loc[g.index, "subcluster"] = g["sub"].values
        allq = D[D.y == t]
        kept = allq[~allq.flag]
        donors_all = allq.donor.unique()
        dsle = allq.groupby("donor").is_sle.first().reindex(donors_all).values
        n_type_d = allq.groupby("donor").size().reindex(donors_all).values
        frac_sle_type = allq.is_sle.mean()
        frac_sle_flag = g.is_sle.mean()
        # count matrix cluster x donor for permutation
        cm = pd.crosstab(g["sub"], g.donor).reindex(columns=donors_all, fill_value=0)
        nk = cm.sum(1).values
        tot_fl = cm.values.sum(0)
        Lp = np.array([rng.permutation(dsle) for _ in range(args.nperm)])  # nperm x nd
        perm_frac = (Lp @ cm.values.T) / nk  # nperm x nclust
        perm_flagfrac = (Lp @ tot_fl) / tot_fl.sum()
        obs_frac = (cm.values @ dsle) / nk
        kept_mu = kept[score_cols].mean() if len(kept) else None
        kept_sd = kept[score_cols].std() if len(kept) else None
        for j, (cname, gc) in enumerate(g.groupby("sub")):
            j = list(cm.index).index(cname)
            per_d = gc.donor.value_counts()
            n_don = int((per_d >= args.min_cells_per_donor_state).sum())
            a_sle = int(gc.is_sle.sum())
            b_sle = int(g.is_sle.sum() - a_sle)
            odds, pf = stats.fisher_exact([[a_sle, len(gc) - a_sle], [b_sle, len(g) - len(gc) - b_sle]])
            p_perm_vs_type = (1 + (perm_frac[:, j] >= obs_frac[j]).sum()) / (1 + args.nperm)
            rel = obs_frac[j] - frac_sle_flag
            p_perm_rel = (1 + ((perm_frac[:, j] - perm_flagfrac) >= rel).sum()) / (1 + args.nperm)
            # donor-level abundance: fraction of donor's cells of type t in this subcluster
            dab = (cm.loc[cname].values / np.maximum(n_type_d, 1))
            ok = n_type_d >= 20
            dm = mwu(dab[ok & (dsle == 1)], dab[ok & (dsle == 0)])
            row = dict(type=t, subcluster=cname, n_cells=len(gc), n_donors_any=int(per_d.size),
                       n_donors_ge_min=n_don, top_donor_share=float(per_d.iloc[0] / len(gc)),
                       frac_sle=float(gc.is_sle.mean()), exp_frac_sle_type=float(frac_sle_type),
                       frac_sle_all_flagged_type=float(frac_sle_flag), fisher_or_vs_other_flagged=float(odds),
                       fisher_p=float(pf), perm_p_sle_vs_type=float(p_perm_vs_type),
                       perm_p_sle_vs_other_flagged=float(p_perm_rel), donor_abund_mwu_p=dm.get("p"),
                       donor_abund_auc_sle=dm.get("auc_a_gt_b"),
                       mean_isg=float(gc[isg_col].mean()), frac_isg_low=float(gc.isg_low.mean()) if "isg_low" in gc else None,
                       mean_novelty=float(gc.novelty_mean.mean()),
                       frac_IIU=float(gc.IIU.mean()) if "IIU" in gc else None,
                       state_called=bool(n_don >= args.min_donors_state and len(gc) >= 50), **ari)
            if "closest_label" in gc:
                vc = gc.closest_label.value_counts(normalize=True)
                row["closest_label"] = vc.index[0]
                row["closest_label_frac"] = float(vc.iloc[0])
                row["median_conf_pmax"] = float(gc.conf_pmax.median())
            if "pred_ARCANE" in gc:
                vc = gc.pred_ARCANE.value_counts(normalize=True)
                row["arcane_pred_top"] = f"{vc.index[0]}:{vc.iloc[0]:.2f}"
            if kept_mu is not None:
                for c in score_cols:
                    row["z_" + c[3:]] = float((gc[c].mean() - kept_mu[c]) / (kept_sd[c] + 1e-9))
            rows.append(row)
        # markers: each subcluster vs kept cells of the same type, and one-vs-rest among flagged
        if not args.skip_de:
            try:
                import scanpy as sc
                ksel = kept.index.values
                if len(ksel) > 10000:
                    ksel = rng.choice(ksel, 10000, replace=False)
                gsel = g.index.values
                lab = pd.Series(g["sub"].values, index=g.index)
                if len(gsel) > args.max_de:
                    gsel = rng.choice(gsel, args.max_de, replace=False)
                sub = A[np.concatenate([gsel, ksel])].copy()
                sub.obs["grp"] = pd.Categorical(list(lab.loc[gsel].values) + ["kept"] * len(ksel))
                vc = sub.obs.grp.value_counts()
                small = vc[vc < 10].index
                sub = sub[~sub.obs.grp.isin(small)].copy()
                sub.obs["grp"] = sub.obs.grp.cat.remove_unused_categories()
                detn = np.asarray((sub.X > 0).sum(0)).ravel()
                sub = sub[:, detn >= max(10, 0.01 * sub.n_obs)].copy()
                sc.tl.rank_genes_groups(sub, "grp", reference="kept", method="wilcoxon", use_raw=False)
                for cname in [c for c in sub.obs.grp.cat.categories if c != "kept"]:
                    df = sc.get.rank_genes_groups_df(sub, cname).rename(
                        columns={"names": "gene", "logfoldchanges": "lfc", "pvals_adj": "padj"})
                    up = de_up_list(df)
                    En = enrich(up, set(sub.var_names), gs)
                    markers[cname] = dict(vs_kept_top=df.head(20)[["gene", "lfc", "padj"]].round(4).values.tolist(),
                                          enrich=En.head(5)[["set", "overlap", "q"]].values.tolist() if len(En) else [])
                fsub = sub[sub.obs.grp != "kept"].copy()
                fsub.obs["grp"] = fsub.obs.grp.cat.remove_unused_categories()
                if fsub.obs.grp.nunique() >= 2:
                    sc.tl.rank_genes_groups(fsub, "grp", method="wilcoxon", use_raw=False)
                    for cname in fsub.obs.grp.cat.categories:
                        df = sc.get.rank_genes_groups_df(fsub, cname)
                        markers.setdefault(cname, {})["vs_other_flagged_top"] = df.head(15)[
                            ["names", "logfoldchanges", "pvals_adj"]].round(4).values.tolist()
            except Exception as e:
                log(f"  markers failed for {t}: {e!r}")
        log(f"  subclustered {t}: {len(g)} flagged cells, {len(set(cl))} clusters, ARI {ari}")
    T = pd.DataFrame(rows)
    if len(T):
        T["fisher_q"] = bh(T.fisher_p)
        T["perm_q_vs_type"] = bh(T.perm_p_sle_vs_type)
        T["perm_q_vs_other_flagged"] = bh(T.perm_p_sle_vs_other_flagged)
        T["donor_abund_q"] = bh(T.donor_abund_mwu_p.astype(float))
        T["top_markers"] = T.subcluster.map(lambda c: ",".join(x[0] for x in markers.get(c, {}).get("vs_kept_top", [])[:8]))
        T["top_sets"] = T.subcluster.map(lambda c: ";".join(f"{x[0]}(q={x[2]:.1g})" for x in markers.get(c, {}).get("enrich", [])[:3]))
    wcsv(T, "subclusters_flagged.csv")
    json.dump(markers, open(os.path.join(OUT, "subcluster_markers.json"), "w"), indent=1, default=jsonable)
    called = T[T.state_called] if len(T) else T
    out = dict(n_types=int(len(types)), n_subclusters=int(len(T)), n_states_called=int(len(called)),
               states_sle_enriched_q05=int(((called.perm_q_vs_type < 0.05) & (called.frac_sle > called.exp_frac_sle_type)).sum())
               if len(called) else 0,
               states=called.drop(columns=[c for c in called.columns if c.startswith("z_")]).to_dict(orient="records")
               if len(called) else [])
    R["S6"] = out
    # heatmap of gene-set z-scores of called states
    try:
        zc = [c for c in T.columns if c.startswith("z_")]
        H = called.set_index("subcluster")[zc]
        if len(H):
            H.columns = [c[2:] for c in zc]
            H = H.head(40)
            fig, ax = plt.subplots(figsize=(0.55 * H.shape[1] + 5, 0.45 * H.shape[0] + 3))
            v = np.nanmax(np.abs(H.values)) if np.isfinite(H.values).any() else 1
            im = ax.imshow(H.values, cmap="RdBu_r", vmin=-min(v, 3), vmax=min(v, 3), aspect="auto")
            ax.set_xticks(range(H.shape[1]))
            ax.set_xticklabels(H.columns, rotation=70, ha="right", fontsize=13)
            ax.set_yticks(range(H.shape[0]))
            ax.set_yticklabels([f"{i} (n={n}, SLE {100 * f:.0f}%)" for i, n, f in
                                zip(H.index, called.n_cells.head(40), called.frac_sle.head(40))], fontsize=12)
            fig.colorbar(im, ax=ax, label="z vs kept cells")
            savefig(fig, "fig_subcluster_genesets.png")
    except Exception as e:
        log("subcluster heatmap failed", e)
    return out


# ----------------------------------------------------------------------------------------------
# 7. dissociation control
# ----------------------------------------------------------------------------------------------
@section("7_dissociation_control")
def dissociation(D, diss_col):
    out = {}
    core = "gs_DISS_core" if "gs_DISS_core" in D else diss_col
    # (a) flagged vs kept dissociation score per condition
    a = {}
    for cn, g in D.groupby("cond"):
        f, k = g.loc[g.flag, diss_col].values, g.loc[~g.flag, diss_col].values
        if len(f) > 100000:
            f = RNG.choice(f, 100000, replace=False)
        if len(k) > 100000:
            k = RNG.choice(k, 100000, replace=False)
        r = mwu(f, k)
        r["cohen_d"] = cohen_d(f, k)
        a[cn] = r
    out["diss_flagged_vs_kept"] = a
    pt = []
    for (t, cn), g in D.groupby(["y", "cond"]):
        if g.flag.sum() >= 20 and (~g.flag).sum() >= 20:
            pt.append(dict(type=t, cond=cn, n_flag=int(g.flag.sum()),
                           d=cohen_d(g.loc[g.flag, diss_col], g.loc[~g.flag, diss_col]),
                           mean_flag=float(g.loc[g.flag, diss_col].mean()),
                           mean_kept=float(g.loc[~g.flag, diss_col].mean())))
    pt = pd.DataFrame(pt)
    wcsv(pt, "diss_flagged_vs_kept_by_type.csv")
    # (b) IEG target genes, flagged vs kept, matched on core dissociation score within donor x type x decile
    D = D.copy()
    D["core_bin"] = D.groupby("y")[core].transform(
        lambda v: pd.qcut(v.rank(method="first"), 10, labels=False) if len(v) >= 20 else 0)
    genes = [c for c in D.columns if c.startswith("g_")] + [diss_col]
    sets = {"SLE_flagged": (D.is_sle == 1), "healthy_flagged": (D.is_sle == 0)}
    if "IIU" in D:
        sets["SLE_IIU"] = (D.is_sle == 1)
    matched = {}
    for nm, msk in sets.items():
        g = D[msk]
        treat = g.IIU if nm == "SLE_IIU" else g.flag
        ctrl = (~g.flag) & (~g.IIU) if ("IIU" in g and nm == "SLE_IIU") else ~g.flag
        g = g[treat | ctrl].assign(treat=treat[treat | ctrl].values)
        st = g.groupby(["donor", "y", "core_bin", "treat"])
        mean = st[genes].mean()
        cnt = st.size()
        idx = mean.index.droplevel("treat").unique()
        try:
            m1 = mean.xs(True, level="treat").reindex(idx)
            m0 = mean.xs(False, level="treat").reindex(idx)
            n1 = cnt.xs(True, level="treat").reindex(idx)
            n0 = cnt.xs(False, level="treat").reindex(idx)
        except KeyError:
            continue
        ok = n1.notna() & n0.notna()
        m1, m0, n1, n0 = m1[ok], m0[ok], n1[ok], n0[ok]
        w = 1.0 / (1.0 / n1 + 1.0 / n0)
        diff = (m1 - m0).mul(w, axis=0)
        donors = diff.index.get_level_values("donor")
        Sd = diff.groupby(donors).sum()
        Wd = w.groupby(donors).sum()
        est = Sd.sum() / Wd.sum()
        rngb = np.random.default_rng(7)
        dl = Sd.index.values
        bs = []
        for _ in range(500):
            s = rngb.choice(len(dl), len(dl), replace=True)
            bs.append(Sd.iloc[s].sum() / Wd.iloc[s].sum())
        bs = pd.DataFrame(bs)
        raw = g[g.treat][genes].mean() - g[~g.treat][genes].mean()
        matched[nm] = {gn: dict(raw_diff=float(raw[gn]), matched_diff=float(est[gn]),
                                ci_lo=float(bs[gn].quantile(0.025)), ci_hi=float(bs[gn].quantile(0.975)))
                       for gn in genes}
        matched[nm]["_n"] = dict(n_strata=int(ok.sum()), n_donors=int(len(dl)), n_treat=int(n1.sum()))
    out["IEG_matched_on_core_dissociation"] = matched
    rows = []
    for nm, d in matched.items():
        for gn, v in d.items():
            if gn != "_n":
                rows.append(dict(contrast=nm, feature=gn, **v))
    MT = pd.DataFrame(rows)
    wcsv(MT, "IEG_matched_contrasts.csv")
    # (c) dissociation-standardized SLE vs healthy flag rates per type
    std_rows = []
    D["diss_bin"] = D.groupby("y")[diss_col].transform(
        lambda v: pd.qcut(v.rank(method="first"), 10, labels=False) if len(v) >= 20 else 0)
    for t, g in D.groupby("y"):
        h, s = g[g.is_sle == 0], g[g.is_sle == 1]
        if len(h) < 100 or len(s) < 100:
            continue
        wh = h.diss_bin.value_counts(normalize=True)
        rs = s.groupby("diss_bin").flag.mean()
        rh = h.groupby("diss_bin").flag.mean()
        std_s = float((wh * rs.reindex(wh.index)).sum() / wh[rs.reindex(wh.index).notna()].sum())
        std_rows.append(dict(type=t, flag_healthy=float(h.flag.mean()), flag_sle_raw=float(s.flag.mean()),
                             flag_sle_diss_standardized=std_s,
                             frac_flag_in_top_diss_decile_healthy=float((h[h.flag].diss_bin == 9).mean()) if h.flag.any() else None,
                             frac_flag_in_top_diss_decile_sle=float((s[s.flag].diss_bin == 9).mean()) if s.flag.any() else None))
    STD = pd.DataFrame(std_rows)
    wcsv(STD, "flag_rate_diss_standardized.csv")
    if len(STD):
        out["diss_standardization"] = dict(median_ratio_raw=float((STD.flag_sle_raw / STD.flag_healthy.clip(1e-4)).median()),
                                           median_ratio_std=float((STD.flag_sle_diss_standardized / STD.flag_healthy.clip(1e-4)).median()))
    R["S7"] = out
    # figure: matched IEG differences
    try:
        if len(MT):
            M = MT[MT.feature != diss_col]
            feats = M.feature.unique()
            fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(feats)), 5.5))
            cs = {"healthy_flagged": COL["healthy"], "SLE_flagged": COL["SLE"], "SLE_IIU": "#7A3B8F"}
            con = list(M.contrast.unique())
            wbar = 0.8 / len(con)
            for i, c in enumerate(con):
                mm = M[M.contrast == c].set_index("feature").reindex(feats)
                x = np.arange(len(feats)) + (i - (len(con) - 1) / 2) * wbar
                ax.bar(x, mm.matched_diff, wbar, color=cs.get(c, "grey"), label=c.replace("_", " "))
                ax.errorbar(x, mm.matched_diff, yerr=[mm.matched_diff - mm.ci_lo, mm.ci_hi - mm.matched_diff],
                            fmt="none", ecolor="k", lw=1)
            ax.axhline(0, c="k", lw=0.8)
            ax.set_xticks(range(len(feats)))
            ax.set_xticklabels([f[2:] for f in feats], rotation=45, ha="right")
            ax.set_ylabel("Flagged - kept (log expr.),\nmatched on core dissociation")
            ax.legend(frameon=False)
            savefig(fig, "fig_IEG_matched.png")
    except Exception as e:
        log("IEG figure failed", e)
    return out


# ----------------------------------------------------------------------------------------------
# 8. ARCANE vs ISG score alone
# ----------------------------------------------------------------------------------------------
@section("8_ARCANE_vs_ISG")
def arcane_vs_isg(D, DN, isg_col):
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    out = {}
    H = DN[DN.is_sle == 0]
    thr_f, thr_i = H.flag_frac.quantile(0.9), H.isg.quantile(0.9)
    S = DN[DN.is_sle == 1]
    hi_flag_low_isg = S[(S.flag_frac > thr_f) & (S.isg <= thr_i)]
    hi_isg_low_flag = S[(S.flag_frac <= thr_f) & (S.isg > thr_i)]
    out["donor_quadrants_SLE"] = dict(
        thr_flag_healthy_q90=float(thr_f), thr_isg_healthy_q90=float(thr_i), n_sle=int(len(S)),
        n_high_flag_low_isg=int(len(hi_flag_low_isg)), n_high_isg_low_flag=int(len(hi_isg_low_flag)),
        n_both_high=int(((S.flag_frac > thr_f) & (S.isg > thr_i)).sum()),
        n_both_low=int(((S.flag_frac <= thr_f) & (S.isg <= thr_i)).sum()))
    keep_cols = [c for c in ["cond", "n_cells", "flag_frac", "isg", "isg_raw", "diss", "iiu", "cohort", "ancestry",
                             "sex", "age_num", "status"] if c in DN]
    wcsv(hi_flag_low_isg[keep_cols].reset_index(), "donors_high_flag_low_isg.csv")
    # donor-level ROC
    y = DN.is_sle.values
    feats = {"ISG (type-adj)": DN.isg.values, "ISG (raw)": DN.isg_raw.values, "flag fraction": DN.flag_frac.values,
             "mean novelty": DN.novelty_mean.values}
    if "iiu" in DN:
        feats["IIU fraction"] = DN.iiu.values
    out["donor_auc"] = {k: float(roc_auc_score(y, v)) for k, v in feats.items() if np.isfinite(v).all()}
    # cross-validated combinations
    combos = {"ISG": ["isg"], "flag": ["flag_logit"], "ISG+flag": ["isg", "flag_logit"]}
    if "iiu" in DN:
        combos["ISG+IIU"] = ["isg", "iiu_logit"]
        combos["ISG+flag+IIU"] = ["isg", "flag_logit", "iiu_logit"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    cvres = {}
    oof_store = {}
    for nm, cols in combos.items():
        X = np.column_stack([zs(DN[c].values) for c in cols])
        aucs = []
        oof = np.zeros(len(y))
        for i, (tr, te) in enumerate(cv.split(X, y)):
            m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
            pr = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], pr))
            if i < 5:
                oof[te] = pr
        cvres[nm] = dict(cv_auc_mean=float(np.mean(aucs)), cv_auc_sd=float(np.std(aucs)))
        oof_store[nm] = oof
    out["donor_cv_auc"] = cvres
    R["S8"] = out
    # likelihood-ratio tests
    lrt = {}
    try:
        import statsmodels.api as sm
    except ImportError as e:
        lrt["error"] = repr(e)
    Xc, terms = covariate_frame(DN)
    for nm, (small, big) in {"flag beyond ISG": (["isg"], ["isg", "flag_logit"]),
                             "IIU beyond ISG": (["isg"], ["isg", "iiu_logit"]),
                             "flag beyond ISG+diss": (["isg", "diss"], ["isg", "diss", "flag_logit"])}.items():
        if any(c not in DN for c in big) or "error" in lrt:
            continue
        for adj in [False, True]:
            try:
                def design(cols):
                    X = pd.DataFrame({c: zs(DN[c].values) for c in cols}, index=DN.index)
                    if adj:
                        for c in ["ancestry", "sex"]:
                            if c in Xc:
                                X = X.join(pd.get_dummies(Xc[c], prefix=c, drop_first=True).astype(float))
                        if "age_z" in Xc:
                            X["age_z"] = Xc["age_z"]
                    return sm.add_constant(X)
                m0 = sm.Logit(y, design(small)).fit(disp=0, maxiter=200)
                m1 = sm.Logit(y, design(big)).fit(disp=0, maxiter=200)
                lr = 2 * (m1.llf - m0.llf)
                df = len(m1.params) - len(m0.params)
                last = big[-1]
                lrt[nm + (" (adj ancestry/sex/age)" if adj else "")] = dict(
                    lr=float(lr), df=int(df), p=float(stats.chi2.sf(lr, df)),
                    coef=float(m1.params[last]), coef_p=float(m1.pvalues[last]),
                    aic0=float(m0.aic), aic1=float(m1.aic), n=int(len(y)))
            except Exception as e:
                lrt[nm + (" adj" if adj else "")] = dict(error=repr(e))
    out["donor_LRT"] = lrt
    # cell level: within each fine type, AUROC (SLE vs healthy cells) of ISG vs novelty, and flagged-but-ISG-low
    rows = []
    for t, g in D.groupby("y"):
        if g.is_sle.sum() >= 50 and (g.is_sle == 0).sum() >= 50:
            rows.append(dict(type=t, auc_isg=roc_auc_score(g.is_sle, g[isg_col]),
                             auc_novelty=roc_auc_score(g.is_sle, g.novelty_mean),
                             auc_resid_novelty=roc_auc_score(g.is_sle, g.nov_resid.fillna(0)) if "nov_resid" in g else np.nan,
                             frac_sle_flagged_isg_low=float(g[(g.is_sle == 1) & g.flag].isg_low.mean())
                             if "isg_low" in g and ((g.is_sle == 1) & g.flag).any() else np.nan))
    CT = pd.DataFrame(rows)
    wcsv(CT, "cell_auc_by_type.csv")
    out["cell_auc_by_type_median"] = CT.drop(columns="type").median().to_dict() if len(CT) else {}
    R["S8"] = out
    # figure: donor ROC
    try:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for k, v in feats.items():
            fpr, tpr, _ = roc_curve(y, v)
            ax.plot(fpr, tpr, lw=2.5, label=f"{k} ({out['donor_auc'][k]:.2f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(frameon=False, fontsize=13, loc="lower right")
        savefig(fig, "fig_donor_roc.png")
    except Exception as e:
        log("ROC figure failed", e)
    return out


# ----------------------------------------------------------------------------------------------
# 9. composition
# ----------------------------------------------------------------------------------------------
@section("9_composition")
def composition(D, DN):
    out = {}
    lab = "pred_ARCANE" if "pred_ARCANE" in D else "y"
    conf = D[~D.flag]
    C1 = pd.crosstab(conf.donor, conf[lab], normalize="index")
    C2 = pd.crosstab(D[D.flag].donor, D[D.flag].y, normalize="index")
    C3 = pd.crosstab(D.donor, D.y, normalize="index")
    FR = D.groupby(["donor", "y"]).flag.mean().unstack()
    for nm, C in [("confident_ARCANE", C1), ("flagged_author_label", C2), ("all_author_label", C3),
                  ("flag_rate_by_type", FR)]:
        C.to_csv(os.path.join(OUT, f"composition_{nm}.csv"))
    rows = []
    cond = DN.is_sle
    for nm, C in [("confident_ARCANE", C1), ("all_author_label", C3), ("flag_rate_by_type", FR)]:
        for t in C.columns:
            v = C[t].reindex(DN.index)
            r = mwu(v[cond == 1], v[cond == 0])
            rows.append(dict(table=nm, type=t, mean_sle=r.get("mean_a"), mean_healthy=r.get("mean_b"),
                             log2fc=np.log2((r.get("mean_a") or 0) + 1e-4) - np.log2((r.get("mean_b") or 0) + 1e-4)
                             if r.get("p") is not None else np.nan, p=r.get("p")))
    T = pd.DataFrame(rows)
    T["q"] = T.groupby("table").p.transform(lambda p: pd.Series(bh(p.values), index=p.index))
    wcsv(T, "composition_tests.csv")
    # agreement between ARCANE-confident and author compositions (per donor, per type)
    common = C1.columns.intersection(C3.columns)
    if len(common):
        a = C1[common].reindex(DN.index).values.ravel()
        b = C3[common].reindex(DN.index).values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        out["confident_vs_author_composition_pearson"] = float(np.corrcoef(a[m], b[m])[0, 1])
    out["sig_q05"] = T[T.q < 0.05].sort_values("p").head(40).to_dict(orient="records")
    R["S9"] = out
    try:
        P = T[T.table.isin(["confident_ARCANE", "all_author_label"])].pivot(index="type", columns="table", values="log2fc")
        P = P.dropna(how="all").sort_values(P.columns[0])
        fig, ax = plt.subplots(figsize=(7, max(5, 0.38 * len(P) + 1.5)))
        y = np.arange(len(P))
        for i, c in enumerate(P.columns):
            ax.barh(y + (i - 0.5) * 0.4, P[c], 0.4, label=c.replace("_", " "),
                    color=["#4A7FB0", "#9AA5B1"][i % 2])
        ax.axvline(0, c="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(P.index, fontsize=13)
        ax.set_xlabel("log2 (SLE / healthy) donor fraction")
        ax.legend(frameon=False)
        savefig(fig, "fig_composition.png")
    except Exception as e:
        log("composition figure failed", e)
    return out


# ==============================================================================================
# CONTROL ANALYSES (sections 10-16): composition, cohort, doublets, ISG leakage, pseudobulk, view disagreement, state gates.
# They address: composition, seed stability, cohort/ancestry/pool confounding, doublets/ambient RNA/QC,
# residual ISG confounding of "ISG-independent" novelty, donor pseudoreplication in DE, the U-vs-S
# view disagreement, and acceptance gates for every candidate "novel state".
# ==============================================================================================
LINEAGE_MARKERS = {"T": ["CD3E", "CD3D", "CD3G", "TRAC", "CD2"],
                   "B": ["CD79A", "CD79B", "MS4A1", "CD19", "BANK1"],
                   "M": ["LYZ", "CST3", "S100A8", "S100A9", "FCN1", "CD14"],
                   "PLT": ["PPBP", "PF4", "GP9", "NRGN"],
                   "HB": ["HBB", "HBA1", "HBA2"],
                   "IG": ["IGKC", "IGHG1", "IGLC2", "IGHA1", "JCHAIN", "IGHM"]}
# gene-set names (genesets.json) describing KNOWN activated / disease-associated subsets; a state matching
# one of them (z >= 1.5 vs kept cells) is reported as 'known state', not as a novel finding
KNOWN_STATE_PAT = (r"MS1|SIGLEC1|DC3|DN2|ABC|atypical|activated_naive|plasmablast|Tph|Tfh|cytotoxic|GZMK|"
                   r"exhaust|Treg_activ|adaptive_NK|LDG|neutrophil|platelet|cell_cycle|prolif|GC_induced")
FOREIGN = {"T": ["B", "M"], "NK": ["B", "M"], "B": ["T", "M"], "M": ["T", "B"]}
GATES = dict(min_seed_stability=0.67, max_doublet_like=0.20, min_closest_is_own=0.50, min_isg_low=0.50,
             max_diss_z=1.0, max_gc_z=1.0, max_cohort_share=0.80, min_rep_auc=0.60, max_rep_p=0.05)


def lineage_of(parent, fine):
    """Coarse lineage (T, NK, B, M, other) from the CELLxGENE parent term, else from the fine author label."""
    p = str(parent).lower()
    if "dendritic" in p or "monocyte" in p or "macrophage" in p or "myeloid" in p:
        return "M"
    if "natural killer" in p:
        return "NK"
    if "t cell" in p or "thymocyte" in p:
        return "T"
    if "b cell" in p or "plasmablast" in p or "plasma cell" in p:
        return "B"
    f = str(fine).lower()
    if re.match(r"^(t4|t8|cd4|cd8|treg|mait|gd|t)($|_)", f):
        return "T"
    if re.match(r"^nk", f):
        return "NK"
    if re.match(r"^(b|pb|plasma)($|_)", f):
        return "B"
    if re.match(r"^(cm|ncm|cdc|pdc|mono|dc)($|_)", f):
        return "M"
    return "other"


def assign_halves(DN, seed=11):
    """Random donor halves stratified by condition x cohort (0/1)."""
    rng = np.random.default_rng(seed)
    strat = DN.cond.astype(str) + "|" + (DN.cohort.astype(str) if "cohort" in DN else "")
    half = pd.Series(0, index=DN.index)
    for _, idx in DN.groupby(strat).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        half.loc[idx[1::2]] = 1
    return half


def strat_perm(y, g, strata, nperm=2000, seed=0):
    """Stratified (van Elteren-like) mean difference g=1 minus g=0 within strata; permute g within strata."""
    y, g, strata = np.asarray(y, float), np.asarray(g, int), np.asarray(strata).astype(str)
    ok = np.isfinite(y)
    y, g, strata = y[ok], g[ok], strata[ok]
    use = []
    for s in np.unique(strata):
        m = strata == s
        if g[m].sum() >= 1 and (1 - g[m]).sum() >= 1:
            use.append(np.where(m)[0])
    if not use:
        return dict(n_strata=0, stat=None, p=None)

    def stat(gg):
        num = den = 0.0
        for ii in use:
            a, b = y[ii][gg[ii] == 1], y[ii][gg[ii] == 0]
            w = len(a) * len(b) / (len(a) + len(b))
            num += w * (a.mean() - b.mean())
            den += w
        return num / den
    obs = stat(g)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(nperm):
        gp = g.copy()
        for ii in use:
            gp[ii] = rng.permutation(g[ii])
        cnt += abs(stat(gp)) >= abs(obs)
    n1 = int(sum(g[ii].sum() for ii in use))
    n0 = int(sum((1 - g[ii]).sum() for ii in use))
    return dict(n_strata=len(use), n_g1=n1, n_g0=n0, stat=float(obs), p=float((1 + cnt) / (1 + nperm)))


def rows_X(A, cells, gidx=None):
    ci = A.obs_names.get_indexer(pd.Index(cells))
    if (ci < 0).any():
        raise KeyError(f"{(ci < 0).sum()} cells not found in AnnData")
    X = A.X[ci]
    if gidx is not None:
        X = X[:, gidx]
    return X


def control_genes(A):
    """Store log-normalized expression of lineage/platelet/erythroid/Ig marker genes as obs columns cg_<gene>."""
    genes = sorted({g for v in LINEAGE_MARKERS.values() for g in v if g in A.var_names})
    if not genes:
        log("control genes: none present")
        return []
    M = A.X[:, A.var_names.get_indexer(genes)]
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    for j, g in enumerate(genes):
        A.obs["cg_" + safe_name(g)] = M[:, j].astype(np.float32)
    log(f"control genes stored: {len(genes)}")
    return genes


def lineage_scores(obs, lab_col, par_col):
    """Per cell: lineage, foreign-lineage score (max mean expression of other-lineage markers), platelet, Hb, Ig."""
    def mscore(key):
        cols = ["cg_" + safe_name(g) for g in LINEAGE_MARKERS[key] if "cg_" + safe_name(g) in obs]
        return obs[cols].mean(1).values if cols else np.zeros(len(obs))
    sc_ = {k: mscore(k) for k in LINEAGE_MARKERS}
    lab = obs[lab_col].astype(str) if lab_col and lab_col in obs else pd.Series("", index=obs.index)
    par = obs[par_col].astype(str) if par_col and par_col in obs else pd.Series("", index=obs.index)
    keys = list(zip(par.values, lab.values))
    lmap = {pl: lineage_of(*pl) for pl in set(keys)}
    lin = np.array([lmap[k] for k in keys])
    foreign = np.zeros(len(obs), np.float32)
    for L, fl in FOREIGN.items():
        m = lin == L
        if m.any():
            foreign[m] = np.max(np.vstack([sc_[k][m] for k in fl]), 0)
    return pd.DataFrame(dict(lineage=lin, foreign=foreign, plt=sc_["PLT"], hb=sc_["HB"], ig=sc_["IG"]),
                        index=obs.index)


# ----------------------------------------------------------------------------------------------
# 10. seed agreement and composition-standardized flag rates
# ----------------------------------------------------------------------------------------------
@section("10_seed_agreement_composition")
def seed_and_composition(D, DN, isg_col):
    out = {}
    fc = sorted([c for c in D.columns if re.match(r"flag_s\d+$", c)])
    if len(fc) >= 2:
        F = D[fc].values.astype(bool)
        jac = {}
        for i in range(len(fc)):
            for j in range(i + 1, len(fc)):
                a, b = F[:, i], F[:, j]
                jac[f"{fc[i]}~{fc[j]}"] = float((a & b).sum() / max((a | b).sum(), 1))
        out["flag_jaccard_between_seeds"] = jac
        out["frac_flagged_any_seed"] = float(F.any(1).mean())
        out["frac_flagged_all_seeds"] = float(F.all(1).mean())
        out["frac_of_seed0_flags_flagged_in_all_seeds"] = float(F.all(1).sum() / max(F[:, 0].sum(), 1))
        dn = D.groupby("donor")[fc].mean()
        out["donor_flag_frac_spearman_between_seeds_min"] = float(dn.corr(method="spearman").values.min())
    # direct standardization of donor flag fraction to the healthy-query type composition
    H = D[D.is_sle == 0]
    w = H.y.value_counts(normalize=True)
    R_ = D.groupby(["donor", "y"]).flag_mean.agg(["mean", "size"]).reset_index()
    R_ = R_[R_["size"] >= 10]
    R_["w"] = R_.y.map(w).fillna(0)
    std = R_.groupby("donor").apply(lambda g: (g["mean"] * g.w).sum() / max(g.w.sum(), 1e-9))
    DN["flag_frac_std"] = std.reindex(DN.index)
    DN["flag_std_logit"] = np.log((DN.flag_frac_std.clip(1e-4, 1 - 1e-4)) / (1 - DN.flag_frac_std.clip(1e-4, 1 - 1e-4)))
    s, h = DN[DN.is_sle == 1], DN[DN.is_sle == 0]
    raw_d = s.flag_frac.median() - h.flag_frac.median()
    std_d = s.flag_frac_std.median() - h.flag_frac_std.median()
    out["composition_standardized"] = dict(
        raw_SLE_vs_healthy=mwu(s.flag_frac, h.flag_frac), std_SLE_vs_healthy=mwu(s.flag_frac_std, h.flag_frac_std),
        frac_of_median_difference_due_to_composition=float(1 - std_d / raw_d) if raw_d else None,
        rho_std_flag_vs_isg_SLE=spearman_boot(s.flag_frac_std, s.isg, nboot=500),
        rho_std_flag_vs_isg_healthy=spearman_boot(h.flag_frac_std, h.isg, nboot=500))
    # how much of the donor flag fraction is carried by rare / low-reference-density types?
    tr = D.groupby(["y", "cond"]).flag_mean.mean().unstack()
    tr["share_of_SLE_flags"] = D[D.is_sle == 1].groupby("y").flag_mean.sum() / max(D[D.is_sle == 1].flag_mean.sum(), 1e-9)
    tr["share_of_SLE_cells"] = D[D.is_sle == 1].y.value_counts(normalize=True)
    wcsv(tr.reset_index(), "flag_rate_and_share_by_type.csv")
    R["S10"] = out
    log(json.dumps(clean(out), default=jsonable)[:1500])
    return out


# ----------------------------------------------------------------------------------------------
# 11. matched strata (cohort x ancestry), within-pool contrasts, clinical/treatment proxies
# ----------------------------------------------------------------------------------------------
@section("11_matched_strata_pool_treatment")
def matched_strata(D, DN, A_obs, det, gc_col):
    try:
        import statsmodels.formula.api as smf
    except ImportError as e:
        smf = None
        log(f"statsmodels unavailable ({e!r}); model parts of section 11 skipped")
    out = {}
    feats = [c for c in ["flag_frac", "flag_frac_std", "iiu", "iiu_strict", "isg", "novelty_mean", "u_only",
                         "gc"] if c in DN]
    # (a) cohort x ancestry strata that contain both conditions
    parts = [collapse_levels(DN[k], 1) for k in ["cohort", "ancestry"] if k in DN]
    if parts:
        stratum = parts[0]
        for p in parts[1:]:
            stratum = stratum + "|" + p
    else:
        stratum = pd.Series("all", index=DN.index)
    DN["stratum"] = stratum
    ct = pd.crosstab(DN.stratum, DN.cond)
    wcsv(ct.reset_index(), "strata_cohort_ancestry_by_condition.csv")
    both = ct[(ct.get("SLE", 0) > 0) & (ct.get("healthy", 0) > 0)].index
    DN["matched_stratum"] = DN.stratum.isin(both)
    out["n_donors_in_matched_strata"] = DN.groupby("cond").matched_stratum.sum().to_dict()
    out["n_donors_unmatched"] = (~DN.matched_stratum).groupby(DN.cond).sum().to_dict()
    res = {}
    for f in feats:
        res[f] = dict(stratified_perm=strat_perm(DN[f], DN.is_sle, DN.stratum, nperm=args.nperm),
                      unstratified=mwu(DN[DN.is_sle == 1][f], DN[DN.is_sle == 0][f]))
    out["SLE_vs_healthy_within_strata"] = res
    # SLE donors whose ancestry/cohort has no healthy counterpart: are they flagged more than matched SLE?
    S = DN[DN.is_sle == 1]
    out["SLE_unmatched_vs_matched"] = {f: mwu(S[~S.matched_stratum][f], S[S.matched_stratum][f]) for f in feats}
    if "ancestry" in DN:
        out["within_SLE_flag_by_ancestry"] = {str(k): dict(n=int(len(g)), median_flag=float(g.flag_frac.median()),
                                                           median_isg=float(g.isg.median()))
                                              for k, g in S.groupby(S.ancestry.astype(str))}
    # within-SLE model with treatment proxy (glucocorticoid score) and stratum fixed effects
    try:
        if smf is None:
            raise ImportError("statsmodels")
        F = DN[DN.is_sle == 1].copy()
        F["isg_z"], F["diss_z"] = zs(F.isg), zs(F.diss)
        terms = ["isg_z", "diss_z"]
        if "gc" in F:
            F["gc_z"] = zs(F.gc)
            terms.append("gc_z")
        if F.stratum.nunique() > 1:
            terms.append("C(stratum)")
        if "status" in F and collapse_levels(F.status, 3).nunique() > 1:
            F["status_c"] = collapse_levels(F.status, 3)
            terms.append("C(status_c)")
        mods = {}
        for yv in ["flag_logit", "flag_std_logit", "iiu_logit"]:
            if yv in F:
                fml = f"{yv} ~ " + " + ".join(terms)
                fit = smf.ols(fml, data=F).fit(cov_type="HC3")
                mods[yv] = dict(formula=fml, n=int(fit.nobs), r2=float(fit.rsquared),
                                terms=coef_table(fit, keep=["isg_z", "diss_z", "gc_z", "status"]))
        out["SLE_models_with_GC_and_strata"] = mods
    except Exception as e:
        out["SLE_models_with_GC_and_strata"] = dict(error=repr(e))
    # all donors, stratum fixed effects only within matched strata
    try:
        if smf is None:
            raise ImportError("statsmodels")
        F = DN[DN.matched_stratum].copy()
        F["isg_z"], F["diss_z"] = zs(F.isg), zs(F.diss)
        mods = {}
        for yv in ["flag_logit", "flag_std_logit", "iiu_logit"]:
            if yv in F:
                for nm, rhs in [("SLE+strata", "is_sle + C(stratum)"),
                                ("SLE+ISG+diss+strata", "is_sle + isg_z + diss_z + C(stratum)")]:
                    fml = f"{yv} ~ {rhs}" if F.stratum.nunique() > 1 else f"{yv} ~ {rhs.replace(' + C(stratum)', '')}"
                    fit = smf.ols(fml, data=F).fit(cov_type="HC3")
                    mods[f"{yv}: {nm}"] = dict(formula=fml, n=int(fit.nobs),
                                               terms=coef_table(fit, keep=["is_sle", "isg_z", "diss_z"]))
        out["matched_strata_models"] = mods
    except Exception as e:
        out["matched_strata_models"] = dict(error=repr(e))
    # (b) within-pool / lane contrasts (cell-level lane, shared ambient and handling)
    lane = det.get("lane")
    if lane and lane in A_obs:
        L = D[["donor", "is_sle", "flag_mean"] + (["IIU"] if "IIU" in D else [])].copy()
        L["lane_cell"] = A_obs.loc[L.index, lane].astype(str).values
        dl = L.groupby(["lane_cell", "donor"]).agg(flag=("flag_mean", "mean"), is_sle=("is_sle", "first"),
                                                    n=("flag_mean", "size"),
                                                    **({"iiu": ("IIU", "mean")} if "IIU" in L else {})).reset_index()
        dl = dl[dl.n >= 50]
        mix = dl.groupby("lane_cell").is_sle.agg(["min", "max"])
        mixed = mix[(mix["min"] == 0) & (mix["max"] == 1)].index
        out["lanes_total"] = int(dl.lane_cell.nunique())
        out["lanes_with_both_conditions"] = int(len(mixed))
        if len(mixed) >= 3:
            dm = dl[dl.lane_cell.isin(mixed)]
            out["within_lane_flag"] = strat_perm(dm.flag, dm.is_sle, dm.lane_cell, nperm=args.nperm)
            if "iiu" in dm:
                out["within_lane_iiu"] = strat_perm(dm.iiu, dm.is_sle, dm.lane_cell, nperm=args.nperm)
            diffs = dm.groupby("lane_cell").apply(lambda g: g[g.is_sle == 1].flag.mean() - g[g.is_sle == 0].flag.mean())
            out["within_lane_diff_summary"] = boot_mean_ci(diffs.values)
            wcsv(dm, "within_lane_donor_flag.csv")
        else:
            out["within_lane_note"] = "fewer than 3 lanes contain both SLE and healthy donors: pool and disease are " \
                                      "confounded and cannot be separated in this cohort"
    R["S11"] = out
    log(json.dumps(clean({k: v for k, v in out.items() if k != "SLE_vs_healthy_within_strata"}),
                   default=jsonable)[:2000])
    return out


# ----------------------------------------------------------------------------------------------
# 12. doublets, ambient RNA (Ig, platelet, Hb) and QC
# ----------------------------------------------------------------------------------------------
@section("12_doublet_ambient_qc")
def doublet_ambient_qc(D, A_obs, det):
    out = {}
    lab_col, role_col, par_col = det.get("label"), det.get("role"), det.get("parent")
    LS = lineage_scores(A_obs, lab_col, par_col)
    ref = (A_obs[role_col].astype(str) == "ref").values if role_col and role_col in A_obs else \
        A_obs.index.isin(D.index[D.is_sle == 0])
    labs = A_obs[lab_col].astype(str).values if lab_col and lab_col in A_obs else np.array([""] * len(A_obs))
    thr = pd.Series(LS.foreign.values[ref]).groupby(labs[ref]).quantile(0.99)
    nref = pd.Series(labs[ref]).value_counts()
    thr = thr[nref.reindex(thr.index).fillna(0) >= 50]
    lin_thr = pd.Series(LS.foreign.values[ref]).groupby(LS.lineage.values[ref]).quantile(0.99)
    for k in ["plt", "hb", "ig"]:
        out[f"ref_q99_{k}"] = float(np.quantile(LS[k].values[ref], 0.99))
    L = LS.loc[D.index]
    t_cell = D.y.map(thr).astype(float)
    t_cell = t_cell.fillna(L.lineage.map(lin_thr).astype(float)).fillna(np.inf)
    D["lineage"] = L.lineage.values
    D["foreign"] = L.foreign.values
    D["doublet_like"] = (L.foreign.values > t_cell.values) & (L.lineage.values != "other")
    D["plt_hi"] = L.plt.values > out["ref_q99_plt"]
    D["hb_hi"] = L.hb.values > out["ref_q99_hb"]
    D["ig_hi_nonB"] = (L.ig.values > out["ref_q99_ig"]) & (L.lineage.values != "B")
    if "doublet_score" in A_obs:
        D["scrublet"] = A_obs.loc[D.index, "doublet_score"].astype(float).values
    feats = ["doublet_like", "plt_hi", "hb_hi", "ig_hi_nonB"]
    rows = []
    for (cn, fl), g in D.groupby(["cond", "flag"]):
        r = dict(cond=cn, flagged=bool(fl), n=len(g), **{f: float(g[f].mean()) for f in feats})
        if "scrublet" in g:
            r["scrublet_median"] = float(g.scrublet.median())
        r.update(pct_mito_median=float(g.pct_mito.median()), n_genes_median=float(g.n_genes.median()))
        rows.append(r)
    T = pd.DataFrame(rows)
    wcsv(T, "doublet_ambient_qc_flagged_vs_kept.csv")
    out["flagged_vs_kept"] = T.to_dict(orient="records")
    # share of the SLE-healthy flag difference explained by artifact-like cells
    art = D[feats].any(axis=1)
    D["artifact_like"] = art.values
    clean_ = D[~art]
    out["flag_rate_all"] = D.groupby("cond").flag_mean.mean().to_dict()
    out["flag_rate_excluding_artifact_like"] = clean_.groupby("cond").flag_mean.mean().to_dict()
    out["frac_flagged_artifact_like"] = D[D.flag].groupby("cond").artifact_like.mean().to_dict()
    # QC strata: flag rate by n_genes decile (within condition): low-quality cells sit far from the reference
    D["_ng_bin"] = pd.qcut(D.n_genes.rank(method="first"), 10, labels=False)
    q = D.groupby(["_ng_bin", "cond"]).flag_mean.mean().unstack()
    wcsv(q.reset_index(), "flag_rate_by_ngenes_decile.csv")
    out["flag_rate_lowest_vs_middle_ngenes_decile"] = {c: [float(q.loc[0, c]), float(q.loc[5, c])] for c in q.columns}
    # per donor: Ig-ambient in non-B cells vs plasmablast fraction (ambient from PB expansion) in SLE
    dd = D.groupby("donor").agg(ig_nonB=("ig_hi_nonB", "mean"), is_sle=("is_sle", "first"))
    pb = D.assign(pb=D.y.astype(str).str.contains(r"^PB|plasma", case=False, regex=True)).groupby("donor").pb.mean()
    dd["pb_frac"] = pb
    out["donor_ig_nonB_vs_PB_frac_spearman"] = spearman_boot(dd.ig_nonB, dd.pb_frac, nboot=300)
    D.drop(columns=["_ng_bin"], inplace=True)
    R["S12"] = out
    log(json.dumps(clean(out), default=jsonable)[:2000])
    return out


# ----------------------------------------------------------------------------------------------
# 13. residual ISG confounding of "ISG-independent" novelty (split-half ISG test) + strict IIU
# ----------------------------------------------------------------------------------------------
@section("13_isg_residual_confounding")
def isg_residual_confounding(D, DN, isg_col, diss_col, gc_col, ifng_col):
    out = {}
    if "gs_ISG_halfA" not in D or "gs_ISG_halfB" not in D:
        raise KeyError("ISG half scores missing")
    A_, B_ = "gs_ISG_halfA", "gs_ISG_halfB"
    D["isg_donor_type"] = D.groupby(["donor", "y"])[isg_col].transform("mean")
    rows = []
    D["nov_resid_strict"] = np.nan
    strict_cov = [isg_col, A_, B_, "isg_donor_type", diss_col, "log_genes", "log_counts", "pct_mito", "pct_ribo"] + \
        [c for c in [gc_col, ifng_col, "foreign"] if c and c in D]
    for t, idx in D.groupby("y").groups.items():
        g = D.loc[idx]
        if len(g) < 200:
            continue
        yv = g.log_novelty.values

        def resid(cols):
            Xm = np.column_stack([np.ones(len(g))] + [zs(g[c].astype(float).values) for c in cols])
            b, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
            return yv - Xm @ b
        rA = resid([A_, diss_col, "log_genes", "log_counts", "pct_mito"])
        rS = resid(strict_cov)
        D.loc[idx, "nov_resid_strict"] = rS
        rel = stats.spearmanr(g[A_], g[B_])[0]
        rows.append(dict(type=t, n=len(g), split_half_rho_isgA_isgB=rel,
                         rho_novelty_isgB=stats.spearmanr(yv, g[B_])[0],
                         rho_residA_isgB=stats.spearmanr(rA, g[B_])[0],
                         rho_residStrict_isgB=stats.spearmanr(rS, g[B_])[0],
                         rho_residStrict_isgfull=stats.spearmanr(rS, g[isg_col])[0]))
    T = pd.DataFrame(rows)
    wcsv(T, "isg_split_half_residual_confounding.csv")
    out["median_split_half_reliability"] = float(T.split_half_rho_isgA_isgB.median())
    out["median_rho_novelty_isgB"] = float(T.rho_novelty_isgB.median())
    out["median_rho_residualizedOnISGA_isgB"] = float(T.rho_residA_isgB.median())
    out["leakage_fraction"] = float((T.rho_residA_isgB / T.rho_novelty_isgB.replace(0, np.nan)).median())
    out["interpretation"] = ("leakage_fraction = residual association of novelty with held-out ISG genes after "
                             "adjusting for the other ISG half, relative to the unadjusted association. Values "
                             "> 0.2 mean a single noisy ISG score under-adjusts and 'ISG-independent' cells still "
                             "carry IFN signal; use IIU_strict.")
    # strict IIU: top within-type decile of the strict residual among flagged cells, AND ISG-low
    q90 = D.groupby("y").nov_resid_strict.transform(lambda v: v.quantile(0.9))
    D["IIU_strict"] = (D.nov_resid_strict > q90) & D.flag & D.isg_low
    dd = D.groupby("donor").IIU_strict.mean()
    DN["iiu_strict"] = dd.reindex(DN.index)
    out["IIU_strict_rate"] = D.groupby("cond").IIU_strict.mean().to_dict()
    out["IIU_strict_donor_SLE_vs_healthy"] = mwu(DN[DN.is_sle == 1].iiu_strict, DN[DN.is_sle == 0].iiu_strict)
    out["IIU_strict_vs_isg_SLE"] = spearman_boot(DN[DN.is_sle == 1].iiu_strict, DN[DN.is_sle == 1].isg, nboot=500)
    if "IIU" in D:
        a, b = D.IIU.values.astype(bool), D.IIU_strict.values.astype(bool)
        out["jaccard_IIU_vs_IIU_strict"] = float((a & b).sum() / max((a | b).sum(), 1))
    R["S13"] = out
    log(json.dumps(clean(out), default=jsonable)[:1500])
    return out


# ----------------------------------------------------------------------------------------------
# 14. donor-paired pseudobulk DE (fixes pseudoreplication of cell-level Wilcoxon in S5/S6)
# ----------------------------------------------------------------------------------------------
def pseudobulk_paired(A, g, grp_a, grp_b, min_cells=10, max_per=300, rng=None):
    """g: cells of one type/condition; grp_a/grp_b boolean Series. Mean log-expression per donor x group,
    paired difference across donors, one-sample t-test and sign consistency."""
    rng = rng or np.random.default_rng(13)
    ca, cb = g[grp_a].groupby("donor").size(), g[grp_b].groupby("donor").size()
    donors = sorted(set(ca[ca >= min_cells].index) & set(cb[cb >= min_cells].index))
    if len(donors) < 6:
        return None, len(donors)
    cells, rows = [], []
    for k, d in enumerate(donors):
        for j, m in enumerate([grp_a, grp_b]):
            c = g.index[m & (g.donor == d)].values
            if len(c) > max_per:
                c = rng.choice(c, max_per, replace=False)
            cells.append(c)
            rows += [2 * k + j] * len(c)
    cells = np.concatenate(cells)
    rows = np.array(rows)
    X = rows_X(A, cells).tocsr()
    cnt = np.bincount(rows, minlength=2 * len(donors)).astype(float)
    Mi = sp.csr_matrix((1.0 / cnt[rows], (rows, np.arange(len(rows)))), shape=(2 * len(donors), len(rows)))
    Mean = np.asarray((Mi @ X).todense())
    det_ = np.asarray((Mi @ (X > 0).astype(np.float32)).todense())
    keep = (det_[0::2].mean(0) >= 0.05) | (det_[1::2].mean(0) >= 0.05)
    diff = Mean[0::2][:, keep] - Mean[1::2][:, keep]
    tt = stats.ttest_1samp(diff, 0, axis=0)
    df = pd.DataFrame(dict(gene=np.asarray(A.var_names)[keep], mean_diff=diff.mean(0), t=tt.statistic,
                           p=tt.pvalue, frac_donors_up=(diff > 0).mean(0)))
    df["q"] = bh(df.p.values)
    return df.sort_values("t", ascending=False), len(donors)


@section("14_pseudobulk_DE")
def pseudobulk_de(D, A, gs):
    out = {}
    if "IIU" not in D:
        return out
    cnt = D[(D.is_sle == 1) & D.IIU].y.value_counts()
    types = cnt[cnt >= 50].index[:args.max_sub_types]
    for t in types:
        r = {}
        for cn, sle in [("SLE", 1), ("healthy", 0)]:
            g = D[(D.y == t) & (D.is_sle == sle)]
            df, nd = pseudobulk_paired(A, g, g.IIU, (~g.IIU) & (~g.flag))
            if df is None:
                r[cn] = dict(n_donors=nd, note="fewer than 6 donors with >=10 IIU and >=10 kept cells")
                continue
            df.head(300).to_csv(os.path.join(OUT, f"pb_IIU_{safe_name(t)}_{cn}.csv"), index=False)
            up = df[(df.q < 0.05) & (df.mean_diff > 0.1) & (df.frac_donors_up >= 0.7)].gene.tolist()
            E = enrich(up[:150], set(df.gene), gs)
            r[cn] = dict(n_donors=nd, n_up_q05=len(up), top_up=up[:25],
                         top_sets=E.head(6)[["set", "overlap", "q"]].values.tolist() if len(E) else [])
            cell_path = os.path.join(OUT, f"de_IIU_{safe_name(t)}_{cn}.csv")
            if os.path.exists(cell_path):
                cl = set(de_up_list(pd.read_csv(cell_path), n=100))
                pb = set(up[:100])
                r[cn]["jaccard_with_cell_level_top100"] = len(cl & pb) / max(len(cl | pb), 1)
                r[cn]["cell_level_only"] = sorted(cl - pb)[:30]
        if "SLE" in r and "healthy" in r and "top_up" in r["SLE"] and "top_up" in r["healthy"]:
            a, b = set(r["SLE"]["top_up"]), set(r["healthy"]["top_up"])
            r["SLE_specific_pseudobulk"] = sorted(a - b)
        out[t] = r
        log(f"  pseudobulk IIU {t}: SLE up {r.get('SLE', {}).get('top_up', [])[:10]}")
    R["S14"] = out
    return out


# ----------------------------------------------------------------------------------------------
# 15. view disagreement: U-only (scVI) vs S-only (scANVI) deviation; thresholds from half of healthy donors
# ----------------------------------------------------------------------------------------------
@section("15_view_disagreement_U_vs_S")
def view_disagreement(D, DN, isg_col):
    out = {}
    if "knn_dist_U" not in D or "knn_dist_S" not in D:
        raise KeyError("knn_dist_U / knn_dist_S not in res npz")
    half = assign_halves(DN, seed=21)
    D["_hhalf"] = D.donor.map(half)
    cal = (D.is_sle == 0) & (D._hhalf == 0)
    out["n_calibration_healthy_donors"] = int(((DN.is_sle == 0) & (half == 0)).sum())
    for v in ["U", "S"]:
        c = f"knn_dist_{v}"
        D[f"d{v}"] = D[c].astype(float) / D.loc[cal, c].astype(float).median()
        q = D[cal].groupby("y")[f"d{v}"].quantile(0.95)
        nc = D[cal].y.value_counts()
        q = q[nc.reindex(q.index).fillna(0) >= 200]
        thr = D.y.map(q).astype(float).fillna(D.loc[cal, f"d{v}"].quantile(0.95))
        D[f"hi{v}"] = D[f"d{v}"] > thr
    D["U_only"] = D.hiU & ~D.hiS
    D["S_only"] = D.hiS & ~D.hiU
    D["US_both"] = D.hiU & D.hiS
    comp = D[~cal]
    dd = comp.groupby("donor").agg(u_only=("U_only", "mean"), s_only=("S_only", "mean"), both=("US_both", "mean"))
    for c in dd:
        DN[c] = dd[c].reindex(DN.index)
    test = DN[(DN.is_sle == 1) | (half == 1)]
    out["donor_rates_SLE_vs_healthy_half2"] = {c: mwu(test[test.is_sle == 1][c], test[test.is_sle == 0][c])
                                               for c in ["u_only", "s_only", "both"]}
    S = DN[DN.is_sle == 1]
    out["rho_vs_isg_SLE"] = {c: spearman_boot(S[c], S.isg, nboot=500) for c in ["u_only", "s_only", "both"]}
    bt = comp.groupby(["y", "cond"])[["U_only", "S_only", "US_both"]].mean().unstack("cond")
    bt.columns = [f"{a}_{b}" for a, b in bt.columns]
    wcsv(bt.reset_index(), "view_disagreement_by_type.csv")
    # are U-only cells ISG-low (i.e. not the known IFN axis)?
    out["frac_isg_low"] = {k: D[D[k] & (D.is_sle == 1)].isg_low.mean() for k in ["U_only", "S_only", "US_both"]}
    D.drop(columns=["_hhalf"], inplace=True)
    R["S15"] = out
    log(json.dumps(clean(out), default=jsonable)[:1500])
    return out


# ----------------------------------------------------------------------------------------------
# 16. acceptance gates for candidate novel states (S6 subclusters)
# ----------------------------------------------------------------------------------------------
@section("16_state_acceptance_gates")
def state_gates(D, DN, A, A_obs, det, gs, isg_key, diss_key, isg_col, diss_col, gc_col):
    p = os.path.join(OUT, "subclusters_flagged.csv")
    if "subcluster" not in D or not os.path.exists(p):
        raise FileNotFoundError("no subclusters from section 6")
    T = pd.read_csv(p)
    if not len(T):
        return {}
    called = T[T.state_called.astype(str).str.lower() == "true"].head(40)
    excl = set(gs.get(isg_key, [])) | set(gs.get(diss_key, [])) | set(IEG_TARGETS)
    for k, v in gs.items():
        if re.search(r"isg|ifn|interferon|dissoc|ieg|heat|shock|ribo|mito", k, re.I):
            excl |= set(v)
    var = pd.Index(A.var_names.astype(str))
    excl |= set(LINEAGE_MARKERS["PLT"]) | set(LINEAGE_MARKERS["HB"]) | set(LINEAGE_MARKERS["IG"])
    bad = var.isin(list(excl)) | var.str.match(r"^(MT-|RP[SL]\d|HSP|IG[HKL][VCJ]?|JCHAIN|HB[ABDGMQZ])")
    half = assign_halves(DN, seed=31)
    lab_col, role_col = det.get("label"), det.get("role")
    is_ref = (A_obs[role_col].astype(str) == "ref") if role_col and role_col in A_obs else None
    rng = np.random.default_rng(17)
    rows = []
    for _, st in called.iterrows():
        t, name = st["type"], st["subcluster"]
        sc_cells = D.index[D.subcluster.astype(str) == str(name)]
        if len(sc_cells) < 50:
            continue
        Ds, Dt = D.loc[sc_cells], D[D.y == t]
        kept_t = Dt[~Dt.flag]
        r = dict(type=t, subcluster=name, n_cells=len(sc_cells), n_donors=int(Ds.donor.nunique()))
        r["seed_stability"] = float(Ds.flag_mean.mean())
        r["frac_flagged_all_seeds"] = float((Ds.flag_mean >= 0.999).mean())
        r["doublet_like"] = float(Ds.doublet_like.mean()) if "doublet_like" in Ds else np.nan
        r["artifact_like"] = float(Ds.artifact_like.mean()) if "artifact_like" in Ds else np.nan
        r["closest_is_own_label"] = float((Ds.closest_label.astype(str) == str(t)).mean()) if "closest_label" in Ds else np.nan
        r["closest_other_label_top"] = Ds.closest_label[Ds.closest_label.astype(str) != str(t)].value_counts().index[0] \
            if "closest_label" in Ds and (Ds.closest_label.astype(str) != str(t)).any() else ""
        r["arcane_pred_is_own"] = float((Ds["pred_ARCANE"].astype(str) == str(t)).mean()) if "pred_ARCANE" in Ds else np.nan
        r["isg_low"] = float(Ds.isg_low.mean())
        for nm, col in [("diss_z", diss_col), ("gc_z", gc_col), ("isg_z", isg_col)]:
            if col and col in Ds and len(kept_t) > 10:
                r[nm] = float((Ds[col].mean() - kept_t[col].mean()) / (kept_t[col].std() + 1e-9))
        for k in ["cohort", "ancestry", "lane"]:
            if k in Ds:
                r[f"max_{k}_share"] = float(Ds[k].astype(str).value_counts(normalize=True).iloc[0])
                r[f"max_{k}_share_type"] = float(Dt[k].astype(str).value_counts(normalize=True).iloc[0])
        r["frac_sle"] = float(Ds.is_sle.mean())
        # does the state match a KNOWN activated/disease subset (gene-set z vs kept cells of the type)?
        known = {}
        for k in gs:
            if re.search(KNOWN_STATE_PAT, k, re.I) and "gs_" + safe_name(k) in Ds and len(kept_t) > 10:
                c = "gs_" + safe_name(k)
                known[k] = float((Ds[c].mean() - kept_t[c].mean()) / (kept_t[c].std() + 1e-9))
        if known:
            kb = max(known, key=known.get)
            r["best_known_set"], r["best_known_z"] = kb, known[kb]
            r["matches_known_state"] = bool(known[kb] >= 1.5)
        # donor-split replication: signature from discovery half, tested in validation half
        for a_, b_ in [(0, 1), (1, 0)]:
            tag = f"rep{a_}{b_}"
            try:
                dA = half[half == a_].index
                dB = half[half == b_].index
                sA = Ds[Ds.donor.isin(dA)].index.values
                kA = kept_t[kept_t.donor.isin(dA)].index.values
                if len(sA) < 30 or Ds[Ds.donor.isin(dA)].donor.nunique() < 3 or len(kA) < 100:
                    r[tag + "_note"] = "too few discovery cells/donors"
                    continue
                sA = rng.choice(sA, min(len(sA), 5000), replace=False)
                kA = rng.choice(kA, min(len(kA), 5000), replace=False)
                Xs, Xk = rows_X(A, sA), rows_X(A, kA)
                ms, mk = np.asarray(Xs.mean(0)).ravel(), np.asarray(Xk.mean(0)).ravel()
                ps = np.asarray((Xs > 0).mean(0)).ravel()
                diff = ms - mk
                ok = (~bad) & (ps >= 0.2) & (diff > 0.25)
                gi = np.where(ok)[0][np.argsort(-diff[ok])][:20]
                if len(gi) < 5:
                    r[tag + "_note"] = "fewer than 5 non-ISG/non-IEG signature genes"
                    continue
                r[tag + "_signature"] = ",".join(var[gi])
                vB = Dt[Dt.donor.isin(dB)]
                vB = vB.sample(frac=1.0, random_state=1)
                vB = vB[vB.groupby("donor").cumcount().values < 300]
                sc_v = np.asarray(rows_X(A, vB.index.values, gi).mean(1)).ravel()
                if is_ref is not None and lab_col:
                    rc = A_obs.index[is_ref.values & (A_obs[lab_col].astype(str) == str(t)).values]
                    rc = rng.choice(rc, min(len(rc), 5000), replace=False) if len(rc) else rc
                    thr = np.quantile(np.asarray(rows_X(A, rc, gi).mean(1)).ravel(), 0.95) if len(rc) >= 50 else None
                else:
                    thr = None
                if thr is None:
                    hv = vB.is_sle.values == 0
                    thr = np.quantile(sc_v[hv], 0.95)
                vB = vB.assign(sig_hi=sc_v > thr)
                dd = vB.groupby("donor").agg(hi=("sig_hi", "mean"), n=("sig_hi", "size"), is_sle=("is_sle", "first"))
                dd = dd[dd.n >= 20]
                m = mwu(dd[dd.is_sle == 1].hi, dd[dd.is_sle == 0].hi)
                r[tag + "_auc"] = m.get("auc_a_gt_b")
                r[tag + "_p"] = m.get("p")
                r[tag + "_n_sle"], r[tag + "_n_healthy"] = m.get("n_a"), m.get("n_b")
                ctab = pd.crosstab(vB.sig_hi, vB.flag)
                if ctab.shape == (2, 2):
                    r[tag + "_OR_flag_in_sighi"] = float(stats.fisher_exact(ctab.values[::-1, ::-1])[0])
                dd["isg"] = DN.isg.reindex(dd.index)
                ss = dd[dd.is_sle == 1]
                r[tag + "_rho_isg_SLE"] = spearman_boot(ss.hi, ss.isg, nboot=200)["rho"]
                try:
                    import statsmodels.formula.api as smf
                    dd["hi_logit"] = logit_frac(dd.hi * dd.n, dd.n)
                    dd["isg_z"] = zs(dd.isg)
                    fit = smf.ols("hi_logit ~ is_sle + isg_z", data=dd).fit(cov_type="HC3")
                    r[tag + "_sle_coef_adjISG"] = float(fit.params["is_sle"])
                    r[tag + "_sle_p_adjISG"] = float(fit.pvalues["is_sle"])
                except Exception as e:
                    r[tag + "_ols_error"] = repr(e)
            except Exception as e:
                r[tag + "_error"] = repr(e)
        G = GATES

        def ge(v, thr):
            return bool(v is not None and np.isfinite(v) and v >= thr)

        def le(v, thr):
            return bool(v is not None and np.isfinite(v) and v <= thr)
        r["gate_seed"] = ge(r["seed_stability"], G["min_seed_stability"])
        r["gate_not_doublet"] = le(r.get("doublet_like", np.nan), G["max_doublet_like"])
        r["gate_not_label_boundary"] = ge(r.get("closest_is_own_label", np.nan), G["min_closest_is_own"])
        r["gate_isg_low"] = ge(r["isg_low"], G["min_isg_low"])
        r["gate_not_dissociation"] = le(r.get("diss_z", np.nan), G["max_diss_z"])
        r["gate_not_glucocorticoid"] = le(r.get("gc_z", np.nan), G["max_gc_z"]) if gc_col else True
        r["gate_not_one_cohort"] = all(le(r.get(f"max_{k}_share", 0), max(G["max_cohort_share"],
                                                                         r.get(f"max_{k}_share_type", 0) + 0.1))
                                       for k in ["cohort", "ancestry", "lane"])
        rep = [ge(r.get(f"rep{a}{b}_auc"), G["min_rep_auc"]) and le(r.get(f"rep{a}{b}_p"), G["max_rep_p"])
               for a, b in [(0, 1), (1, 0)]]
        r["gate_replicates_both_halves"] = all(rep)
        r["gate_beyond_ISG"] = all(le(r.get(f"rep{a}{b}_sle_p_adjISG"), 0.05) for a, b in [(0, 1), (1, 0)])
        gcols = [c for c in r if c.startswith("gate_")]
        r["n_gates_passed"] = int(sum(r[c] for c in gcols))
        r["ACCEPT"] = all(r[c] for c in gcols)
        rows.append(r)
        log(f"  gates {name}: {r['n_gates_passed']}/{len(gcols)} accept={r['ACCEPT']}")
    G = pd.DataFrame(rows)
    wcsv(G, "state_acceptance_gates.csv")
    if len(G) and "matches_known_state" in G:
        G["ACCEPT_as_novel"] = G.ACCEPT & ~G.matches_known_state.fillna(False).astype(bool)
        wcsv(G, "state_acceptance_gates.csv")
    out = dict(n_states=int(len(G)), n_accepted=int(G.ACCEPT.sum()) if len(G) else 0,
               n_accepted_not_known=int(G.ACCEPT_as_novel.sum()) if "ACCEPT_as_novel" in G else None,
               gate_pass_rates={c: float(G[c].mean()) for c in G.columns if c.startswith("gate_")} if len(G) else {},
               accepted=G[G.ACCEPT].to_dict(orient="records") if len(G) else [], thresholds=GATES)
    R["S16"] = out
    return out


def write_controls_md():
    L = ["# Control analyses (sections 10-16)", ""]
    s10 = R.get("S10") or {}
    L += ["## Seeds and composition",
          f"- flag Jaccard between seeds: {json.dumps({k: round(v, 3) for k, v in (s10.get('flag_jaccard_between_seeds') or {}).items()})}; "
          f"seed-0 flags flagged in all seeds: {fmt(s10.get('frac_of_seed0_flags_flagged_in_all_seeds'))}",
          f"- composition-standardized donor flag rate SLE vs healthy: "
          f"{json.dumps(clean(gget(s10, 'composition_standardized', 'std_SLE_vs_healthy') or {}))}; share of median "
          f"difference due to composition {fmt(gget(s10, 'composition_standardized', 'frac_of_median_difference_due_to_composition'))}", ""]
    s11 = R.get("S11") or {}
    L.append("## Matched strata (cohort x ancestry), pools, treatment proxy")
    L.append(f"- donors in matched strata: {s11.get('n_donors_in_matched_strata')}; unmatched: {s11.get('n_donors_unmatched')}")
    for f, v in (s11.get("SLE_vs_healthy_within_strata") or {}).items():
        sp_ = v.get("stratified_perm", {})
        L.append(f"- {f}: stratified SLE-healthy diff {fmt(sp_.get('stat'))} (perm p {fmt(sp_.get('p'))}, strata "
                 f"{sp_.get('n_strata')}); unstratified MWU p {fmt(gget(v, 'unstratified', 'p'))}")
    L.append(f"- lanes with both conditions: {s11.get('lanes_with_both_conditions')} / {s11.get('lanes_total')}; "
             f"within-lane flag diff {json.dumps(clean(s11.get('within_lane_flag')))} {s11.get('within_lane_note', '')}")
    for yv, m in (s11.get("SLE_models_with_GC_and_strata") or {}).items():
        if isinstance(m, dict) and "terms" in m:
            L.append(f"- SLE-only {yv}: " + "; ".join(f"{t['term']} {fmt(t['coef'])} (p {fmt(t['p'])})" for t in m["terms"]))
    L.append("")
    s12 = R.get("S12") or {}
    L += ["## Doublets / ambient / QC",
          f"- flag rate all: {json.dumps(clean(s12.get('flag_rate_all')))}; excluding artifact-like cells: "
          f"{json.dumps(clean(s12.get('flag_rate_excluding_artifact_like')))}",
          f"- fraction of flagged cells that are artifact-like: {json.dumps(clean(s12.get('frac_flagged_artifact_like')))}",
          f"- flag rate lowest vs middle n_genes decile: {json.dumps(clean(s12.get('flag_rate_lowest_vs_middle_ngenes_decile')))}", ""]
    s13 = R.get("S13") or {}
    L += ["## Residual ISG confounding (split-half)",
          f"- split-half ISG reliability {fmt(s13.get('median_split_half_reliability'))}; rho(novelty, ISG_B) "
          f"{fmt(s13.get('median_rho_novelty_isgB'))}; after adjusting ISG_A {fmt(s13.get('median_rho_residualizedOnISGA_isgB'))}; "
          f"leakage {fmt(s13.get('leakage_fraction'))}",
          f"- IIU_strict rate {json.dumps(clean(s13.get('IIU_strict_rate')))}; donor MWU p "
          f"{fmt(gget(s13, 'IIU_strict_donor_SLE_vs_healthy', 'p'))}; rho with ISG (SLE) "
          f"{fmt(gget(s13, 'IIU_strict_vs_isg_SLE', 'rho'))}", ""]
    L.append("## Pseudobulk (donor-paired) DE of IIU vs kept")
    for t, r in (R.get("S14") or {}).items():
        L.append(f"- {t}: SLE donors {gget(r, 'SLE', 'n_donors')}, up {gget(r, 'SLE', 'n_up_q05')}: "
                 f"{', '.join((gget(r, 'SLE', 'top_up') or [])[:12])}; Jaccard with cell-level "
                 f"{fmt(gget(r, 'SLE', 'jaccard_with_cell_level_top100'))}; SLE-specific {', '.join((r.get('SLE_specific_pseudobulk') or [])[:10])}")
    L.append("")
    s15 = R.get("S15") or {}
    L.append("## View disagreement (U-only vs S-only deviation)")
    for c, v in (s15.get("donor_rates_SLE_vs_healthy_half2") or {}).items():
        L.append(f"- {c}: SLE median {fmt(v.get('median_a'))} vs healthy {fmt(v.get('median_b'))}, AUC {fmt(v.get('auc_a_gt_b'))}, "
                 f"p {fmt(v.get('p'))}; rho with ISG (SLE) {fmt(gget(s15, 'rho_vs_isg_SLE', c, 'rho'))}")
    L.append("")
    s16 = R.get("S16") or {}
    L += ["## Candidate state acceptance gates",
          f"- states evaluated {s16.get('n_states')}, accepted {s16.get('n_accepted')}; pass rates "
          f"{json.dumps({k: round(v, 2) for k, v in (s16.get('gate_pass_rates') or {}).items()})}"]
    for r in s16.get("accepted") or []:
        L.append(f"- ACCEPTED {r['subcluster']}: n {r['n_cells']}, donors {r['n_donors']}, signature "
                 f"{r.get('rep01_signature', '')}; rep AUC {fmt(r.get('rep01_auc'))}/{fmt(r.get('rep10_auc'))}")
    with open(os.path.join(OUT, "CONTROLS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------------------------
# 17. RESULTS.md
# ----------------------------------------------------------------------------------------------
def fmt(x, d=3):
    if x is None:
        return "NA"
    try:
        if isinstance(x, (int, np.integer)):
            return str(x)
        x = float(x)
        if not np.isfinite(x):
            return "NA"
        if abs(x) < 1e-3 and x != 0:
            return f"{x:.1e}"
        return f"{x:.{d}f}"
    except Exception:
        return str(x)


def gget(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def write_results_md():
    L = ["# ARCANE deep lupus analysis (full Perez cohort)", ""]
    ov = R.get("overview", {})
    L += ["## Data", f"- query cells: {ov.get('n_query_cells')}, query donors: {ov.get('n_donors')} "
          f"(SLE {ov.get('n_sle_donors')}, held-out healthy {ov.get('n_healthy_donors')}); seeds: {ov.get('seeds')}",
          f"- flagged fraction (seed-mean): healthy {fmt(gget(ov, 'flag_rate', 'healthy'))}, SLE {fmt(gget(ov, 'flag_rate', 'SLE'))}",
          f"- columns detected: {json.dumps({k: v for k, v in R.get('columns', {}).items() if k != 'clinical_candidates'})}",
          f"- gene sets: {R.get('geneset_source')}; ISG set `{R.get('isg_key')}`, dissociation set `{R.get('diss_key')}`", ""]
    s3 = R.get("S3", {})
    L.append("## Donor-level association of flags with ISG (Spearman, 95% bootstrap CI over donors)")
    for grp in ["SLE", "healthy_query", "all"]:
        r = gget(s3, "donor_spearman", grp, "flag_vs_isg_typeadj") or {}
        L.append(f"- {grp}: rho {fmt(r.get('rho'))} [{fmt(r.get('ci_lo'))}, {fmt(r.get('ci_hi'))}], "
                 f"p {fmt(r.get('p'))}, n {r.get('n')}")
    for c in ["SLE", "healthy"]:
        r = gget(s3, "within_patient", c, "rho_novelty_isg") or {}
        a = gget(s3, "within_patient", c, "auc_isg_flagged") or {}
        L.append(f"- within-patient (cells within donor x type), {c}: mean rho(novelty, ISG) {fmt(r.get('mean'))} "
                 f"[{fmt(r.get('ci_lo'))}, {fmt(r.get('ci_hi'))}] over {r.get('n')} donors; "
                 f"AUROC of ISG for flagged {fmt(a.get('mean'))} [{fmt(a.get('ci_lo'))}, {fmt(a.get('ci_hi'))}]")
    L.append("")
    L.append("## Adjusted models")
    for nm, m in (gget(s3, "donor_ols") or {}).items():
        if "terms" in m:
            key = [t for t in m["terms"] if t["term"] in ("is_sle", "isg_z", "diss_z", "is_sle:isg_z")]
            L.append(f"- OLS {nm} (n={m['n']}, R2={fmt(m['r2'])}): " +
                     "; ".join(f"{t['term']} {fmt(t['coef'])} (p {fmt(t['p'])})" for t in key))
        else:
            L.append(f"- OLS {nm}: {m.get('error')}")
    mm = gget(s3, "donor_mixedlm_cohort_RE") or {}
    if "terms" in mm:
        L.append("- MixedLM (cohort random intercept): " + "; ".join(
            f"{t['term']} {fmt(t['coef'])} (p {fmt(t['p'])})" for t in mm["terms"] if t["term"] in ("is_sle", "isg_z", "diss_z")))
    for nm, m in (gget(s3, "cell_glm_donor_clustered") or {}).items():
        if "terms" in m:
            key = [t for t in m["terms"] if t["term"] in ("is_sle", "isgt_z", "diss_z", "mito_z", "lc_z")]
            L.append(f"- cell GLM {nm} (cells {m['n_cells']}, donors {m['n_donors']}): " +
                     "; ".join(f"{t['term']} log-OR {fmt(t['coef'])} [{fmt(t['ci_lo'])}, {fmt(t['ci_hi'])}] p {fmt(t['p'])}" for t in key))
    lg = R.get("S3b") or {}
    if lg.get("within_donor_spearman_centered"):
        r = lg["within_donor_spearman_centered"]
        L.append(f"- longitudinal (donors with >=2 samples, n={lg.get('n_donors')}): within-donor rho {fmt(r.get('rho'))} "
                 f"[{fmt(r.get('ci_lo'))}, {fmt(r.get('ci_hi'))}]")
    L.append("")
    s4 = R.get("S4", {})
    L.append("## Technical control: processing cohort")
    for r in gget(s4, "cohort", "table") or []:
        L.append(f"- cohort {r['level']}: healthy-query median {fmt(r.get('healthy_median'))} (n {r['n_healthy_query']}), "
                 f"SLE median {fmt(r.get('sle_median'))} (n {r['n_sle']}), MWU p {fmt(r.get('mwu_p'))}, "
                 f"ref donors {r.get('n_ref_donors', 'NA')}")
    L.append(f"- Kruskal-Wallis across cohorts (healthy query donors): p {fmt(gget(s4, 'cohort', 'kruskal_healthy_query_p'))}; "
             f"across lanes: p {fmt(gget(s4, 'lane', 'kruskal_healthy_query_p'))}")
    L.append("")
    s5 = R.get("S5", {})
    L.append("## Beyond ISG: ISG-independent unmatched (IIU) cells")
    L.append(f"- novelty residualization within type: median R2 {fmt(gget(s5, 'residualization', 'median_r2_full'))}, "
             f"median partial R2 of ISG {fmt(gget(s5, 'residualization', 'median_r2_isg_partial'))}")
    L.append(f"- fraction of SLE flagged cells that are ISG-low (< healthy-reference 75th pct of their type): "
             f"{fmt(s5.get('frac_SLE_flagged_isg_low'))} (healthy: {fmt(s5.get('frac_healthy_flagged_isg_low'))})")
    cr = s5.get("cell_rates", {})
    for k in ["IIU", "IIU_resid", "IIU_flag_isglow"]:
        L.append(f"- {k} cell rate: healthy {fmt(gget(cr, k, 'healthy'))}, SLE {fmt(gget(cr, k, 'SLE'))}; donor-level MWU p "
                 f"{fmt(gget(s5, 'donor_IIU_SLE_vs_healthy', k.lower(), 'p'))}")
    a = s5.get("donor_IIU_associations", {})
    r = a.get("spearman_iiu_vs_isg_SLE", {})
    L.append(f"- SLE donors: IIU fraction vs ISG rho {fmt(r.get('rho'))} [{fmt(r.get('ci_lo'))}, {fmt(r.get('ci_hi'))}]")
    for k, v in a.items():
        if k.startswith("kruskal"):
            L.append(f"- {k}: p {fmt(v['p'])}; medians {', '.join(f'{l}={fmt(m)}' for l, m in v['medians'].items())}")
    for t, r in (s5.get("DE_IIU") or {}).items():
        top = [x[0] for x in gget(r, "SLE", "top_up", default=[])[:12]]
        sets = [f"{x[0]}(q={fmt(x[2], 2)})" for x in gget(r, "SLE", "top_sets", default=[])[:3]]
        L.append(f"- DE IIU vs kept, {t} (SLE): {', '.join(top)} | sets: {', '.join(sets)} | "
                 f"Jaccard with healthy IIU {fmt(r.get('jaccard_up_SLE_vs_healthy'))}; SLE-specific: "
                 f"{', '.join(r.get('SLE_specific_up', [])[:10])}")
    L.append("")
    s6 = R.get("S6", {})
    L.append("## Sub-clusters of flagged cells (U+S latent, Leiden)")
    L.append(f"- types: {s6.get('n_types')}, subclusters: {s6.get('n_subclusters')}, states called (>= "
             f"{args.min_donors_state} donors with >= {args.min_cells_per_donor_state} cells, >=50 cells): "
             f"{s6.get('n_states_called')}, SLE-enriched (donor-permutation q<0.05): {s6.get('states_sle_enriched_q05')}")
    st = sorted(s6.get("states", []), key=lambda r: (r.get("perm_q_vs_type") or 1))[:20]
    for r in st:
        L.append(f"- {r['subcluster']}: n {r['n_cells']}, donors {r['n_donors_ge_min']}, SLE {fmt(r['frac_sle'], 2)} "
                 f"(exp {fmt(r['exp_frac_sle_type'], 2)}), perm q {fmt(r.get('perm_q_vs_type'))}, ISG-low "
                 f"{fmt(r.get('frac_isg_low'), 2)}, closest {r.get('closest_label')}, markers {r.get('top_markers')}; "
                 f"{r.get('top_sets')}")
    L.append("")
    s7 = R.get("S7", {})
    L.append("## Dissociation control")
    for c in ["healthy", "SLE"]:
        r = gget(s7, "diss_flagged_vs_kept", c) or {}
        L.append(f"- {c}: dissociation score flagged vs kept, Cohen d {fmt(r.get('cohen_d'))}, AUC {fmt(r.get('auc_a_gt_b'))}")
    for nm, d in (s7.get("IEG_matched_on_core_dissociation") or {}).items():
        items = [f"{k[2:]} {fmt(v['matched_diff'])} [{fmt(v['ci_lo'])}, {fmt(v['ci_hi'])}]"
                 for k, v in d.items() if k.startswith("g_")]
        L.append(f"- {nm} (matched on donor x type x core-dissociation decile; {d.get('_n')}): " + "; ".join(items))
    ds = s7.get("diss_standardization") or {}
    L.append(f"- SLE/healthy flag-rate ratio per type (median): raw {fmt(ds.get('median_ratio_raw'))}, "
             f"dissociation-standardized {fmt(ds.get('median_ratio_std'))}")
    L.append("")
    s8 = R.get("S8", {})
    L.append("## ARCANE vs ISG score alone")
    L.append(f"- donor AUROC (SLE vs held-out healthy): " + ", ".join(f"{k} {fmt(v)}" for k, v in (s8.get('donor_auc') or {}).items()))
    L.append(f"- CV AUROC: " + ", ".join(f"{k} {fmt(v['cv_auc_mean'])}" for k, v in (s8.get('donor_cv_auc') or {}).items()))
    for k, v in (s8.get("donor_LRT") or {}).items():
        if not isinstance(v, dict):
            v = {"error": v}
        L.append(f"- LRT {k}: LR {fmt(v.get('lr'))}, p {fmt(v.get('p'))}, coef {fmt(v.get('coef'))}" if "lr" in v else f"- LRT {k}: {v.get('error')}")
    q = s8.get("donor_quadrants_SLE", {})
    L.append(f"- SLE donors above healthy q90 of flag fraction but not ISG: {q.get('n_high_flag_low_isg')} / {q.get('n_sle')}; "
             f"ISG-high but flag-low: {q.get('n_high_isg_low_flag')}; both high: {q.get('n_both_high')}")
    L.append(f"- cell-level median AUROC across types (SLE vs healthy cells): {json.dumps({k: round(v, 3) for k, v in (s8.get('cell_auc_by_type_median') or {}).items() if v == v})}")
    L.append("")
    s9 = R.get("S9", {})
    L.append("## Composition")
    L.append(f"- ARCANE-confident vs author composition Pearson r: {fmt(s9.get('confident_vs_author_composition_pearson'))}")
    for r in (s9.get("sig_q05") or [])[:15]:
        L.append(f"- {r['table']} {r['type']}: SLE {fmt(r['mean_sle'])} vs healthy {fmt(r['mean_healthy'])}, "
                 f"log2FC {fmt(r['log2fc'], 2)}, q {fmt(r['q'])}")
    L.append("")
    if R["errors"]:
        L.append("## Errors")
        for k, v in R["errors"].items():
            L.append(f"- {k}: {v}")
    with open(os.path.join(OUT, "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
A_OBS_COLS = []
DET = {}


def main():
    import anndata as ad
    global A_OBS_COLS, DET
    per = load_res()
    s0 = args.seed0
    R["overview"] = {"seeds": sorted(per)}
    h5 = args.h5ad or os.path.join(ROOT, "data", "std", f"{args.tag}.h5ad")
    log("reading", h5)
    A = ad.read_h5ad(h5)
    A.obs_names = A.obs_names.astype(str)
    if not A.obs_names.is_unique:
        raise ValueError("obs_names not unique")
    A_OBS_COLS = list(A.obs.columns)
    det = detect_columns(A.obs) or {}
    DET = det
    gs, isg_key, diss_key, src = load_genesets()
    R.update(geneset_source=src, isg_key=isg_key, diss_key=diss_key)
    used = expression_scores(A, gs)
    if used is None:
        raise RuntimeError("expression scoring failed; cannot continue")
    isg_col, diss_col = "gs_" + safe_name(isg_key), "gs_" + safe_name(diss_key)
    D0 = per[s0][0]
    meta = donor_table_meta(A.obs, det, D0.donor.unique())
    wcsv(meta.reset_index(), "donor_metadata.csv")
    D = build_cells(per, A, det, meta)
    # type-adjusted ISG: z relative to healthy reference cells of the same fine type
    role_col = det.get("role")
    lab_col = det.get("label")
    if role_col and lab_col:
        ref = A.obs[A.obs[role_col].astype(str) == "ref"]
        ref_mu = ref.groupby(lab_col)[isg_col].mean()
        ref_sd = ref.groupby(lab_col)[isg_col].std()
        ref_q75 = ref.groupby(lab_col)[isg_col].quantile(0.75)
        log(f"reference cells for ISG baselines: {len(ref)}")
    else:
        log("WARNING: no role/label columns; using held-out healthy query cells for ISG baselines")
        h = D[D.is_sle == 0]
        ref_mu, ref_sd, ref_q75 = h.groupby("y")[isg_col].mean(), h.groupby("y")[isg_col].std(), \
            h.groupby("y")[isg_col].quantile(0.75)
    ref_mu.index, ref_sd.index, ref_q75.index = [x.astype(str) for x in (ref_mu.index, ref_sd.index, ref_q75.index)]
    D["isg_tz"] = (D[isg_col] - D.y.map(ref_mu).astype(float)) / D.y.map(ref_sd).astype(float).clip(lower=1e-3)
    D["isg_tz"] = D.isg_tz.fillna((D[isg_col] - D[isg_col].mean()) / D[isg_col].std())
    D["isg_low"] = D[isg_col].values < D.y.map(ref_q75).astype(float).fillna(np.inf).values
    # donor table
    agg = dict(cond=("cond", "first"), is_sle=("is_sle", "first"), n_cells=("flag", "size"),
               flag_frac=("flag_mean", "mean"), flag_frac_seed0=("flag", "mean"), novelty_mean=("novelty_mean", "mean"),
               isg=("isg_tz", "mean"), isg_raw=(isg_col, "mean"), diss=(diss_col, "mean"),
               frac_isg_high=("isg_low", lambda v: 1 - v.mean()))
    for s in sorted(per):
        agg[f"flag_frac_s{s}"] = (f"flag_s{s}", "mean")
    DN = D.groupby("donor").agg(**agg)
    DN = DN.join(meta)
    DN["flag_logit"] = logit_frac(DN.flag_frac * DN.n_cells, DN.n_cells)
    ov = R["overview"]
    ov.update(n_query_cells=int(len(D)), n_donors=int(len(DN)), n_sle_donors=int(DN.is_sle.sum()),
              n_healthy_donors=int((DN.is_sle == 0).sum()),
              flag_rate=D.groupby("cond").flag_mean.mean().to_dict(),
              flag_rate_seed0=D.groupby("cond").flag.mean().to_dict(),
              flag_rate_donor_median=DN.groupby("cond").flag_frac.median().to_dict(),
              n_fine_types=int(D.y.nunique()), fine_types=sorted(D.y.unique().tolist()))
    log("overview", json.dumps(clean(ov), default=jsonable)[:2000])
    save_results()

    donor_models(D, DN, isg_col, diss_col)
    longitudinal(D, A.obs, det, isg_col)
    cohort_control(DN, A.obs, det)
    beyond_isg(D, DN, A, gs, isg_col, diss_col, ref_q75)
    if "IIU" not in D:
        D["IIU"] = False
        D["nov_resid"] = np.nan
    subcluster(D, A, gs, isg_col)
    dissociation(D, diss_col)
    arcane_vs_isg(D, DN, isg_col)
    composition(D, DN)
    if not args.skip_controls:
        def find_key(pat):
            for k in gs:
                if re.search(pat, k, re.I) and "gs_" + safe_name(k) in D:
                    return "gs_" + safe_name(k)
            return None
        gc_col = find_key(r"glucocorticoid")
        ifng_col = find_key(r"ifn.?g|gamma")
        R["control_columns"] = dict(gc=gc_col, ifng=ifng_col)
        if gc_col:
            DN["gc"] = D.groupby("donor")[gc_col].mean().reindex(DN.index)
        seed_and_composition(D, DN, isg_col)
        doublet_ambient_qc(D, A.obs, det)
        isg_residual_confounding(D, DN, isg_col, diss_col, gc_col, ifng_col)
        view_disagreement(D, DN, isg_col)
        matched_strata(D, DN, A.obs, det, gc_col)
        if not args.skip_de:
            pseudobulk_de(D, A, gs)
        state_gates(D, DN, A, A.obs, det, gs, isg_key, diss_key, isg_col, diss_col, gc_col)
        try:
            write_controls_md()
        except Exception as e:
            R["errors"]["controls_md"] = repr(e)
            log(traceback.format_exc())
    try:
        wcsv(DN.reset_index(), "donor_table.csv")
        keep = [c for c in ["y", "parent", "cond", "donor", "flag", "flag_mean", "novelty_mean", "nov_resid",
                            "IIU", "IIU_resid", "IIU_flag_isglow", "isg_low", "isg_tz", "subcluster", "pred_ARCANE",
                            "closest_label", "conf_pmax", "total_counts", "n_genes", "pct_mito", "lineage", "foreign",
                            "doublet_like", "artifact_like", "plt_hi", "hb_hi", "ig_hi_nonB", "scrublet",
                            "nov_resid_strict", "IIU_strict", "dU", "dS", "U_only", "S_only", "US_both"] if c in D] + \
               [c for c in D.columns if c.startswith("gs_")]
        D[keep].to_csv(os.path.join(OUT, "cells_seed0.csv.gz"), compression="gzip")
    except Exception as e:
        R["errors"]["write_tables"] = repr(e)
    try:
        write_results_md()
    except Exception as e:
        R["errors"]["results_md"] = repr(e)
        log(traceback.format_exc())
    save_results()
    log(f"done in {time.time() - T0:.0f}s; outputs in {OUT}; errors: {list(R['errors'])}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        R["errors"]["main"] = repr(e)
        log(traceback.format_exc())
        save_results()
        sys.exit(1)
