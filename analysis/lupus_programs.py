#!/usr/bin/env python
"""Donor-level programs of reference-unmatched (ARCANE-flagged) states in SLE that are NOT the ISG program.

Run from ~/arcane:
    python lupus_programs.py [--root .] [--tag lupusfull] [--seeds 0,1,2] [--out summary/lupusprog]

Inputs (relative to --root):
  data/std/<tag>.h5ad                raw counts (X), obs: label, parent, batch (donor), role, condition, Processing_Cohort,
                                     self_reported_ethnicity, sex, development_stage, disease_state, ...
  res2/<tag>_seed<s>_cells.npz       per query cell: names, y, cond, batch, p_unknown_ctrl, p_unknown_bonf, ...
  genesets.json                      {set: [genes]}
  emb/<tag>_isgblind300_seed0_full_dropped.txt (optional; extra ISG-correlated genes to exclude)

Steps
  1  strict unmatched flag (p_unknown_ctrl<=a and p_unknown_bonf<=a; majority of seeds); donor x fine-type flag rates;
     donor x type x {flagged, kept} pseudobulk (mean CP10k) and donor gene-set scores (ISG, GC, dissociation, ...)
  2  per fine type: donor-paired flagged-vs-kept DE (signed-rank over donors, BH) in SLE and healthy-query donors
     separately + SLE-vs-healthy interaction; curated ISG/IFN/IEG/HSP/ribo/mito/Ig/Hb/platelet/sex genes and genes
     correlated with donor ISG are removed from the program features
  3  cNMF-like consensus NMF of the SLE donor x (type, gene, sign) matrix of shrunken flagged-vs-kept log2FC residualized
     on donor ISG, cohort, ancestry, age, sex; k chosen by consensus stability (silhouette), column-permutation null
  4  per program: top genes, lineage shares, disease_state, SLE vs healthy-query (cohort-stratified), correlation with
     ISG / GC / dissociation / heat shock, cohort-stratified split-half and leave-one-cohort-out replication, gates
  5  hypergeometric enrichment of program top genes in every genesets.json set
Outputs: <out>/RESULTS.md, results.json, csv tables, PNG figures, log.txt.  Sections never crash the run.
Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, anndata (only read_h5ad, .X, .obs, .var_names, .var).
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tag", default="lupusfull")
    ap.add_argument("--h5ad", default=None)
    ap.add_argument("--res_dir", default="res2")
    ap.add_argument("--seeds", default="0,1,2", help="seeds to use; missing files are skipped")
    ap.add_argument("--seed_rule", default="majority", choices=["majority", "all", "any", "first"])
    ap.add_argument("--alpha", type=float, default=0.05, help="threshold on p_unknown_ctrl and p_unknown_bonf")
    ap.add_argument("--genesets", default="genesets.json")
    ap.add_argument("--isg_key", default="ISG_typeI")
    ap.add_argument("--extra_exclude", default=None,
                    help="file with genes to exclude (default: emb/<tag>_isgblind300_seed0_full_dropped.txt if present)")
    ap.add_argument("--out", default="summary/lupusprog")
    ap.add_argument("--min_cells_gene", type=int, default=50)
    ap.add_argument("--min_flag", type=int, default=5, help="min flagged cells in a donor x type")
    ap.add_argument("--min_kept", type=int, default=20, help="min kept cells in a donor x type")
    ap.add_argument("--min_cells_score", type=int, default=20, help="min cells in a donor x type for set scores")
    ap.add_argument("--pc", type=float, default=0.1, help="pseudo-count (CP10k units) for log2FC")
    ap.add_argument("--shrink_k0", type=float, default=10.0, help="LFC shrink weight nf/(nf+k0)")
    ap.add_argument("--min_mean", type=float, default=0.2, help="min mean CP10k of a gene in a type")
    ap.add_argument("--min_donors_de", type=int, default=8)
    ap.add_argument("--isg_rho", type=float, default=0.3, help="exclude genes with |rho| vs donor ISG >= this")
    ap.add_argument("--isg_top", type=int, default=300, help="also exclude this many most ISG-correlated genes")
    ap.add_argument("--min_type_cov", type=float, default=0.4,
                    help="type enters NMF if >= this fraction of SLE donors have a valid LFC")
    ap.add_argument("--genes_per_type", type=int, default=300)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--n_rep", type=int, default=30, help="NMF restarts per k (donor-subsampled)")
    ap.add_argument("--sub_frac", type=float, default=0.8)
    ap.add_argument("--n_splits", type=int, default=20, help="cohort-stratified split-half refits")
    ap.add_argument("--n_rep_split", type=int, default=10)
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--top_n_enrich", type=int, default=50)
    ap.add_argument("--regress_extra", default="", help="comma list of extra donor covariates to regress: gc,diss")
    ap.add_argument("--max_cells", type=int, default=0, help="optional cap on query cells (0 = all)")
    return ap.parse_args(argv)


ARGS = None
OUT = None
LOGF = None
R = {"errors": {}, "timing_s": {}}
T0 = time.time()
RNG = np.random.default_rng(0)


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
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Series,)):
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


def signed_rank(L):
    """Vectorized Wilcoxon signed-rank (normal approximation; zeros split half/half) per column of L (n x G)."""
    L = np.asarray(L, float)
    n = L.shape[0]
    if n < 2:
        return np.full(L.shape[1], np.nan), np.full(L.shape[1], np.nan)
    r = stats.rankdata(np.abs(L), axis=0)
    pos = L > 0
    zer = L == 0
    wp = (r * pos).sum(0) + 0.5 * (r * zer).sum(0)
    mu = n * (n + 1) / 4.0
    t0 = zer.sum(0).astype(float)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - (t0 ** 3 - t0) / 48.0
    z = np.where(var > 0, (wp - mu) / np.sqrt(np.maximum(var, 1e-12)), 0.0)
    p = 2 * stats.norm.sf(np.abs(z))
    p[t0 == n] = 1.0
    return z, p


def rank_cols(M):
    return stats.rankdata(M, axis=0)


def spearman_vec(x, M):
    """Spearman of vector x (n) with each column of M (n x G); NaN-free inputs."""
    rx = stats.rankdata(x)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    RM = rank_cols(M)
    RM = (RM - RM.mean(0)) / (RM.std(0) + 1e-12)
    return (rx[:, None] * RM).mean(0)


def spearman_ci(x, y, nboot=None, rng=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 6 or np.std(x) == 0 or np.std(y) == 0:
        return dict(rho=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=int(len(x)))
    rho, p = stats.spearmanr(x, y)
    nb = nboot or ARGS.nboot
    rng = rng or np.random.default_rng(1)
    bs = []
    for _ in range(nb):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            bs.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    return dict(rho=float(rho), lo=float(lo), hi=float(hi), p=float(p), n=int(len(x)))


def auroc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = stats.mannwhitneyu(a, b, alternative="two-sided")[0]
    return float(u / (len(a) * len(b)))


def auroc_ci(a, b, nboot=None, rng=None):
    rng = rng or np.random.default_rng(2)
    a, b = np.asarray(a, float), np.asarray(b, float)
    est = auroc(a, b)
    if not np.isfinite(est):
        return dict(auc=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    bs = [auroc(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(nboot or ARGS.nboot)]
    p = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(auc=est, lo=float(lo), hi=float(hi), p=float(p), n1=len(a), n0=len(b))


def strat_perm(y, g, strata, nperm=None, seed=0):
    """Stratified permutation test of mean(y | g=1) - mean(y | g=0) within strata containing both groups."""
    y, g, strata = np.asarray(y, float), np.asarray(g, int), np.asarray(strata).astype(str)
    ok = np.zeros(len(y), bool)
    for s in np.unique(strata):
        m = strata == s
        if g[m].min() != g[m].max():
            ok |= m
    if ok.sum() < 6:
        return dict(stat=np.nan, p=np.nan, n_matched=int(ok.sum()))
    y, g, strata = y[ok], g[ok], strata[ok]
    lv = np.unique(strata)
    idx = [np.where(strata == s)[0] for s in lv]
    w = np.array([len(i) for i in idx], float)

    def stat(gg):
        d = [y[i][gg[i] == 1].mean() - y[i][gg[i] == 0].mean() for i in idx]
        return float(np.sum(w * np.array(d)) / w.sum())
    s0 = stat(g)
    rng = np.random.default_rng(seed)
    cnt = 0
    nperm = nperm or ARGS.nperm
    for _ in range(nperm):
        gg = g.copy()
        for i in idx:
            gg[i] = rng.permutation(gg[i])
        cnt += abs(stat(gg)) >= abs(s0) - 1e-12
    return dict(stat=s0, p=(cnt + 1) / (nperm + 1), n_matched=int(ok.sum()), n_strata=int(len(lv)))


def kruskal_groups(v, grp, min_n=3):
    v = np.asarray(v, float)
    grp = np.asarray(grp).astype(str)
    gs = [v[(grp == g) & np.isfinite(v)] for g in np.unique(grp)]
    lv = [g for g in np.unique(grp)]
    keep = [(l, x) for l, x in zip(lv, gs) if len(x) >= min_n]
    if len(keep) < 2:
        return dict(p=np.nan, eta2=np.nan, medians={l: float(np.median(x)) for l, x in zip(lv, gs) if len(x)})
    h, p = stats.kruskal(*[x for _, x in keep])
    n = sum(len(x) for _, x in keep)
    eta2 = (h - len(keep) + 1) / max(n - len(keep), 1)
    return dict(p=float(p), H=float(h), eta2=float(eta2), medians={l: float(np.median(x)) for l, x in keep},
                ns={l: int(len(x)) for l, x in keep})


def collapse_levels(s, min_n=3, na="unknown"):
    s = pd.Series(s).astype(object)
    s = s.where(s.notna(), na).astype(str)
    s = s.replace({"nan": na, "None": na, "": na})
    vc = s.value_counts()
    rare = vc[vc < min_n].index
    return s.where(~s.isin(rare), "other")


def tucker(A, B):
    """Cosine congruence matrix between rows of A and rows of B."""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def match_programs(Href, H):
    from scipy.optimize import linear_sum_assignment
    C = tucker(Href, H)
    r, c = linear_sum_assignment(-C)
    out = np.full(Href.shape[0], np.nan)
    out[r] = C[r, c]
    perm = np.full(Href.shape[0], -1)
    perm[r] = c
    return out, perm


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


# ----------------------------------------------------------------------------------------------
# column detection (reused from lupus_deep.py)
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


def lineage_of(parent, fine):
    s = f"{parent} {fine}".lower()
    rules = [("B", r"\bb[_ ]|^b|bnaive|bmem|batyp|plasma|\bpb\b|b cell|b_"), ("T", r"\bt[48_ ]|t cell|cd4|cd8|mait|treg|"
             r"gdt|\btreg|t4|t8|^t"), ("NK", r"\bnk|natural killer"), ("Mono", r"mono|cm\b|ncm|^cm|^ncm|macro"),
             ("DC", r"dc\b|dendritic|pdc|cdc"), ("Prolif", r"prolif|cycling"), ("Progen", r"progen|hsc|stem")]
    for lab, pat in rules:
        if re.search(pat, s):
            return lab
    return str(parent)


# ----------------------------------------------------------------------------------------------
# 0. load data
# ----------------------------------------------------------------------------------------------
@section("0_load_cells")
def load_cells():
    seeds = [int(s) for s in ARGS.seeds.split(",") if s.strip() != ""]
    per = {}
    for s in seeds:
        f = os.path.join(ARGS.root, ARGS.res_dir, f"{ARGS.tag}_seed{s}_cells.npz")
        if not os.path.exists(f):
            log("missing", f)
            continue
        C = np.load(f, allow_pickle=True)
        classes = np.asarray(C["classes"]).astype(str) if "classes" in C.files else None

        def lab(v):
            v = np.asarray(v)
            if v.dtype.kind in "iu" and classes is not None:
                return np.where(v >= 0, classes[np.clip(v, 0, len(classes) - 1)], "none")
            return v.astype(str)
        df = pd.DataFrame(index=pd.Index(np.asarray(C["names"]).astype(str), name="cell"))
        df["y"] = lab(C["y"])
        df["cond_raw"] = np.asarray(C["cond"]).astype(str)
        df["donor"] = np.asarray(C["batch"]).astype(str)
        if "p_unknown_ctrl" in C.files and "p_unknown_bonf" in C.files:
            pc, pb = np.asarray(C["p_unknown_ctrl"], float), np.asarray(C["p_unknown_bonf"], float)
            df["flag"] = (pc <= ARGS.alpha) & (pb <= ARGS.alpha)
            df["p_ctrl"], df["p_bonf"] = pc, pb
            rule = "ctrl&bonf"
        else:
            df["flag"] = np.asarray(C["unknown"]).astype(bool)
            rule = "unknown (fallback: p_unknown_ctrl/p_unknown_bonf missing)"
        per[s] = df
        log(f"seed {s}: {len(df)} cells; strict flag rate {df.flag.mean():.4f} ({rule})")
        R.setdefault("flag_rule", {})[s] = rule
    if not per:
        raise FileNotFoundError("no cells npz found")
    s0 = sorted(per)[0]
    D = per[s0].copy()
    common = D.index
    for df in per.values():
        common = common.intersection(df.index)
    D = D.loc[common]
    F = np.vstack([per[s].loc[common, "flag"].values for s in sorted(per)]).astype(float)
    frac = F.mean(0)
    rule = ARGS.seed_rule
    D["flag"] = {"majority": frac > 0.5, "all": frac == 1, "any": frac > 0, "first": F[0] > 0}[rule]
    D["flag_seedfrac"] = frac
    if len(per) > 1:
        jac = []
        ks = sorted(per)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = F[i] > 0, F[j] > 0
                jac.append(float((a & b).sum() / max((a | b).sum(), 1)))
        R["seed_jaccard"] = jac
    D["is_sle"] = D.cond_raw.map(is_sle_str).astype(int)
    if ARGS.max_cells and len(D) > ARGS.max_cells:
        D = D.sample(ARGS.max_cells, random_state=0)
    R["overview"] = dict(seeds=sorted(per), seed_rule=rule, n_cells=int(len(D)), flag_rate=float(D.flag.mean()),
                         flag_rate_by_cond=D.groupby("is_sle").flag.mean().rename({0: "healthy", 1: "SLE"}).to_dict())
    log(R["overview"])
    return D


@section("0_load_expression")
def load_expression(D):
    import anndata as ad
    h5 = ARGS.h5ad or os.path.join(ARGS.root, "data", "std", f"{ARGS.tag}.h5ad")
    log("reading", h5)
    A = ad.read_h5ad(h5)
    obs = A.obs.copy()
    obs.index = obs.index.astype(str)
    var = pd.Index(np.asarray(A.var_names).astype(str))
    try:
        if var.str.startswith("ENSG").mean() > 0.5:
            for c in ["feature_name", "gene_symbols", "gene_name", "symbol"]:
                if c in A.var.columns:
                    var = pd.Index(A.var[c].astype(str).values)
                    log("using var column", c, "as gene symbols")
                    break
    except Exception:  # noqa: BLE001
        pass
    if obs.index.is_unique and D.index.is_unique:
        pos = obs.index.get_indexer(D.index)
    else:
        # non-unique names: npz cells are in the order of the h5ad query cells (CONTEXT); match positionally
        rc = find_col(list(obs.columns), ["role"])
        qpos = np.where(obs[rc].astype(str).values == "query")[0] if rc else np.arange(len(obs))
        log("WARNING: non-unique obs_names/npz names; matching npz to h5ad query cells by position")
        if len(qpos) != len(D) or not (obs.index.values[qpos].astype(str) == D.index.values).all():
            raise ValueError("cannot align npz cells to h5ad: non-unique names and positional order differs")
        pos = qpos
    if (pos < 0).any():
        log(f"WARNING: {(pos < 0).sum()} query cells not in h5ad; dropped")
        R["n_cells_not_in_h5ad"] = int((pos < 0).sum())
    keep = pos >= 0
    D = D.loc[keep]
    pos = pos[keep]
    obs = obs.iloc[pos]
    # join sanity: donor id in npz must equal the h5ad batch column for the same cell
    bc = find_col(list(obs.columns), ["batch"])
    if bc:
        mis = (obs[bc].astype(str).values != D.donor.values)
        R["join_donor_mismatch"] = int(mis.sum())
        if mis.mean() > 0.001:
            raise ValueError(f"npz/h5ad join broken: {mis.sum()} cells with different donor (batch) id")
    X = A.X
    if sp.issparse(X):
        X = sp.csr_matrix(X)  # no copy if already CSR
        X = X[pos]
    else:  # dense X: convert the selected rows in chunks (never densify all cells x genes twice)
        ch = [sp.csr_matrix(np.asarray(X[pos[i:i + 20000]], dtype=np.float32)) for i in range(0, len(pos), 20000)]
        X = sp.vstack(ch).tocsr()
        del ch
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    del A
    samp = X.data[: min(len(X.data), 200000)]
    if len(samp) and not np.allclose(samp, np.round(samp)):
        log("WARNING: X does not look like raw counts; proceeding (treated as counts)")
    # dedupe gene symbols
    if not var.is_unique:
        u, inv = np.unique(np.asarray(var), return_inverse=True)
        M = sp.csr_matrix((np.ones(len(var), np.float32), (np.arange(len(var)), inv)), shape=(len(var), len(u)))
        X = (X @ M).tocsr()
        var = pd.Index(u)
        log("summed duplicated gene symbols ->", len(var))
    tot = np.asarray(X.sum(1)).ravel()
    X.data *= np.repeat((1e4 / np.maximum(tot, 1)).astype(np.float32), np.diff(X.indptr))  # CP10k
    det = np.bincount(X.indices, minlength=X.shape[1])
    gk = det >= ARGS.min_cells_gene
    X = X[:, np.where(gk)[0]].tocsr()
    var = var[gk]
    log(f"cells {X.shape[0]}, genes kept {X.shape[1]}")
    return D, X, var, obs


def detect_meta(obs, D):
    cols = list(obs.columns)
    det = dict(
        cohort=find_col(cols, ["Processing_Cohort", "processing_cohort", "cohort"], [r"process.*cohort", r"cohort"]),
        ancestry=find_col(cols, ["self_reported_ethnicity", "pop_cov", "ancestry", "ethnicity", "race"],
                          [r"ethnic", r"ancestr", r"pop_cov", r"race"]),
        sex=find_col(cols, ["sex", "Sex", "sex_cov", "gender"], [r"^sex", r"gender"]),
        age=find_col(cols, ["Age", "age", "age_cov", "age_years"], [r"^age"]),
        dev_stage=find_col(cols, ["development_stage"], [r"development_stage"]),
        status=find_col(cols, ["disease_state", "Status", "status"], [r"disease_state", r"^status", r"flare"]),
        parent=find_col(cols, ["parent"]),
        individual=find_col(cols, ["donor_id", "ind_cov", "individual"], [r"^donor_id$", r"ind_cov$"]),
    )
    R["columns"] = det
    o = obs.copy()
    o["_donor"] = D.donor.values
    meta = pd.DataFrame(index=pd.Index(sorted(D.donor.unique()), name="donor"))
    nonconst = {}
    for key in ["cohort", "ancestry", "sex", "age", "dev_stage", "status"]:
        c = det.get(key)
        if c is None or c not in o.columns:
            continue
        g = o.groupby("_donor")[c]
        nonconst[key] = int((g.nunique() > 1).sum())
        meta[key] = g.agg(lambda s: s.astype(str).value_counts().index[0]).reindex(meta.index)
    R["donor_nonconstant"] = nonconst
    # donors (units) whose cells span >1 disease_state are excluded from disease_state tests (pooled states)
    if det.get("status") in o.columns:
        meta["status_mixed"] = (o.groupby("_donor")[det["status"]].nunique() > 1).reindex(meta.index).fillna(False)
    # if one individual contributes several units (e.g. flare + treated samples), keep them together
    ic = det.get("individual")
    if ic and ic in o.columns:
        meta["individual"] = o.groupby("_donor")[ic].agg(
            lambda s: s.astype(str).value_counts().index[0]).reindex(meta.index).astype(str)
    else:
        meta["individual"] = meta.index.astype(str)
    R["n_units_per_individual_gt1"] = int((meta.individual.value_counts() > 1).sum())
    if "age" in meta:
        meta["age_num"] = meta["age"].map(parse_age)
    elif "dev_stage" in meta:
        meta["age_num"] = meta["dev_stage"].map(parse_age)
    else:
        meta["age_num"] = np.nan
    meta["is_sle"] = D.groupby("donor").is_sle.agg(lambda v: int(np.round(v.mean()))).reindex(meta.index)
    if det.get("parent") and det["parent"] in obs:
        par = pd.Series(obs[det["parent"]].astype(str).values, index=D.index)
    else:
        par = D.y
    tp = pd.DataFrame({"y": D.y.values, "parent": par.values}).groupby("y").parent.agg(
        lambda s: s.value_counts().index[0])
    lineage = {t: lineage_of(tp.get(t, t), t) for t in sorted(D.y.unique())}
    log("lineages:", lineage)
    log("columns:", det, "nonconstant within donor:", nonconst)
    return meta, lineage


# ----------------------------------------------------------------------------------------------
# 1. flag rates, pseudobulk, donor scores
# ----------------------------------------------------------------------------------------------
@section("1_pseudobulk")
def pseudobulk(D, X, meta):
    donors = list(meta.index)
    types = sorted(D.y.unique())
    di = pd.Index(donors).get_indexer(D.donor.values)
    ti = pd.Index(types).get_indexer(D.y.values)
    fi = D.flag.values.astype(int)
    nD, nT, G = len(donors), len(types), X.shape[1]
    code = (di * nT + ti) * 2 + fi
    ng = nD * nT * 2
    Ind = sp.csr_matrix((np.ones(len(code), np.float32), (code, np.arange(len(code)))), shape=(ng, len(code)))
    S = np.asarray((Ind @ X).todense(), dtype=np.float32) if G * ng < 2e9 else None
    if S is None:
        raise MemoryError("pseudobulk too large")
    N = np.bincount(code, minlength=ng).astype(float)
    S = S.reshape(nD, nT, 2, G)
    N = N.reshape(nD, nT, 2)
    PB = dict(donors=donors, types=types, nkept=N[:, :, 0], nflag=N[:, :, 1],
              Mk=S[:, :, 0] / np.maximum(N[:, :, 0], 1)[..., None],
              Mf=S[:, :, 1] / np.maximum(N[:, :, 1], 1)[..., None],
              Ma=(S[:, :, 0] + S[:, :, 1]) / np.maximum(N.sum(2), 1)[..., None])
    del S
    # flag rates
    rows = []
    for a, d in enumerate(donors):
        for b, t in enumerate(types):
            n = N[a, b].sum()
            if n > 0:
                rows.append(dict(donor=d, type=t, n_cells=int(n), n_flag=int(N[a, b, 1]), rate=N[a, b, 1] / n,
                                 is_sle=int(meta.loc[d, "is_sle"])))
    RT = pd.DataFrame(rows)
    wcsv(RT, "flag_rates_donor_type.csv")
    summ = []
    for t, g in RT[RT.n_cells >= ARGS.min_kept].groupby("type"):
        a, h = g[g.is_sle == 1].rate.values, g[g.is_sle == 0].rate.values
        summ.append(dict(type=t, n_sle=len(a), n_h=len(h), med_sle=np.median(a) if len(a) else np.nan,
                         med_h=np.median(h) if len(h) else np.nan,
                         auc=auroc(a, h), p=stats.mannwhitneyu(a, h)[1] if len(a) > 2 and len(h) > 2 else np.nan,
                         n_valid_sle=int(((g.is_sle == 1) & (g.n_flag >= ARGS.min_flag)).sum()),
                         n_valid_h=int(((g.is_sle == 0) & (g.n_flag >= ARGS.min_flag)).sum())))
    ST = pd.DataFrame(summ)
    if len(ST):
        ST["q"] = bh(ST.p.values)
    wcsv(ST, "flag_rates_by_type.csv")
    dn = RT.groupby("donor").agg(n=("n_cells", "sum"), k=("n_flag", "sum"))
    meta["flag_rate"] = (dn.k / dn.n).reindex(meta.index)
    meta["n_flag_total"] = dn.k.reindex(meta.index)
    R["step1"] = dict(n_donors=nD, n_sle=int(meta.is_sle.sum()), n_healthy=int((meta.is_sle == 0).sum()),
                      n_types=nT, donor_flag_rate_median={
                          "SLE": float(meta.loc[meta.is_sle == 1, "flag_rate"].median()),
                          "healthy": float(meta.loc[meta.is_sle == 0, "flag_rate"].median())},
                      flag_rate_auc=auroc_ci(meta.loc[meta.is_sle == 1, "flag_rate"].values,
                                             meta.loc[meta.is_sle == 0, "flag_rate"].values),
                      by_type=ST.to_dict(orient="records"))
    return PB


def load_genesets():
    p = ARGS.genesets if os.path.isabs(ARGS.genesets) else os.path.join(ARGS.root, ARGS.genesets)
    gs = json.load(open(p))
    gs = {str(k): [str(g) for g in v] for k, v in gs.items() if isinstance(v, (list, tuple))}
    log("gene sets:", len(gs), "from", p)
    return gs


def pick(gs, *pats):
    for p in pats:
        for k in gs:
            if re.search(p, k, re.I):
                return k
    return None


@section("1_donor_scores")
def donor_scores(PB, var, meta, gs):
    """Composition-adjusted donor gene-set scores: per type z vs healthy-query donors, averaged over genes and types."""
    L = np.log2(PB["Ma"] + ARGS.pc)  # nD x nT x G
    ntot = PB["nkept"] + PB["nflag"]
    ok = ntot >= ARGS.min_cells_score
    healthy = (meta.is_sle.values == 0)
    ref = healthy if healthy.sum() >= 5 else np.ones_like(healthy)
    Z = np.full(L.shape, np.nan, np.float32)
    for b in range(L.shape[1]):
        m = ok[:, b] & ref
        if m.sum() < 3:
            continue
        mu = L[m, b].mean(0)
        sd = np.maximum(L[m, b].std(0), 0.1)
        Z[ok[:, b], b] = (L[ok[:, b], b] - mu) / sd
    PB["Z"] = Z
    PB["L"] = L
    vi = pd.Index(var)
    scores = {}
    for name, genes in gs.items():
        gi = vi.get_indexer([g for g in genes if g in vi])
        gi = gi[gi >= 0]
        if len(gi) < 3:
            continue
        s = np.nanmean(Z[:, :, gi], axis=2)  # nD x nT
        scores[name] = np.nanmean(s, axis=1)
    SC = pd.DataFrame(scores, index=meta.index)
    isg_k = ARGS.isg_key if ARGS.isg_key in SC else pick(SC, r"^ISG", r"IFN.?alpha|type.?I")
    keys = dict(isg=isg_k, gc=pick(SC, r"glucocorticoid_lymphoid", r"glucocorticoid"),
                gc_core=pick(SC, r"glucocorticoid_core"), diss=pick(SC, r"dissoc", r"IEG"),
                hsp=pick(SC, r"heat"), ifng=pick(SC, r"IFN_gamma", r"gamma"))
    for k, v in keys.items():
        meta[k] = SC[v] if v else np.nan
    R["score_keys"] = keys
    wcsv(SC.reset_index(), "donor_geneset_scores.csv")
    wcsv(meta.reset_index(), "donor_table.csv")
    log("score keys", keys)
    log(meta.groupby("is_sle")[["isg", "gc", "diss"]].median())
    return SC


# ----------------------------------------------------------------------------------------------
# 2. donor-paired flagged-vs-kept DE + exclusions
# ----------------------------------------------------------------------------------------------
ART_RE = r"^MT-|^RP[SL]\d|^RPLP|^HB[ABDGEMQZ]\d?$|^IG[HKL][VDJCAGEM]|^IGLL|^JCHAIN$|^PPBP$|^PF4|^MALAT1$|^NEAT1$|" \
         r"^XIST$|^TSIX$|^RPS4Y|^DDX3Y$|^UTY$|^KDM5D$|^EIF1AY$|^USP9Y$|^ZFY$|^TXLNGY$|^NLGN4Y$|^MTRNR|^AC\d|^AL\d|^LINC"
EXCL_SET_RE = r"ISG|IFN|interferon|SIGLEC1|IEG|dissoc|heat|ribosom|mitochond|platelet|erythro"


@section("2_exclusions")
def exclusions(PB, var, meta, gs):
    vi = pd.Index(var)
    ex = {}
    for k, genes in gs.items():
        if re.search(EXCL_SET_RE, k, re.I):
            for g in genes:
                ex.setdefault(g, f"set:{k}")
    for g in var:
        if re.search(ART_RE, g):
            ex.setdefault(g, "artifact_regex")
    f = ARGS.extra_exclude or os.path.join(ARGS.root, "emb", f"{ARGS.tag}_isgblind300_seed0_full_dropped.txt")
    if os.path.exists(f):
        for g in open(f).read().split():
            ex.setdefault(g.strip(), "isgblind300_list")
        log("extra exclusion list", f)
    # genes whose donor-level expression correlates with donor ISG within SLE donors (within type; using all
    # query donors would confound disease with ISG and drop every SLE-associated gene)
    isg = np.where(meta.is_sle.values == 1, meta.isg.values, np.nan)
    L = PB["L"]
    ntot = PB["nkept"] + PB["nflag"]
    rho = np.full((L.shape[1], L.shape[2]), np.nan)
    for b in range(L.shape[1]):
        m = (ntot[:, b] >= ARGS.min_cells_score) & np.isfinite(isg)
        if m.sum() < 10:
            continue
        expr = PB["Ma"][m, b].mean(0) >= ARGS.min_mean
        r = np.full(L.shape[2], np.nan)
        r[expr] = spearman_vec(isg[m], L[m, b][:, expr])
        rho[b] = r
    med = np.nanmedian(np.abs(rho), axis=0)
    mx = np.nanmax(np.abs(rho), axis=0)
    bad = (med >= ARGS.isg_rho) | (mx >= 0.5)
    order = np.argsort(-np.nan_to_num(np.nanmean(np.abs(rho), 0), nan=-1))
    top = set(order[:ARGS.isg_top].tolist())
    for j in range(len(var)):
        if bad[j] or j in top:
            ex.setdefault(var[j], "ISG_correlated")
    PB["isg_rho_gene"] = pd.Series(np.nanmean(rho, 0), index=vi)
    cnt = pd.Series(ex).value_counts().to_dict()
    cnt = {k if not k.startswith("set:") else "curated_sets": v for k, v in cnt.items()}
    R["exclusions"] = dict(n_total=int(sum(1 for g in ex if g in vi)), by_reason=pd.Series(
        [("curated_sets" if v.startswith("set:") else v) for g, v in ex.items() if g in vi]).value_counts().to_dict())
    pd.Series(ex, name="reason").rename_axis("gene").reset_index().to_csv(os.path.join(OUT, "excluded_genes.csv"),
                                                                          index=False)
    log("excluded genes:", R["exclusions"])
    return set(g for g in ex if g in vi)


@section("2_paired_de")
def paired_de(PB, var, meta, excl):
    rows, summ = [], []
    isg = meta.isg.values
    sle = meta.is_sle.values == 1
    LFC = {}
    for b, t in enumerate(PB["types"]):
        valid = (PB["nflag"][:, b] >= ARGS.min_flag) & (PB["nkept"][:, b] >= ARGS.min_kept)
        expr = PB["Ma"][valid, b].mean(0) >= ARGS.min_mean if valid.sum() else np.zeros(len(var), bool)
        if valid.sum() < 3 or expr.sum() == 0:
            continue
        Lf = np.log2(PB["Mf"][:, b][:, expr] + ARGS.pc) - np.log2(PB["Mk"][:, b][:, expr] + ARGS.pc)
        LFC[t] = dict(valid=valid, genes=np.where(expr)[0], L=Lf)
        rec = dict(type=t, n_sle=int((valid & sle).sum()), n_h=int((valid & ~sle).sum()), n_genes=int(expr.sum()))
        out = pd.DataFrame({"type": t, "gene": np.asarray(var)[expr]})
        out["excluded"] = out.gene.isin(excl)
        for nm, m in [("sle", valid & sle), ("h", valid & ~sle)]:
            if m.sum() >= ARGS.min_donors_de:
                z, p = signed_rank(Lf[m])
                out[f"med_lfc_{nm}"] = np.median(Lf[m], 0)
                out[f"fracpos_{nm}"] = (Lf[m] > 0).mean(0)
                out[f"z_{nm}"], out[f"p_{nm}"] = z, p
                out[f"q_{nm}"] = bh(p)
                rec[f"n_up_{nm}"] = int(((out[f"q_{nm}"] < 0.05) & (z > 0)).sum())
                rec[f"n_dn_{nm}"] = int(((out[f"q_{nm}"] < 0.05) & (z < 0)).sum())
            else:
                for c in ["med_lfc", "fracpos", "z", "p", "q"]:
                    out[f"{c}_{nm}"] = np.nan
        if (valid & sle).sum() >= ARGS.min_donors_de and (valid & ~sle).sum() >= 5:
            a, h = Lf[valid & sle], Lf[valid & ~sle]
            ra = stats.rankdata(np.vstack([a, h]), axis=0)
            U = ra[: len(a)].sum(0) - len(a) * (len(a) + 1) / 2
            mu, sd = len(a) * len(h) / 2, np.sqrt(len(a) * len(h) * (len(a) + len(h) + 1) / 12)
            out["p_int"] = 2 * stats.norm.sf(np.abs((U - mu) / sd))
            out["q_int"] = bh(out.p_int.values)
        else:
            out["p_int"] = out["q_int"] = np.nan
        # correlation of the gene's flagged-vs-kept LFC with donor ISG in SLE donors
        m = valid & sle & np.isfinite(isg)
        out["rho_lfc_isg"] = spearman_vec(isg[m], Lf[m]) if m.sum() >= 10 else np.nan
        sig_sle = (out.q_sle < 0.05)
        spec = sig_sle & ((out.q_h >= 0.05) | out.q_h.isna()) & (out.q_int < 0.1)
        rec["n_sle_specific"] = int(spec.sum())
        rec["n_sle_specific_kept"] = int((spec & ~out.excluded & (out.rho_lfc_isg.abs() < ARGS.isg_rho)).sum())
        top = out[spec & ~out.excluded & (out.rho_lfc_isg.abs() < ARGS.isg_rho)].sort_values("z_sle", ascending=False)
        rec["top_sle_specific_up"] = ",".join(top[top.z_sle > 0].gene.head(8))
        summ.append(rec)
        rows.append(out)
    DE = pd.concat(rows) if rows else pd.DataFrame()
    DE.to_csv(os.path.join(OUT, "paired_de_flag_vs_kept.csv.gz"), index=False, compression="gzip")
    SM = pd.DataFrame(summ)
    wcsv(SM, "paired_de_summary.csv")
    R["step2"] = SM.to_dict(orient="records")
    log(SM.to_string())
    return LFC, DE


# ----------------------------------------------------------------------------------------------
# 3. residual LFC matrix + consensus NMF
# ----------------------------------------------------------------------------------------------
def design(meta, rows, extra=()):
    """Covariate design for donors in rows: intercept, ISG z, cohort, ancestry, sex dummies, age z (+extra)."""
    M = meta.loc[rows]
    cols = {"intercept": np.ones(len(M))}
    for k in ["isg"] + list(extra):
        if k in M and M[k].notna().sum() > 3:
            v = M[k].astype(float)
            v = v.fillna(v.median())
            cols[k] = ((v - v.mean()) / (v.std() + 1e-9)).values
    if "age_num" in M and M.age_num.notna().sum() > 5:
        v = M.age_num.astype(float)
        v = v.fillna(v.median())
        cols["age"] = ((v - v.mean()) / (v.std() + 1e-9)).values
    for k in ["cohort", "ancestry", "sex"]:
        if k in M:
            lv = collapse_levels(M[k].values, 3)
            levels = sorted(lv.unique())
            for l in levels[1:]:
                cols[f"{k}={l}"] = (lv.values == l).astype(float)
    Z = pd.DataFrame(cols, index=rows)
    Z = Z.loc[:, (Z.std() > 0) | (Z.columns == "intercept")]
    return Z


def design_like(meta, rows, Zref, extra=()):
    """Design for other donors using the column set of Zref (unseen levels -> 0) and Zref's standardization."""
    M = meta.loc[rows]
    out = pd.DataFrame(0.0, index=rows, columns=Zref.columns)
    out["intercept"] = 1.0
    ref = meta.loc[Zref.index]
    for k in ["isg"] + list(extra):
        if k in out:
            mu, sd = ref[k].astype(float).mean(), ref[k].astype(float).std() + 1e-9
            out[k] = ((M[k].astype(float).fillna(ref[k].median()) - mu) / sd).values
    if "age" in out:
        a = ref.age_num.astype(float)
        out["age"] = ((M.age_num.astype(float).fillna(a.median()) - a.mean()) / (a.std() + 1e-9)).values
    for c in Zref.columns:
        if "=" in c:
            k, l = c.split("=", 1)
            out[c] = (M[k].astype(str).values == l).astype(float)
    return out


@section("3_feature_matrix")
def feature_matrix(LFC, PB, var, meta, excl):
    sle_d = meta.index[meta.is_sle == 1]
    extra = [x for x in ARGS.regress_extra.split(",") if x]
    Z = design(meta, sle_d, extra)
    # center covariates on SLE donors so the residual keeps each feature's SLE mean flagged-vs-kept shift
    # (programs are then nonnegative deviations of flagged cells, not mirror images of a centered matrix)
    cm = Z.mean()
    cm["intercept"] = 0.0
    Z = Z - cm
    sle_mask = meta.is_sle.values == 1
    blocks, feats, Hblocks, used_types, tinfo, mu_h, mu_s = [], [], [], [], [], [], []
    hl_d = meta.index[meta.is_sle == 0]
    Zh = design_like(meta, hl_d, Z + cm, extra) - cm
    var = np.asarray(var)
    for t, d in LFC.items():
        b = PB["types"].index(t)
        valid = d["valid"]
        cov = (valid & sle_mask).sum() / max(sle_mask.sum(), 1)
        if cov < ARGS.min_type_cov or (valid & sle_mask).sum() < max(ARGS.min_donors_de, Z.shape[1] + 3):
            continue
        genes = d["genes"]
        keepg = np.array([var[g] not in excl for g in genes])
        Lf = d["L"][:, keepg]
        gidx = genes[keepg]
        w = PB["nflag"][:, b] / (PB["nflag"][:, b] + ARGS.shrink_k0)
        Ls = Lf * w[:, None]
        rows_s = valid & sle_mask
        # residualize on SLE donors with valid LFC
        Zs = Z.loc[meta.index[rows_s]].values
        beta, *_ = np.linalg.lstsq(Zs, Ls[rows_s], rcond=None)
        beta[0] = 0.0  # keep the intercept (SLE mean shift); remove only covariate effects
        res = Ls[rows_s] - Zs @ beta
        # drop features whose residual LFC still tracks ISG (nonlinear leakage)
        iv = meta.isg.values[rows_s].astype(float)
        fin = np.isfinite(iv)
        rr = spearman_vec(iv[fin], res[fin]) if fin.sum() >= 10 else np.zeros(res.shape[1])
        okf = np.abs(rr) < ARGS.isg_rho
        # feature selection: residual variance
        v = (res - res.mean(0)).var(0) * okf
        sel = np.argsort(-v)[: min(ARGS.genes_per_type, int(okf.sum()))]
        sel = sel[v[sel] > 0]
        if len(sel) < 10:
            continue
        sc = np.sqrt(np.mean(res[:, sel] ** 2)) + 1e-9
        full = np.zeros((len(sle_d), len(sel)), np.float32)
        full[np.where(rows_s[sle_mask])[0]] = res[:, sel] / sc
        blocks.append(full)
        # healthy projection with SLE-fitted coefficients
        rows_h = valid & ~sle_mask
        hb = np.zeros((len(hl_d), len(sel)), np.float32)
        if rows_h.sum():
            Zhh = Zh.loc[meta.index[rows_h]].values
            hb[np.where(rows_h[~sle_mask])[0]] = (Ls[rows_h][:, sel] - Zhh @ beta[:, sel]) / sc
        Hblocks.append(hb)
        for j in sel:
            feats.append((t, var[gidx[j]]))
        tinfo.append(dict(type=t, valid=valid, L=Lf[:, sel] / sc, n=len(sel)))
        hv_ = valid & ~sle_mask
        mu_h.append(Lf[hv_][:, sel].mean(0) / sc if hv_.sum() >= 3 else np.full(len(sel), np.nan))
        mu_s.append(Lf[rows_s][:, sel].mean(0) / sc)
        used_types.append(dict(type=t, coverage=float(cov), n_feat=int(len(sel)), n_isg_resid_drop=int((~okf).sum()),
                               r2_covariates=float(1 - res[:, sel].var() / max(Ls[rows_s][:, sel].var(), 1e-12))))
    if not blocks:
        raise RuntimeError("no fine type passed coverage filters for NMF")
    Rm = np.hstack(blocks)
    Rh = np.hstack(Hblocks)
    F = pd.DataFrame(feats, columns=["type", "gene"])
    Xn = np.hstack([np.maximum(Rm, 0), np.maximum(-Rm, 0)])
    Xh = np.hstack([np.maximum(Rh, 0), np.maximum(-Rh, 0)])
    FF = pd.concat([F.assign(sign="+"), F.assign(sign="-")], ignore_index=True)
    R["step3_matrix"] = dict(n_sle_donors=int(len(sle_d)), n_healthy=int(len(hl_d)), n_types=len(used_types),
                             n_features=int(Rm.shape[1]), covariates=list(Z.columns), types=used_types)
    wcsv(pd.DataFrame(used_types), "nmf_types.csv")
    log(R["step3_matrix"]["n_types"], "types,", Rm.shape[1], "features; covariates", list(Z.columns))
    return dict(X=Xn, Xh=Xh, R=Rm, Rh=Rh, F=FF, sle=list(sle_d), healthy=list(hl_d), Z=Z, tinfo=tinfo,
                mu_h=np.concatenate(mu_h), mu_s=np.concatenate(mu_s))


def program_scores(FM, H, meta, min_mass=0.5):
    """Signed projection of each donor's RAW (unshrunk, unresidualized) flagged-vs-kept log2FC on each program,
    using only the fine types observed for that donor. Noise has zero mean, so donors with few flagged cells
    (e.g. healthy) are not biased toward 0, unlike NNLS usage of the nonnegative split matrix."""
    nf = H.shape[1] // 2
    V = H[:, :nf] - H[:, nf:]
    nD = len(meta)
    num = np.zeros((nD, H.shape[0]))
    den = np.zeros((nD, H.shape[0]))
    tot = (V ** 2).sum(1)
    j0 = 0
    for ti in FM["tinfo"]:
        vt = V[:, j0:j0 + ti["n"]]
        j0 += ti["n"]
        m = ti["valid"]
        Lt = np.nan_to_num(ti["L"][m])
        num[m] += Lt @ vt.T
        den[m] += (vt ** 2).sum(1)[None, :]
    S = np.where(den >= min_mass * tot[None, :], num / np.maximum(den, 1e-12), np.nan)
    return pd.DataFrame(S, index=meta.index, columns=[f"P{i + 1}" for i in range(H.shape[0])])


def nmf_fit(X, k, seed, max_iter=400):
    from sklearn.decomposition import NMF
    m = NMF(n_components=k, init="random", random_state=seed, max_iter=max_iter, tol=1e-4)
    W = m.fit_transform(X)
    return W, m.components_


def consensus(X, k, n_rep, frac, seed0=0, rng=None):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rng = rng or np.random.default_rng(seed0)
    n = X.shape[0]
    spectra, rep_id = [], []
    for r in range(n_rep):
        idx = rng.choice(n, max(int(frac * n), k + 2), replace=False) if frac < 1 else np.arange(n)
        _, H = nmf_fit(X[idx], k, seed0 * 1000 + r)
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        spectra.append(H)
        rep_id += [r] * k
    S = np.vstack(spectra)
    rep_id = np.array(rep_id)
    # cNMF-style outlier filter on local density (mean distance to nearest neighbours)
    nn = max(2, int(0.3 * n_rep))
    Dm = np.sqrt(np.maximum(2 - 2 * S @ S.T, 0))
    np.fill_diagonal(Dm, np.inf)
    dens = np.sort(Dm, axis=1)[:, :nn].mean(1)
    keep = dens <= 0.5
    if keep.sum() < 0.5 * len(S):
        keep = dens <= np.quantile(dens, 0.9)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(S[keep])
    lab = km.labels_
    sil = float(silhouette_score(S[keep], lab)) if len(np.unique(lab)) > 1 else np.nan
    cons = []
    for c in range(k):
        m = np.median(S[keep][lab == c], axis=0)
        cons.append(m / (np.linalg.norm(m) + 1e-12))
    H = np.vstack(cons)
    # replicate consistency: fraction of replicates whose k spectra hit k distinct clusters
    lab_full = np.full(len(S), -1)
    lab_full[np.where(keep)[0]] = lab
    rc = np.mean([len(set(lab_full[rep_id == r]) - {-1}) == k for r in range(n_rep)])
    W = usage(X, H)
    err = float(np.linalg.norm(X - W @ H) / (np.linalg.norm(X) + 1e-12))
    return dict(H=H, W=W, sil=sil, rep_consistency=float(rc), err=err, frac_kept=float(keep.mean()))


def usage(X, H):
    from scipy.optimize import nnls
    W = np.zeros((X.shape[0], H.shape[0]))
    for i in range(X.shape[0]):
        W[i] = nnls(H.T, X[i])[0]
    return W


@section("3_cnmf")
def run_cnmf(FM):
    X = FM["X"]
    res, rows = {}, []
    kmax = min(ARGS.kmax, X.shape[0] // 4)
    rng_null = np.random.default_rng(99)
    Xnull = np.column_stack([rng_null.permutation(X[:, j]) for j in range(X.shape[1])])
    for k in range(ARGS.kmin, max(kmax, ARGS.kmin) + 1):
        t = time.time()
        c = consensus(X, k, ARGS.n_rep, ARGS.sub_frac, seed0=k)
        cn = consensus(Xnull, k, max(6, ARGS.n_rep // 3), ARGS.sub_frac, seed0=500 + k)
        res[k] = c
        rows.append(dict(k=k, silhouette=c["sil"], rep_consistency=c["rep_consistency"], rel_error=c["err"],
                         frac_kept=c["frac_kept"], null_silhouette=cn["sil"], null_rel_error=cn["err"]))
        log(f"k={k}: sil {c['sil']:.3f} (null {cn['sil']:.3f}) consistency {c['rep_consistency']:.2f} "
            f"err {c['err']:.3f} (null {cn['err']:.3f}) [{time.time() - t:.0f}s]")
    K = pd.DataFrame(rows)
    wcsv(K, "k_selection.csv")
    # choose k: maximal silhouette margin over null among k with consistency >= 0.5; ties -> smaller k
    K["margin"] = K.silhouette - K.null_silhouette
    cand = K[K.rep_consistency >= 0.5]
    if len(cand) == 0:
        cand = K
    kbest = int(cand.sort_values(["margin", "k"], ascending=[False, True]).iloc[0].k)
    R["step3_k"] = dict(table=K.to_dict(orient="records"), k=kbest)
    try:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot(K.k, K.silhouette, "o-", label="data")
        ax[0].plot(K.k, K.null_silhouette, "s--", color="grey", label="permuted")
        ax[0].axvline(kbest, color="k", lw=1, ls=":")
        ax[0].set_xlabel("k")
        ax[0].set_ylabel("consensus silhouette")
        ax[0].legend()
        ax[1].plot(K.k, K.rel_error, "o-")
        ax[1].plot(K.k, K.null_rel_error, "s--", color="grey")
        ax[1].set_xlabel("k")
        ax[1].set_ylabel("relative error")
        savefig(fig, "fig_k_selection.png")
    except Exception as e:  # noqa: BLE001
        log("fig k failed", e)
    log("chosen k =", kbest)
    return kbest, res[kbest]


# ----------------------------------------------------------------------------------------------
# 4. program characterisation
# ----------------------------------------------------------------------------------------------
def type_markers(PB, var, n=50):
    """Top marker genes per fine type (kept-cell pseudobulk, log2 ratio vs mean of other types)."""
    Mk = np.nanmean(np.where((PB["nkept"] >= ARGS.min_kept)[..., None], PB["Mk"], np.nan), axis=0)  # nT x G
    L = np.log2(np.nan_to_num(Mk) + ARGS.pc)
    mk = {}
    for b, t in enumerate(PB["types"]):
        oth = np.delete(L, b, axis=0)
        sc = L[b] - oth.max(0)
        mk[t] = set(np.asarray(var)[np.argsort(-sc)[:n]])
    return mk


def gene_level(Hrow, F):
    s = pd.Series(np.where(F.sign.values == "+", 1, -1) * Hrow, index=F.index)
    df = F.assign(w=s.values)
    g = df.groupby("gene").w.sum()
    dom = df.assign(a=df.w.abs()).sort_values("a", ascending=False).drop_duplicates("gene").set_index("gene").type
    return g, dom


@section("4_programs")
def characterize(FM, k, C, PB, var, meta, DE, lineage, gs):
    F, H, W = FM["F"], C["H"], C["W"]
    sle, hl = FM["sle"], FM["healthy"]
    Wh = usage(FM["Xh"], H) if len(hl) else np.zeros((0, k))
    M = meta.loc[sle]
    mk = type_markers(PB, var)
    # cohort-stratified split-half replication and leave-one-cohort-out
    rep = replication(FM, k, H, meta)
    SCo = rep.pop("_oof", None)
    prog_rows, gene_tables = [], []
    U = pd.DataFrame(W, index=sle, columns=[f"P{i + 1}" for i in range(k)])
    SC = program_scores(FM, H, meta)
    if SCo is None:
        log("WARNING: no out-of-fold scores; clinical tests use in-sample scores")
        SCo = SC
    SC.add_suffix("_score").join(SCo.add_suffix("_score_oof")).join(meta).reset_index().to_csv(os.path.join(OUT, "program_scores_donors.csv"),
                                                             index=False)
    Uh = pd.DataFrame(Wh, index=hl, columns=U.columns)
    pd.concat([U.assign(group="SLE"), Uh.assign(group="healthy")]).join(meta).reset_index().to_csv(
        os.path.join(OUT, "program_usage_donors.csv"), index=False)
    for p in range(k):
        name = f"P{p + 1}"
        info = dict(program=name)
        u, uh = U[name].values, Uh[name].values
        # loadings
        Fp = F.assign(load=H[p])
        Fp = Fp.sort_values("load", ascending=False)
        ts = F.assign(load=H[p] ** 2).groupby("type").load.sum()
        ts = ts / ts.sum()
        ls = ts.groupby(lambda t: lineage.get(t, t)).sum().sort_values(ascending=False)
        info["type_share"] = ts.sort_values(ascending=False).round(3).head(6).to_dict()
        info["lineage_share"] = ls.round(3).to_dict()
        info["n_lineages_ge015"] = int((ls >= 0.15).sum())
        nf_ = H.shape[1] // 2
        v_ = H[p, :nf_] - H[p, nf_:]
        for key, mu in [("cos_healthy_shift", FM["mu_h"]), ("cos_sle_shift", FM["mu_s"])]:
            mm = np.isfinite(mu)
            info[key] = float(v_[mm] @ mu[mm] / (np.linalg.norm(v_[mm]) * np.linalg.norm(mu[mm]) + 1e-12)) \
                if mm.sum() > 10 else np.nan
        g, dom = gene_level(H[p], F)
        up = g.sort_values(ascending=False)
        dn = g.sort_values()
        info["top_up"] = [f"{x}({dom[x]})" for x in up.index[:20] if up[x] > 0]
        info["top_down"] = [f"{x}({dom[x]})" for x in dn.index[:10] if dn[x] < 0]
        gt = pd.DataFrame({"program": name, "gene": g.index, "weight": g.values, "dominant_type": dom.reindex(g.index)})
        gene_tables.append(gt.sort_values("weight", ascending=False))
        top_feat = Fp.head(20)
        top_up_feat = Fp[Fp.sign == "+"].head(20)  # foreign markers matter only when UP in flagged cells
        foreign = [(r.type, r.gene) for r in top_up_feat.itertuples()
                   if any(r.gene in mk[t2] for t2 in mk if t2 != r.type and lineage.get(t2) != lineage.get(r.type))]
        info["foreign_marker_frac_top20"] = len(foreign) / max(len(top_up_feat), 1)
        info["foreign_examples"] = [f"{g_}@{t_}" for t_, g_ in foreign[:5]]
        # paired-DE support: top up features significant in SLE flagged-vs-kept DE (same sign)
        if len(DE):
            dd = DE.set_index(["type", "gene"])
            sup = 0
            for r in top_feat.itertuples():
                if (r.type, r.gene) in dd.index:
                    row = dd.loc[(r.type, r.gene)]
                    sgn = 1 if r.sign == "+" else -1
                    if np.isfinite(row.q_sle) and row.q_sle < 0.05 and np.sign(row.z_sle) == sgn:
                        sup += 1
            info["top20_paired_de_sle"] = sup
        # associations: signed program score on raw LFC (primary) and NNLS usage of the residual matrix
        sc_s, sc_h = SC.loc[sle, name].values, SC.loc[hl, name].values
        info["n_scored"] = dict(sle=int(np.isfinite(sc_s).sum()), healthy=int(np.isfinite(sc_h).sum()))
        info["rho_isg"] = spearman_ci(sc_s, M.isg.values)
        info["rho_isg_usage"] = spearman_ci(u, M.isg.values)
        info["rho_usage_score"] = spearman_ci(u, sc_s)
        for key, col in [("gc", "gc"), ("gc_core", "gc_core"), ("diss", "diss"), ("hsp", "hsp"), ("ifng", "ifng"),
                         ("flag_rate", "flag_rate"), ("age", "age_num"), ("nflag", "n_flag_total")]:
            info[f"rho_{key}"] = spearman_ci(sc_s, M[col].astype(float).values) if col in M else None
        for c in ["cohort", "ancestry", "sex"]:
            if c in M:
                lv = collapse_levels(M[c].values, 3).values
                info[f"kw_{c}"] = kruskal_groups(sc_s, lv)
                info[f"kw_{c}_usage"] = kruskal_groups(u, lv)
        # clinical tests use OUT-OF-FOLD scores for SLE donors (spectra not fitted on that donor)
        so_s, so_h = SCo.loc[sle, name].values, SCo.loc[hl, name].values
        info["rho_isg_oof"] = spearman_ci(so_s, M.isg.values)
        coh_s = M["cohort"].astype(str).values if "cohort" in M else np.full(len(M), "all")
        if "status" in M:
            st = M.status.astype(str).str.lower().values.copy()
            if "status_mixed" in M:
                st[M.status_mixed.values.astype(bool)] = "mixed"
            info["n_status_mixed"] = int((st == "mixed").sum())
            stt = np.where(st == "mixed", "zz_excluded", st)
            info["kw_status"] = kruskal_groups(np.where(st == "mixed", np.nan, so_s), stt)
            f_, m_, t_ = so_s[st == "flare"], so_s[st == "managed"], so_s[st == "treated"]
            ok = lambda v: v[np.isfinite(v)]  # noqa: E731
            info["flare_vs_managed"] = auroc_ci(ok(f_), ok(m_)) if len(ok(f_)) >= 3 and len(ok(m_)) >= 3 else None
            info["treated_vs_flare"] = auroc_ci(ok(t_), ok(f_)) if len(ok(t_)) >= 3 and len(ok(f_)) >= 3 else None
            # disease_state is often processed in specific cohorts: flare vs managed within cohort
            mm = np.isin(st, ["flare", "managed"]) & np.isfinite(so_s)
            info["flare_vs_managed_strat"] = strat_perm(so_s[mm], (st[mm] == "flare").astype(int), coh_s[mm])
        mh = np.isfinite(so_h)
        ms = np.isfinite(so_s)
        if mh.sum() >= 3:
            # one value per individual (repeat samples of one person are not independent)
            dfc = pd.DataFrame({"v": np.r_[so_s[ms], so_h[mh]], "g": np.r_[np.ones(ms.sum()), np.zeros(mh.sum())],
                                "c": np.r_[coh_s[ms], (meta.loc[hl, "cohort"].astype(str).values if "cohort" in meta
                                                       else np.full(len(hl), "all"))[mh]],
                                "i": np.r_[M.individual.astype(str).values[ms],
                                           meta.loc[hl, "individual"].astype(str).values[mh]]})
            dfc = dfc.groupby(["i", "g"], as_index=False).agg(v=("v", "mean"), c=("c", "first"))
            a_, h_ = dfc.v[dfc.g == 1].values, dfc.v[dfc.g == 0].values
            info["sle_vs_healthy"] = auroc_ci(a_, h_)
            info["sle_vs_healthy_insample"] = auroc(sc_s[np.isfinite(sc_s)], sc_h[np.isfinite(sc_h)])
            info["sle_vs_healthy_strat"] = strat_perm(dfc.v.values, dfc.g.values.astype(int), dfc.c.values)
        info["replication"] = {kk: (vv[p] if isinstance(vv, (list, np.ndarray)) else vv) for kk, vv in rep.items()
                               if kk in ["half_vs_full_median", "half_vs_full_p10", "halfA_vs_halfB_median",
                                         "loco_min", "proj_usage_rho_median"]}
        prog_rows.append(info)
    GT = pd.concat(gene_tables)
    wcsv(GT, "program_gene_weights.csv")
    F.assign(**{f"P{i + 1}": H[i] for i in range(k)}).to_csv(os.path.join(OUT, "program_spectra.csv.gz"),
                                                            index=False, compression="gzip")
    R["step4"] = dict(programs=prog_rows, replication_overall=rep)
    figures_programs(SC.loc[sle], SC.loc[hl], M, F, H, lineage)
    return prog_rows, GT


def replication(FM, k, Hfull, meta=None):
    X = FM["X"]
    meta_c = R.get("_cohort_by_donor", {})
    coh = np.array([meta_c.get(d, "all") for d in FM["sle"]])
    ind = (meta.loc[FM["sle"], "individual"].astype(str).values if meta is not None and "individual" in meta
           else np.asarray(FM["sle"]).astype(str))
    rng = np.random.default_rng(7)
    hv, ab, pu = [], [], []
    # out-of-fold program scores: SLE donors scored with spectra fitted on the other half (no in-sample overfit);
    # healthy donors scored with both halves' spectra
    oof_sum = np.zeros((len(meta), k)) if meta is not None else None
    oof_n = np.zeros((len(meta), k)) if meta is not None else None
    sle_pos = meta.index.get_indexer(FM["sle"]) if meta is not None else None
    for s in range(ARGS.n_splits):
        half = np.zeros(len(coh), bool)
        for c in np.unique(coh):
            # assign whole individuals (not units) to halves, within cohort
            iu = rng.permutation(np.unique(ind[coh == c]))
            half[(coh == c) & np.isin(ind, iu[: len(iu) // 2])] = True
        if half.sum() < k + 3 or (~half).sum() < k + 3:
            continue
        ca = consensus(X[half], k, ARGS.n_rep_split, 1.0, seed0=1000 + s)
        cb = consensus(X[~half], k, ARGS.n_rep_split, 1.0, seed0=2000 + s)
        ma, pa = match_programs(Hfull, ca["H"])
        mb, pb_ = match_programs(Hfull, cb["H"])
        if meta is not None:
            for Hx, test_mask in [(ca["H"][pa], ~half), (cb["H"][pb_], half)]:
                Sx = program_scores(FM, Hx, meta).values
                rows_t = np.zeros(len(meta), bool)
                rows_t[sle_pos[test_mask]] = True
                rows_t |= (meta.is_sle.values == 0)
                v = np.where(rows_t[:, None] & np.isfinite(Sx), Sx, 0.0)
                oof_sum += v
                oof_n += rows_t[:, None] & np.isfinite(Sx)
        hv.append(np.r_[ma, mb].reshape(2, -1).min(0))
        ab.append(np.array([tucker(ca["H"][pa[i]:pa[i] + 1], cb["H"][pb_[i]:pb_[i] + 1])[0, 0]
                            if pa[i] >= 0 and pb_[i] >= 0 else np.nan for i in range(k)]))
        # held-out projection: usage of half-B donors under half-A spectra vs under full spectra
        Wb_a = usage(X[~half], ca["H"][pa])
        Wb_f = usage(X[~half], Hfull)
        pu.append(np.array([stats.spearmanr(Wb_a[:, i], Wb_f[:, i])[0] for i in range(k)]))
    out = {}
    if hv:
        hv, ab, pu = np.vstack(hv), np.vstack(ab), np.vstack(pu)
        out.update(n_splits=int(len(hv)), half_vs_full_median=np.nanmedian(hv, 0).round(3).tolist(),
                   half_vs_full_p10=np.nanpercentile(hv, 10, axis=0).round(3).tolist(),
                   halfA_vs_halfB_median=np.nanmedian(ab, 0).round(3).tolist(),
                   proj_usage_rho_median=np.nanmedian(pu, 0).round(3).tolist())
    lo = []
    for c in np.unique(coh):
        m = coh != c
        if m.sum() >= max(10, 2 * k) and (~m).sum() >= 3 and len(np.unique(coh)) > 1:
            cc = consensus(X[m], k, ARGS.n_rep_split, 1.0, seed0=3000)
            lo.append(match_programs(Hfull, cc["H"])[0])
    if lo:
        out["loco_min"] = np.nanmin(np.vstack(lo), 0).round(3).tolist()
        out["loco_n"] = len(lo)
    if meta is not None and oof_n is not None and oof_n.sum() > 0:
        out["_oof"] = pd.DataFrame(np.where(oof_n > 0, oof_sum / np.maximum(oof_n, 1), np.nan), index=meta.index,
                                   columns=[f"P{i + 1}" for i in range(k)])
    log("replication:", {kk: vv for kk, vv in out.items() if kk != "_oof"})
    return out


def figures_programs(U, Uh, M, F, H, lineage):
    k = U.shape[1]
    try:
        fig, axs = plt.subplots(1, k, figsize=(4.2 * k, 4.2), squeeze=False)
        for i, c in enumerate(U.columns):
            ax = axs[0, i]
            ax.scatter(M.isg, U[c], s=25, color="#C8553D", alpha=0.8)
            ax.set_xlabel("donor ISG (z)")
            ax.set_ylabel(f"{c} score")
        savefig(fig, "fig_score_vs_isg.png")
    except Exception as e:  # noqa: BLE001
        log("fig usage failed", e)
    try:
        ts = pd.DataFrame({f"P{i + 1}": F.assign(l=H[i] ** 2).groupby("type").l.sum() for i in range(k)})
        ts = ts / ts.sum(0)
        fig, ax = plt.subplots(figsize=(1.2 * k + 4, 0.38 * len(ts) + 2))
        im = ax.imshow(ts.values, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(ts)))
        ax.set_yticklabels([f"{t} [{lineage.get(t, '')}]" for t in ts.index], fontsize=12)
        ax.set_xticks(range(k))
        ax.set_xticklabels(ts.columns)
        fig.colorbar(im, ax=ax, label="loading share")
        savefig(fig, "fig_program_type_share.png")
    except Exception as e:  # noqa: BLE001
        log("fig share failed", e)
    try:
        fig, axs = plt.subplots(1, k, figsize=(3.6 * k, 4.2), squeeze=False)
        for i, c in enumerate(U.columns):
            ax = axs[0, i]
            st = M.status.astype(str).str.lower() if "status" in M else pd.Series("SLE", index=M.index)
            groups = [("healthy", Uh[c].values)] + [(s, U[c].values[st.values == s]) for s in sorted(st.unique())]
            groups = [(n_, v[np.isfinite(v)]) for n_, v in groups]
            groups = [(n_, v) for n_, v in groups if len(v)]
            ax.boxplot([v for _, v in groups], showfliers=False)
            for j, (_, v) in enumerate(groups):
                ax.scatter(np.full(len(v), j + 1) + RNG.normal(0, 0.06, len(v)), v, s=10, alpha=0.6)
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([n_ for n_, _ in groups], rotation=45, ha="right")
            ax.set_ylabel(f"{c} score")
        savefig(fig, "fig_score_by_state.png")
    except Exception as e:  # noqa: BLE001
        log("fig state failed", e)


# ----------------------------------------------------------------------------------------------
# 5. enrichment + gates
# ----------------------------------------------------------------------------------------------
@section("5_enrichment")
def enrichment(GT, gs, k):
    universe = set(GT.gene.unique())
    rows = []
    for p in sorted(GT.program.unique()):
        g = GT[GT.program == p].set_index("gene").weight
        for side, sel in [("up", g[g > 0].sort_values(ascending=False).index[:ARGS.top_n_enrich]),
                          ("down", g[g < 0].sort_values().index[:ARGS.top_n_enrich])]:
            sel = set(sel)
            for name, genes in gs.items():
                S = set(genes) & universe
                x = len(S & sel)
                p_ = stats.hypergeom.sf(x - 1, len(universe), len(S), len(sel)) if len(S) and len(sel) else np.nan
                rows.append(dict(program=p, side=side, set=name, n_set_universe=len(S), overlap=x,
                                 genes=",".join(sorted(S & sel)), p=p_))
    E = pd.DataFrame(rows)
    if len(E):
        E["q"] = np.nan
        for (p, s), idx in E.groupby(["program", "side"]).groups.items():
            E.loc[idx, "q"] = bh(E.loc[idx, "p"].values)
    wcsv(E, "program_enrichment.csv")
    R["step5_n_sets"] = len(gs)
    return E


TREAT_ART_RE = r"glucocorticoid|GC_induced|MS1|IEG|dissoc|heat|ribosom|mitochond|platelet|erythro|ISG|IFN|SIGLEC1"
KNOWN_RE = r"DN2|activated_naive|plasmablast|Tph|Tfh|cytotoxic|GZMK|exhaustion|Treg|adaptive_NK|DC3|LDG|neutrophil|" \
           r"HLA_classII|NFkB|hypoxia|cell_cycle|AHR|naive_quiescence"


@section("5_gates")
def gates(progs, E):
    rows = []
    for P in progs:
        p = P["program"]
        rep = P.get("replication", {})
        g = {}
        rr = lambda key: (P.get(key) or {}).get("rho", np.nan)  # noqa: E731
        r = P.get("rho_isg") or {}
        g["isg_independent"] = bool(np.isfinite(r.get("rho", np.nan)) and abs(r["rho"]) < 0.3 and r["lo"] > -0.45
                                    and r["hi"] < 0.45 and abs(rr("rho_isg_usage")) < 0.3
                                    and not abs(rr("rho_isg_oof")) >= 0.3)
        g["not_gc"] = bool(all(not np.isfinite(rr(k)) or abs(rr(k)) < 0.3 for k in ["rho_gc", "rho_gc_core"]))
        g["not_diss_hsp"] = bool(all(not np.isfinite(rr(k)) or abs(rr(k)) < 0.3 for k in ["rho_diss", "rho_hsp"]))
        coh_p = [(P.get(f"kw_{c}") or {}).get("p", np.nan) for c in ["cohort", "ancestry"]]
        g["not_cohort_ancestry"] = bool(all(not np.isfinite(x) or x >= 0.01 for x in coh_p))
        g["cross_lineage"] = P["n_lineages_ge015"] >= 2
        g["not_foreign_markers"] = P["foreign_marker_frac_top20"] < 0.25
        hv = rep.get("halfA_vs_halfB_median", np.nan)
        lc = rep.get("loco_min", np.nan)
        g["replicates"] = bool(np.isfinite(hv) and hv >= 0.7 and (not np.isfinite(lc) or lc >= 0.7))
        sh = P.get("sle_vs_healthy") or {}
        shs = P.get("sle_vs_healthy_strat") or {}
        fms = P.get("flare_vs_managed_strat") or {}
        kws = fms.get("p", np.nan)
        if not np.isfinite(kws):
            kws = (P.get("kw_status") or {}).get("p", np.nan)
        g["clinical"] = bool((np.isfinite(sh.get("lo", np.nan)) and sh["lo"] > 0.5
                              and np.isfinite(shs.get("p", np.nan)) and shs["p"] < 0.05 and shs.get("stat", 0) > 0)
                             or (np.isfinite(kws) and kws < 0.05))
        ch = P.get("cos_healthy_shift", np.nan)
        g["not_generic_outlier_axis"] = bool(not np.isfinite(ch) or abs(ch) < 0.5)
        sig = E[(E.program == p) & (E.side == "up") & (E.q < 0.05)] if len(E) else pd.DataFrame(columns=["set"])
        art = [s for s in sig.set if re.search(TREAT_ART_RE, s, re.I)]
        known = [s for s in sig.set if re.search(KNOWN_RE, s, re.I)]
        g["no_artifact_enrichment"] = len(art) == 0
        verdict = "CANDIDATE_NOVEL" if all(g.values()) else "fail:" + ",".join(k_ for k_, v in g.items() if not v)
        if all(g.values()) and known:
            verdict = "KNOWN_STATE(" + ",".join(known) + ")"
        P["gates"], P["verdict"], P["enriched_sets"] = g, verdict, list(sig.set)
        rows.append(dict(program=p, verdict=verdict, **g))
    G = pd.DataFrame(rows)
    wcsv(G, "program_gates.csv")
    R["gates"] = rows
    log(G.to_string())
    return G


# ----------------------------------------------------------------------------------------------
# RESULTS.md
# ----------------------------------------------------------------------------------------------
def write_md(progs):
    L = []
    ov = R.get("overview", {})
    s1 = R.get("step1", {})
    L.append("# SLE non-ISG programs of reference-unmatched states (lupus_programs.py)\n")
    L.append(f"Flag: p_unknown_ctrl<={ARGS.alpha} & p_unknown_bonf<={ARGS.alpha}, seeds {ov.get('seeds')} "
             f"({ov.get('seed_rule')}); seed Jaccard {[round(x, 2) for x in R.get('seed_jaccard', [])]}. "
             f"Cells {ov.get('n_cells')}, flag rate {fmt(ov.get('flag_rate'), 4)}.")
    if s1:
        a = s1.get("flag_rate_auc", {})
        L.append(f"Donors: {s1.get('n_sle')} SLE / {s1.get('n_healthy')} healthy-query, {s1.get('n_types')} types. "
                 f"Median donor flag rate SLE {fmt(s1['donor_flag_rate_median']['SLE'], 4)} vs healthy "
                 f"{fmt(s1['donor_flag_rate_median']['healthy'], 4)}; AUROC {fmt(a.get('auc'))} "
                 f"[{fmt(a.get('lo'))}, {fmt(a.get('hi'))}].")
    L.append("\n## Step 1-2: flagged vs kept, donor-paired DE (signed rank over donors, BH)")
    s2 = R.get("step2", [])
    if s2:
        L.append("type | nSLE/nH donors | up/dn SLE | up/dn H | SLE-specific (kept after excl.) | top SLE-specific up")
        L.append("---|---|---|---|---|---")
        for r in sorted(s2, key=lambda r: -r.get("n_sle_specific_kept", 0))[:12]:
            L.append(f"{r['type']} | {r['n_sle']}/{r['n_h']} | {r.get('n_up_sle', 'NA')}/{r.get('n_dn_sle', 'NA')} | "
                     f"{r.get('n_up_h', 'NA')}/{r.get('n_dn_h', 'NA')} | {r['n_sle_specific']} "
                     f"({r['n_sle_specific_kept']}) | {r.get('top_sle_specific_up', '')}")
    ex = R.get("exclusions", {})
    L.append(f"\nExcluded genes: {ex.get('n_total')} {ex.get('by_reason')}.")
    m3 = R.get("step3_matrix", {})
    k3 = R.get("step3_k", {})
    L.append(f"\n## Step 3: consensus NMF\nSLE donors {m3.get('n_sle_donors')}, types {m3.get('n_types')}, features "
             f"{m3.get('n_features')} (x2 signs); residualized on {m3.get('covariates')}.")
    if k3:
        L.append("k: " + "; ".join(f"{int(r['k'])}: sil {fmt(r['silhouette'])} (null {fmt(r['null_silhouette'])})"
                                   for r in k3["table"]) + f". **Chosen k = {k3['k']}**.")
    L.append("\n## Step 4-5: programs")
    for P in progs:
        def rc(key):
            r = P.get(key) or {}
            return f"{fmt(r.get('rho'))} [{fmt(r.get('lo'))},{fmt(r.get('hi'))}]"
        sh = P.get("sle_vs_healthy") or {}
        shs = P.get("sle_vs_healthy_strat") or {}
        fm = P.get("flare_vs_managed") or {}
        rep = P.get("replication", {})
        L.append(f"\n### {P['program']}: {P.get('verdict', '')}")
        L.append("Top genes (dominant type): " + ", ".join(P["top_up"][:20]))
        if P.get("top_down"):
            L.append("Down: " + ", ".join(P["top_down"][:6]))
        L.append(f"cos with healthy / SLE mean flagged-vs-kept shift {fmt(P.get('cos_healthy_shift'))} / "
                 f"{fmt(P.get('cos_sle_shift'))}.")
        L.append(f"Lineage share {P['lineage_share']}; foreign-marker frac {fmt(P['foreign_marker_frac_top20'])}; "
                 f"top20 with SLE paired-DE support {P.get('top20_paired_de_sle', 'NA')}.")
        L.append(f"rho ISG {rc('rho_isg')}, GC {rc('rho_gc')}, diss {rc('rho_diss')}, HSP {rc('rho_hsp')}, "
                 f"IFNg {rc('rho_ifng')}, flag rate {rc('rho_flag_rate')}, n flagged {rc('rho_nflag')}; "
                 f"usage-ISG {rc('rho_isg_usage')}; donors scored {P.get('n_scored')}.")
        L.append(f"SLE vs healthy AUROC {fmt(sh.get('auc'))} [{fmt(sh.get('lo'))},{fmt(sh.get('hi'))}], "
                 f"cohort-stratified p {fmt(shs.get('p'), 3)} (out-of-fold, per individual; in-sample AUROC "
                 f"{fmt(P.get('sle_vs_healthy_insample'))}); status KW p {fmt((P.get('kw_status') or {}).get('p'), 3)}"
                 f" (mixed-state units excluded: {P.get('n_status_mixed', 0)}), flare vs managed within-cohort p "
                 f"{fmt((P.get('flare_vs_managed_strat') or {}).get('p'), 3)}"
                 f", flare vs managed AUROC {fmt(fm.get('auc'))} [{fmt(fm.get('lo'))},{fmt(fm.get('hi'))}]; "
                 f"cohort KW p {fmt((P.get('kw_cohort') or {}).get('p'), 3)}, ancestry KW p "
                 f"{fmt((P.get('kw_ancestry') or {}).get('p'), 3)}.")
        L.append(f"Replication: split-half vs full congruence median {fmt(rep.get('half_vs_full_median'))} "
                 f"(p10 {fmt(rep.get('half_vs_full_p10'))}), A vs B {fmt(rep.get('halfA_vs_halfB_median'))}, "
                 f"LOCO min {fmt(rep.get('loco_min'))}, projected-usage rho {fmt(rep.get('proj_usage_rho_median'))}. "
                 f"Enriched sets (q<0.05): {', '.join(P.get('enriched_sets', [])[:5]) or 'none'}.")
    L.append("\nNovel = CANDIDATE_NOVEL: |rho ISG|<0.3 (CI in +-0.45), |rho| GC/diss/HSP<0.3, cohort/ancestry KW "
             "p>=0.01, ISG also on out-of-fold score, |cosine| with the healthy-query flagged-vs-kept mean shift<0.5 (not the generic outlier axis), "
             ">=2 lineages with loading share>=0.15, foreign-marker frac<0.25, split-half and LOCO "
             "congruence (independent halves A vs B, LOCO)>=0.7, SLE>healthy (AUROC CI lower>0.5, cohort-stratified p<0.05; out-of-fold scores) or "
             "flare vs managed within-cohort p<0.05, no enrichment in GC/IEG/HSP/ISG/ribo/mito/platelet/Hb sets; KNOWN_STATE if it passes but "
             "matches a known SLE state set.")
    if R["errors"]:
        L.append(f"\nErrors: {R['errors']}")
    txt = "\n".join(L)
    if len(txt.encode()) > 8000:
        log(f"RESULTS.md is {len(txt.encode())} bytes (> 8 KB); details in results.json")
    open(os.path.join(OUT, "RESULTS.md"), "w").write(txt + "\n")


def main(argv=None):
    global ARGS, OUT, LOGF
    ARGS = parse_args(argv)
    OUT = ARGS.out if os.path.isabs(ARGS.out) else os.path.join(ARGS.root, ARGS.out)
    os.makedirs(OUT, exist_ok=True)
    LOGF = open(os.path.join(OUT, "log.txt"), "w")
    R["args"] = vars(ARGS)
    D = load_cells()
    if D is None:
        raise RuntimeError("could not load cells")
    out = load_expression(D)
    if out is None:
        raise RuntimeError("could not load expression")
    D, X, var, obs = out
    gs = load_genesets()
    meta, lineage = detect_meta(obs, D)
    R["lineage"] = lineage
    R["_cohort_by_donor"] = meta["cohort"].astype(str).to_dict() if "cohort" in meta else {}
    PB = pseudobulk(D, X, meta)
    del X
    if PB is None:
        raise RuntimeError("pseudobulk failed")
    donor_scores(PB, var, meta, gs)
    excl = exclusions(PB, var, meta, gs) or set()
    o = paired_de(PB, var, meta, excl)
    LFC, DE = o if o else ({}, pd.DataFrame())
    FM = feature_matrix(LFC, PB, var, meta, excl)
    progs = []
    if FM is not None:
        o = run_cnmf(FM)
        if o:
            k, C = o
            o2 = characterize(FM, k, C, PB, var, meta, DE, lineage, gs)
            if o2:
                progs, GT = o2
                E = enrichment(GT, gs, k)
                gates(progs, E if E is not None else pd.DataFrame())
    R.pop("_cohort_by_donor", None)
    try:
        write_md(progs)
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
