"""Semi-synthetic benchmark of disease-state (out-of-reference) detection with matched controls.

Runs on the held-out-type runs (e.g. lupusfull_ho-ncM): the type was removed from the reference, so its query
cells are truly out-of-reference. Only the healthy query donors (cond == 'normal') are used. In each replicate
the donors are split into pseudo-cases (all cells kept, or a fraction f of the held-out-type cells: dilution)
and pseudo-controls (all held-out-type cells removed). Truth: held-out-type cells of pseudo-cases are novel.
Every method is calibrated or fitted on pseudo-controls only (no query labels are used):

  ARCANE-ctrl / ARCANE-ctrl-U  leave-one-donor-out control-calibrated p-values of `novelty` (both views) and
                               `novelty_U` (scVI view), exactly as control_recal() in evaluate2.py but with the
                               pseudo-control donors as controls; flag p <= 0.05, and BH q <= 0.1 ("... BH").
  ARCANE-ref                   reference-calibrated p_unknown_bonf as stored in res2 (flag 0.05, BH 0.1).
  scPSS-U / scPSS-PCA          mean distance to the k=15 nearest reference cells (scVI latent; or 50 PCs of
                               log-normalised expression fitted on <= 50k reference cells), empirical p-value
                               against pseudo-control cells (same per-donor-capped, leave-one-donor-out scheme).
  Milo                         Python re-implementation of Milo (Dann et al. 2022/2024): kNN graph (k=30) on the
                               scVI latent of case + control cells, refined 10% neighbourhoods, NB GLM per
                               neighbourhood (log library-size offset, method-of-moments dispersion shrunk towards
                               the common value), LRT, k-distance-weighted spatial FDR; a cell is case-enriched if
                               it is in any neighbourhood with SpatialFDR < 0.1 and logFC > 0.
  1-scANVI maxprob, 1-pmax_S   uncertainty baselines, thresholded at the 95th percentile of pseudo-control cells.

Usage (from ~/arcane):
  python analysis/dstate_bench.py --tags lupusfull_ho-ncM lupusfull_ho-pDC --seeds 0 1 2 --reps 20
Outputs: res3/dstate_bench/{per_replicate.csv, summary.json, key.txt, cache/}
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2, rankdata, spearmanr
from sklearn.neighbors import NearestNeighbors

ALPHA = 0.05
Q_BH = 0.1

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["lupusfull_ho-ncM", "lupusfull_ho-pDC"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--dilutions", nargs="+", type=float, default=[1.0, 0.5, 0.2])
ap.add_argument("--emb", default="emb")
ap.add_argument("--res2", default="res2")
ap.add_argument("--std", default="data/std")
ap.add_argument("--out", default="res3/dstate_bench")
ap.add_argument("--pca", dest="pca", action="store_true", default=None,
                help="scPSS-PCA variant (default: on when data/std/<ds>.h5ad and anndata are available)")
ap.add_argument("--no-pca", dest="pca", action="store_false")
ap.add_argument("--pca_ref", type=int, default=50000)
ap.add_argument("--k_pss", type=int, default=15)
ap.add_argument("--k_milo", type=int, default=30)
ap.add_argument("--milo_prop", type=float, default=0.1)
ap.add_argument("--milo_max", type=int, default=150000)
ap.add_argument("--max_per_donor", type=int, default=300)
ap.add_argument("--jobs", type=int, default=-1)
A = ap.parse_args()
OUT = A.out
os.makedirs(f"{OUT}/cache", exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


# ------------------------------------------------------------------------------------------ helpers
def fold_p(cal, q):
    """Split-conformal p-value (identical to arcane._fold_p)."""
    s = np.sort(cal)
    return (1.0 + len(s) - np.searchsorted(s, q, side="left")) / (len(s) + 1.0)


def bh(p):
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def auroc(score, pos):
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ctrl_calibrated_p(score, case_idx, ctrl_idx, donor, rng, max_per=300, balanced=True):
    """control_recal() of evaluate2.py with arbitrary control cells: each control donor contributes
    min(per, n_d) random cells, per = max(50, min(max_per, 10th percentile of control-donor sizes)); case cells
    are tested against all calibration cells, each control cell against the other control donors only."""
    dc = donor[ctrl_idx]
    dons, inv = np.unique(dc, return_inverse=True)
    sizes = np.bincount(inv)
    per = int(max(50, min(max_per, np.percentile(sizes, 10)))) if balanced else int(sizes.max())
    pick_d = []
    for j in range(len(dons)):
        w = np.where(inv == j)[0]
        pick_d.append(rng.choice(w, min(per, len(w)), replace=False))
    cal_all = np.sort(score[ctrl_idx[np.concatenate(pick_d)]])
    p_case = fold_p(cal_all, score[case_idx])
    p_ctrl = np.ones(len(ctrl_idx))
    n_all = len(cal_all)
    for j in range(len(dons)):
        m = inv == j
        q = score[ctrl_idx[m]]
        own = np.sort(score[ctrl_idx[pick_d[j]]])
        ge = (n_all - np.searchsorted(cal_all, q, side="left")) - (len(own) - np.searchsorted(own, q, side="left"))
        p_ctrl[m] = (1.0 + ge) / (n_all - len(own) + 1.0)
    return p_case, p_ctrl, per


def quantile_flags(score, case_idx, ctrl_idx, donor, rng, level=0.95):
    """Threshold at the 95th percentile of control cells; control cells use the other control donors only.
    Ties (e.g. maxprob == 1, kNN proportions in steps of 1/k) are broken at random so that the control flag rate
    is ~5% and the excess estimator applies."""
    span = float(np.ptp(score)) or 1.0
    score = score + rng.uniform(0, 1e-9 * span, len(score))
    thr = np.quantile(score[ctrl_idx], level)
    f_case = score[case_idx] > thr
    dc = donor[ctrl_idx]
    f_ctrl = np.zeros(len(ctrl_idx), bool)
    sc = score[ctrl_idx]
    for d in np.unique(dc):
        m = dc == d
        f_ctrl[m] = sc[m] > np.quantile(sc[~m], level)
    return f_case, f_ctrl


def knn_mean_dist(R, Z, k, jobs):
    return NearestNeighbors(n_neighbors=k, n_jobs=jobs).fit(R).kneighbors(Z)[0].mean(1)


# ------------------------------------------------------------------------------------------ Milo
def _nb_ll(Y, mu, a):
    """NB2 log-likelihood up to terms that do not depend on mu (they cancel in the LRT)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(Y > 0, Y * np.log(np.where(mu > 0, mu, 1.0)), 0.0)
    return (t - (Y + 1.0 / a[:, None]) * np.log1p(a[:, None] * mu)).sum(1)


def _nb_fit_rate(Y, L, a, iters=100):
    """Per-row MLE of eta in mu_d = L_d exp(eta) under NB2 with dispersion a (vectorised Newton; the
    log-likelihood is concave in eta). Rows with zero counts get eta = -inf (mu = 0)."""
    s = Y.sum(1)
    pos = s > 0
    eta = np.full(len(Y), -np.inf)
    eta[pos] = np.log(s[pos] / L.sum())
    Yp, ap_ = Y[pos], a[pos][:, None]
    e = eta[pos]
    for _ in range(iters):
        mu = L[None, :] * np.exp(e)[:, None]
        g = ((Yp - mu) / (1 + ap_ * mu)).sum(1)
        h = (mu * (1 + ap_ * Yp) / (1 + ap_ * mu) ** 2).sum(1)
        step = np.clip(g / np.maximum(h, 1e-300), -3, 3)
        e = e + step
        if np.abs(step).max() < 1e-10:
            break
    eta[pos] = e
    return eta


def nb_two_group_test(Y, L, x, prior_df=10.0):
    """NB GLM log(mu_dj) = log L_d + b0_j + b1_j x_d for every neighbourhood j (rows of Y), LRT for b1.
    Dispersion: method of moments under the group-wise Poisson fit, per neighbourhood, shrunk towards the
    common (pooled) estimate with prior_df degrees of freedom (in the spirit of edgeR's tagwise dispersion)."""
    x = x.astype(bool)
    nD = len(L)
    df = max(nD - 2, 1)
    mu_p = np.zeros_like(Y)
    for g in (False, True):
        m = x == g
        rate = Y[:, m].sum(1) / L[m].sum()
        mu_p[:, m] = rate[:, None] * L[None, m]
    num = ((Y - mu_p) ** 2 - mu_p).sum(1) * nD / df
    den = (mu_p ** 2).sum(1)
    common = max(num.sum() / max(den.sum(), 1e-12), 1e-4)
    raw = np.where(den > 0, num / np.maximum(den, 1e-12), common)
    a = np.clip((prior_df * common + df * raw) / (prior_df + df), 1e-4, None)
    e0 = _nb_fit_rate(Y[:, ~x], L[~x], a)
    e1 = _nb_fit_rate(Y[:, x], L[x], a)
    en = _nb_fit_rate(Y, L, a)
    mu1 = np.zeros_like(Y)
    with np.errstate(over="ignore"):
        mu1[:, ~x] = L[None, ~x] * np.exp(e0)[:, None]
        mu1[:, x] = L[None, x] * np.exp(e1)[:, None]
        mun = L[None, :] * np.exp(en)[:, None]
    lrt = np.maximum(2 * (_nb_ll(Y, mu1, a) - _nb_ll(Y, mun, a)), 0.0)
    pval = chi2.sf(lrt, 1)
    # log2 fold change for reporting (a group with zero counts gets half a count)
    f0 = np.where(np.isfinite(e0), e0, np.log(0.5 / L[~x].sum()))
    f1 = np.where(np.isfinite(e1), e1, np.log(0.5 / L[x].sum()))
    return (f1 - f0) / np.log(2), pval, a, common, (e0, e1, en)


def spatial_fdr(p, w):
    """miloR graphSpatialFDR (k-distance weighting): weighted BH with weights w."""
    p = np.where(np.isfinite(p), p, 1.0)
    o = np.argsort(p)
    pw, ww = p[o], w[o]
    adj = ww.sum() * pw / np.cumsum(ww)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(p)
    out[o] = np.minimum(adj, 1.0)
    return out


def milo_present_knn(nbK, dK, pres, Zp, k, jobs):
    """kNN (k, self included) among present universe cells from a precomputed larger kNN list; rows with fewer
    than k present neighbours in the list are recomputed exactly."""
    pidx = np.cumsum(pres) - 1
    rows = np.where(pres)[0]
    nb, dd = nbK[rows], dK[rows]
    ok = pres[nb]
    c = np.cumsum(ok, 1)
    sel = ok & (c <= k)
    full = sel.sum(1) == k
    idx = np.empty((len(rows), k), np.int64)
    dist = np.empty((len(rows), k))
    idx[full] = pidx[nb[full][sel[full]]].reshape(-1, k)
    dist[full] = dd[full][sel[full]].reshape(-1, k)
    tree = NearestNeighbors(n_neighbors=k, n_jobs=jobs).fit(Zp)
    if (~full).any():
        d2, i2 = tree.kneighbors(Zp[~full])
        idx[~full], dist[~full] = i2, d2
    return idx, dist, tree, int((~full).sum())


def milo_da(Zp, idx, dist, tree, don, case_don, k, prop, rng, check=False):
    """Milo on present cells. don: donor code per cell (0..nD-1); case_don: bool per donor code."""
    n = len(Zp)
    A_ = sp.csr_matrix((np.ones(n * k, np.float32), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    A_ = ((A_ + A_.T) > 0).astype(np.float32).tocsc()
    samp = rng.choice(n, max(1, int(round(prop * n))), replace=False)
    med = np.median(Zp[idx[samp]], axis=1)                       # refined sampling (Gut et al. / miloR)
    index_cells = np.unique(tree.kneighbors(med, n_neighbors=1)[1][:, 0])
    M = A_[:, index_cells].tocsr()                               # cells x nhoods membership
    kdist = np.maximum(dist[index_cells, k - 1], 1e-12)
    nD = int(don.max()) + 1
    Dm = sp.csr_matrix((np.ones(n, np.float32), (don, np.arange(n))), shape=(nD, n))
    Y = np.asarray((Dm @ M).todense(), float).T                  # nhoods x donors
    L = np.bincount(don, minlength=nD).astype(float)
    keep = L > 0
    Y, L, x = Y[:, keep], L[keep], case_don[keep]
    lfc, pval, a, common, etas = nb_two_group_test(Y, L, x)
    sfdr = spatial_fdr(pval, 1.0 / kdist)
    sig = (sfdr < 0.1) & (lfc > 0)
    flag = np.asarray(M @ sig.astype(np.float32)).ravel() > 0
    s = -np.log10(np.maximum(sfdr, 1e-300)) * np.sign(lfc)
    shift = s.min() - 1.0
    Ms = M.multiply((s - shift)[None, :]).tocsr()
    sc = np.asarray(Ms.max(axis=1).todense()).ravel()
    score = np.where(sc > 0, sc + shift, shift)                  # cells in no nhood get the lowest score
    info = dict(n_nhoods=int(len(index_cells)), n_sig_up=int(sig.sum()), n_sig_any=int((sfdr < 0.1).sum()),
                dispersion_common=float(common), dispersion_median=float(np.median(a)),
                mean_nhood_size=float(np.asarray(M.sum(0)).mean()))
    if check:
        info["statsmodels_check"] = statsmodels_check(Y, L, x, a, lfc, pval)
    return flag, score, info


def statsmodels_check(Y, L, x, a, lfc, pval, n=20):
    """Compare the vectorised NB fit with statsmodels GLM(NegativeBinomial(alpha)) on a few nhoods."""
    try:
        import statsmodels.api as sm
    except Exception as e:  # noqa
        return dict(skipped=str(e))
    both = np.where((Y[:, x].sum(1) > 0) & (Y[:, ~x].sum(1) > 0))[0][:n]
    X = sm.add_constant(x.astype(float))
    dl, dp = [], []
    for j in both:
        fam = sm.families.NegativeBinomial(alpha=float(a[j]))
        r1 = sm.GLM(Y[j], X, family=fam, offset=np.log(L)).fit()
        r0 = sm.GLM(Y[j], np.ones((len(L), 1)), family=fam, offset=np.log(L)).fit()
        p = chi2.sf(max(2 * (r1.llf - r0.llf), 0), 1)
        dl.append(abs(r1.params[1] / np.log(2) - lfc[j]))
        dp.append(abs(np.log10(max(p, 1e-300)) - np.log10(max(pval[j], 1e-300))))
    return dict(n=int(len(both)), max_abs_dlog2fc=float(np.max(dl)) if dl else None,
                max_abs_dlog10p=float(np.max(dp)) if dp else None)


# ------------------------------------------------------------------------------------------ scPSS-PCA
def pca_pss_scores(tag, holdout, names_H, k, n_ref, jobs):
    """Mean distance of healthy query cells to their k nearest reference cells in 50 PCs of log-normalised
    expression (2000 HVGs, z-scaled), all fitted on a random subsample of <= n_ref reference cells."""
    ds = tag.split("_ho-")[0]
    path = f"{A.std}/{ds}.h5ad"
    cache = f"{OUT}/cache/pss_pca_{tag}.npz"
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        s = pd.Series(c["score"], index=c["names"]).reindex(names_H).values
        if not np.isnan(s).any():
            return s, "cache"
    if not os.path.exists(path):
        return None, f"missing {path}"
    try:
        import anndata as ad
    except Exception as e:
        return None, f"anndata unavailable: {e}"
    from sklearn.decomposition import PCA
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    role = obs["role"].astype(str).values
    lab = obs["label"].astype(str).values
    ref_pos = np.where((role == "ref") & (lab != holdout))[0]
    rs = np.sort(np.random.default_rng(0).choice(ref_pos, min(n_ref, len(ref_pos)), replace=False))
    qpos = pd.Index(np.asarray(a.obs_names).astype(str)).get_indexer(np.asarray(names_H).astype(str))
    if (qpos < 0).any():
        return None, f"{(qpos < 0).sum()} query names not found in {path}"

    def read(pos, cols=None, chunk=20000):
        o = np.argsort(pos)
        sp_ = pos[o]
        parts = []
        for i in range(0, len(sp_), chunk):
            ii = sp_[i:i + chunk]
            try:
                X = a.X[ii]
            except Exception:
                X = a[ii].to_memory().X
            X = sp.csr_matrix(X, dtype=np.float32)
            lib = np.asarray(X.sum(1)).ravel()
            X = sp.diags((1e4 / np.maximum(lib, 1)).astype(np.float32)) @ X
            X.data = np.log1p(X.data)
            parts.append(X[:, cols] if cols is not None else X)
        X = sp.vstack(parts).tocsr()
        inv = np.empty_like(o)
        inv[o] = np.arange(len(o))
        return X[inv]

    Xr = read(rs)
    m = np.asarray(Xr.mean(0)).ravel()
    v = np.asarray(Xr.multiply(Xr).mean(0)).ravel() - m ** 2
    expd = m > 0
    disp = np.full(len(m), -np.inf)
    disp[expd] = np.log(np.maximum(v[expd], 1e-12) / m[expd])
    bins = pd.cut(m[expd], 20, labels=False)
    z = np.full(len(m), -np.inf)
    dz = pd.Series(disp[expd]).groupby(bins)
    mu_b, sd_b = dz.transform("mean").values, dz.transform("std").fillna(1.0).replace(0, 1.0).values
    z[expd] = (disp[expd] - mu_b) / sd_b
    hvg = np.sort(np.argsort(-z)[:min(2000, int(expd.sum()))])
    Xr = Xr[:, hvg].toarray()
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-6
    Zr = np.clip((Xr - mu) / sd, -10, 10)
    pca = PCA(n_components=min(50, Zr.shape[1] - 1, Zr.shape[0] - 1), svd_solver="randomized", random_state=0).fit(Zr)
    Rr = pca.transform(Zr)
    del Xr, Zr
    Xq = read(qpos, cols=hvg)
    Rq = np.vstack([pca.transform(np.clip((Xq[i:i + 20000].toarray() - mu) / sd, -10, 10))
                    for i in range(0, Xq.shape[0], 20000)])
    s = knn_mean_dist(Rr, Rq, k, jobs)
    np.savez_compressed(cache, names=np.asarray(names_H).astype(str), score=s)
    return s, f"computed ({len(rs)} ref cells, {len(hvg)} HVGs, {pca.n_components_} PCs)"


# ------------------------------------------------------------------------------------------ design
def split_donors(dons, nov_frac, rng):
    """Matched-pair stratification on the donor's held-out-type fraction: donors sorted by that fraction are
    paired, one donor of each pair becomes a pseudo-case (odd donor out: coin flip)."""
    o = np.lexsort((rng.random(len(dons)), nov_frac))
    case = np.zeros(len(dons), bool)
    for i in range(0, len(o) - 1, 2):
        case[o[i + int(rng.random() < 0.5)]] = True
    if len(o) % 2:
        case[o[-1]] = rng.random() < 0.5
    return case


def metrics(score_case, flag_case, flag_ctrl, truth, d_case, d_ctrl, alpha_type):
    n_nov = int(truth.sum())
    nflag = int(flag_case.sum())
    tp = int((flag_case & truth).sum())
    F = float(flag_case.mean())
    est = (F - ALPHA) / (1 - ALPHA) if alpha_type else F
    pi = float(truth.mean())
    dfc = pd.DataFrame(dict(d=d_case, f=flag_case, t=truth)).groupby("d").mean()
    dfo = pd.DataFrame(dict(d=d_ctrl, f=flag_ctrl, t=np.zeros(len(d_ctrl)))).groupby("d").mean()
    Fd = float(dfc.f.mean())
    est_d = (Fd - ALPHA) / (1 - ALPHA) if alpha_type else Fd

    def rho(df_):
        if len(df_) < 3 or df_.f.nunique() < 2 or df_.t.nunique() < 2:
            return np.nan
        return float(spearmanr(df_.f, df_.t)[0])
    return dict(auroc=auroc(score_case, truth) if score_case is not None else np.nan,
                n_case=int(len(truth)), n_novel=n_nov, pi=pi, n_flag=nflag, flag_rate=F,
                precision=tp / nflag if nflag else np.nan, recall=tp / max(n_nov, 1),
                fdr=(nflag - tp) / nflag if nflag else 0.0, fa_ctrl=float(flag_ctrl.mean()),
                fa_ctrl_donor_max=float(dfo.f.max()), est=est, bias=est - pi, ratio=est / pi if pi > 0 else np.nan,
                pi_donor=float(dfc.t.mean()), est_donor=est_d, bias_donor=est_d - float(dfc.t.mean()),
                rho_donor=rho(dfc), rho_donor_all=rho(pd.concat([dfc, dfo])))


# ------------------------------------------------------------------------------------------ main
rows, meta = [], {"settings": vars(A), "alpha": ALPHA, "q_bh": Q_BH, "runs": {}}
for ti, tag in enumerate(A.tags):
    if "_ho-" not in tag:
        log("skip (not a held-out run):", tag)
        continue
    holdout = tag.split("_ho-")[1].split("_")[0].replace("~", " ")
    pca_cache = None
    for seed in A.seeds:
        fc = f"{A.res2}/{tag}_seed{seed}_cells.npz"
        fe = f"{A.emb}/{tag}_seed{seed}_full.npz"
        if not (os.path.exists(fc) and os.path.exists(fe)):
            log("missing", fc if not os.path.exists(fc) else fe)
            continue
        C = np.load(fc, allow_pickle=True)
        E = np.load(fe, allow_pickle=True)
        y = np.asarray(C["y"]).astype(str)
        cond = np.asarray(C["cond"]).astype(str)
        donor_all = np.asarray(C["batch"]).astype(str)
        if "names" in C.files and "tgt_names" in E.files:
            assert np.array_equal(np.asarray(C["names"]).astype(str), np.asarray(E["tgt_names"]).astype(str)), \
                "cells npz and emb full npz are not in the same order"
        H = np.where(cond == "normal")[0]
        nov_all = y == holdout
        if not nov_all.any():
            nov_all = np.char.lower(y) == holdout.lower()
        if not nov_all[H].any():
            log("no held-out-type cells among healthy donors for", tag, holdout)
            continue
        nH = len(H)
        donor = donor_all[H]
        novel = nov_all[H]
        names_H = np.asarray(C["names"]).astype(str)[H] if "names" in C.files else np.asarray(E["tgt_names"]).astype(str)[H]
        log(f"{tag} seed {seed}: {nH} healthy query cells, {len(np.unique(donor))} donors, "
            f"{novel.sum()} {holdout} cells ({novel.mean():.3%})")
        # ---- per-cell scores (split-independent)
        tgt_U = np.asarray(E["tgt_U"])[H].astype(np.float32)
        S = {"novelty": np.asarray(C["novelty"])[H], "novelty_U": np.asarray(C["novelty_U"])[H],
             "p_ref": np.asarray(C["p_unknown_bonf"])[H],
             "unc_scanvi": 1 - np.asarray(C["scanvi_maxprob"])[H], "unc_pmaxS": 1 - np.asarray(C["pmax_S"])[H]}
        S["pss_U"] = knn_mean_dist(np.asarray(E["ref_U"]).astype(np.float32), tgt_U, A.k_pss, A.jobs)
        log("scPSS-U scores done")
        use_pca = A.pca is not False
        pca_note = "disabled"
        if use_pca:
            if pca_cache is None:
                pca_cache = pca_pss_scores(tag, holdout, names_H, A.k_pss, A.pca_ref, A.jobs)
            s_pca, pca_note = pca_cache
            if s_pca is not None and len(s_pca) == nH:
                S["pss_pca"] = s_pca
            log("scPSS-PCA:", pca_note)
            if A.pca is True and "pss_pca" not in S:
                log("WARNING --pca requested but unavailable:", pca_note)
        # ---- Milo universe and precomputed kNN
        rng_u = np.random.default_rng(1000 + seed)
        uni = np.sort(rng_u.choice(nH, min(A.milo_max, nH), replace=False))
        Kbig = min(2 * A.k_milo, len(uni))
        dK, nbK = NearestNeighbors(n_neighbors=Kbig, n_jobs=A.jobs).fit(tgt_U[uni]).kneighbors(tgt_U[uni])
        log(f"Milo universe {len(uni)} cells, kNN({Kbig}) done")
        dons = np.unique(donor)
        dcode = pd.Index(dons).get_indexer(donor)
        nov_frac_d = np.bincount(dcode, weights=novel, minlength=len(dons)) / np.bincount(dcode, minlength=len(dons))
        run_meta = dict(n_healthy_cells=int(nH), n_donors=int(len(dons)), n_novel=int(novel.sum()),
                        novel_frac=float(novel.mean()), holdout=holdout, pca=pca_note, milo_universe=int(len(uni)),
                        milo=[])
        for r in range(A.reps):
            rng_r = np.random.default_rng([r, 7919, ti])  # same split for every seed of a tag (paired)
            case_d = split_donors(dons, nov_frac_d, rng_r)
            is_case = case_d[dcode]
            u = rng_r.random(nH)                          # nested dilution: f=0.2 cells are a subset of f=0.5
            for f in A.dilutions:
                present = ~novel | (is_case & (u < f))
                ci_ = np.where(present & is_case)[0]
                co_ = np.where(present & ~is_case)[0]
                truth = novel[ci_]
                dca, dco = donor[ci_], donor[co_]
                rng_m = np.random.default_rng([seed, r, int(f * 1000), ti])
                res = {}

                def add(name, sc_case, fl_case, fl_ctrl, alpha_type=True):
                    res[name] = metrics(sc_case, fl_case, fl_ctrl, truth, dca, dco, alpha_type)

                pers = {}
                for nm, key in (("ARCANE-ctrl", "novelty"), ("ARCANE-ctrl-U", "novelty_U"),
                                ("scPSS-U", "pss_U"), ("scPSS-PCA", "pss_pca")):
                    if key not in S:
                        continue
                    # ARCANE: exactly control_recal (<= 300 cells per control donor); scPSS: all control cells
                    pc, po, per = ctrl_calibrated_p(S[key], ci_, co_, donor, np.random.default_rng([seed, r, ti]),
                                                    A.max_per_donor, balanced=nm.startswith("ARCANE"))
                    pers[nm] = per
                    add(nm, S[key][ci_], pc <= ALPHA, po <= ALPHA)
                    add(nm + " BH", S[key][ci_], bh(pc) <= Q_BH, bh(po) <= Q_BH, alpha_type=False)
                pr = S["p_ref"]
                add("ARCANE-ref", 1 - pr[ci_], pr[ci_] <= ALPHA, pr[co_] <= ALPHA)
                add("ARCANE-ref BH", 1 - pr[ci_], bh(pr[ci_]) <= Q_BH, bh(pr[co_]) <= Q_BH, alpha_type=False)
                for nm, key in (("1-scANVI maxprob", "unc_scanvi"), ("1-pmax_S", "unc_pmaxS")):
                    fcs, fco = quantile_flags(S[key], ci_, co_, donor, np.random.default_rng([seed, r, ti, 5]))
                    add(nm, S[key][ci_], fcs, fco)
                # Milo on the universe cells present in this replicate
                pres_u = present[uni]
                cells_m = uni[pres_u]
                Zp = tgt_U[cells_m]
                idx, dist, tree, n_fix = milo_present_knn(nbK, dK, pres_u, Zp, A.k_milo, A.jobs)
                dm = pd.Index(dons).get_indexer(donor[cells_m])
                check = (r == 0 and f == A.dilutions[0])
                fl, scm, info = milo_da(Zp, idx, dist, tree, dm, case_d, A.k_milo, A.milo_prop, rng_m, check=check)
                info.update(rep=r, f=f, n_cells=int(len(cells_m)), n_knn_recomputed=n_fix)
                run_meta["milo"].append(info)
                cm = is_case[cells_m]
                tm = novel[cells_m][cm]
                res["Milo"] = metrics(scm[cm], fl[cm], fl[~cm], tm, donor[cells_m][cm], donor[cells_m][~cm], False)
                for nm, mt in res.items():
                    rows.append(dict(tag=tag, holdout=holdout, seed=seed, rep=r, f=f, method=nm,
                                     n_case_donors=int(case_d.sum()), n_ctrl_donors=int((~case_d).sum()),
                                     cells_per_ctrl_donor=pers.get(nm.replace(" BH", ""), np.nan), **mt))
            if r == 0 or (r + 1) % 5 == 0:
                log(f"  rep {r + 1}/{A.reps} done; Milo nhoods {info['n_nhoods']}, sig up {info['n_sig_up']}")
        meta["runs"][f"{tag}_seed{seed}"] = run_meta
        pd.DataFrame(rows).to_csv(f"{OUT}/per_replicate.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/per_replicate.csv", index=False)
if df.empty:
    raise SystemExit("no runs found")
MET = ["auroc", "recall", "precision", "fdr", "fa_ctrl", "flag_rate", "est", "pi", "bias", "ratio", "est_donor",
       "pi_donor", "bias_donor", "rho_donor", "rho_donor_all"]
order = ["ARCANE-ctrl", "ARCANE-ctrl BH", "ARCANE-ctrl-U", "ARCANE-ctrl-U BH", "ARCANE-ref", "ARCANE-ref BH",
         "scPSS-U", "scPSS-U BH", "scPSS-PCA", "scPSS-PCA BH", "Milo", "1-scANVI maxprob", "1-pmax_S"]
agg = {}
key_lines = []
for (tag, f), g in df.groupby(["tag", "f"], sort=False):
    for m in [m for m in order if m in set(g.method)]:
        h = g[g.method == m]
        st = {c: dict(mean=float(h[c].mean()), sd=float(h[c].std(ddof=1)) if len(h) > 1 else 0.0) for c in MET}
        st["n"] = int(len(h))
        agg.setdefault(tag, {}).setdefault(f"f={f:g}", {})[m] = st

        def ms(c, d=3):
            return f"{st[c]['mean']:.{d}f}±{st[c]['sd']:.{d}f}"
        key_lines.append(f"KEY {tag} f={f:g} {m:<17s} AUC {ms('auroc')} | power {ms('recall')} | FDR {ms('fdr')} | "
                         f"FA_ctrl {ms('fa_ctrl')} | est {ms('est')} vs pi {st['pi']['mean']:.3f} "
                         f"(bias {st['bias']['mean']:+.3f}, ratio {st['ratio']['mean']:.2f}) | "
                         f"rho_donor {ms('rho_donor', 2)} | n={st['n']}")
meta["aggregate"] = agg
meta["notes"] = [
    "est = (F - 0.05)/(1 - 0.05) for alpha-level flags (p<=0.05 or 95th-percentile threshold); for BH and Milo "
    "est is the raw discovery fraction F among pseudo-case cells.",
    "pi = true novel fraction among pseudo-case cells (cell-pooled); *_donor versions average over case donors.",
    "FA_ctrl: control-calibrated methods use leave-one-donor-out calibration for pseudo-control cells; "
    "ARCANE-ref uses the stored reference-calibrated p-values; BH variants apply BH within the control cells; "
    "Milo: fraction of control cells in significant case-enriched neighbourhoods.",
    "Milo metrics are computed on its <= milo_max cell universe; others on all healthy query cells.",
    "rho_donor: Spearman of per-donor flag rate vs true novel fraction among pseudo-case donors; "
    "rho_donor_all also includes pseudo-control donors (true fraction 0).",
    "scPSS-U uses the same statistic as ARCANE novelty_U (mean k=15 reference-neighbour distance in the scVI "
    "latent, up to a constant), so the two differ only by calibration (scPSS: all pseudo-control cells; ARCANE: "
    "<= 300 cells per control donor). scPSS-PCA is the expression-space variant closer to the published method.",
    "Uncertainty baselines: ties broken at random before the 95th-percentile threshold; AUROC uses raw scores.",
    "Splits are matched pairs on the donor's held-out-type fraction; identical across seeds of a tag.",
]
meta["runtime_min"] = (time.time() - T0) / 60
json.dump(meta, open(f"{OUT}/summary.json", "w"), indent=1, default=lambda o: None if o is None else str(o))
open(f"{OUT}/key.txt", "w").write("\n".join(key_lines) + "\n")
print("\n".join(key_lines))
log("done; wrote", OUT)
