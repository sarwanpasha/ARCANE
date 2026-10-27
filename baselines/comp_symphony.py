"""(c) Symphony (Kang et al. 2021, Nat Commun) via symphonypy (Python port).

Usage: python comp_symphony.py <ds> <seed> [holdout] [--n_hvg 2000] [--n_pcs 30] [--k 15]
Input : data/std/<ds>.h5ad (raw counts).
Pipeline (symphonypy documented usage): reference normalize_total(1e5) + log1p, HVGs per batch
(n_top_genes), scale(max_value=10), PCA (30 PCs, zero_center=False), Harmony on batch
(sp.pp.harmony_integrate) -> Symphony reference; query normalize_total(1e5) + log1p,
sp.tl.map_embedding (query batch = donor), sp.tl.per_cell_confidence (Symphony mapping score =
weighted Mahalanobis distance to reference clusters; higher = worse).
Label transfer: kNN (k=15, majority vote) in the harmonized PC space (Symphony's standard kNN
transfer; computed here directly so the output does not depend on symphonypy obs key names).
Output: res/comp/Symphony_<tag>_seed<s>.npz
  pred = kNN label (no reject); score = Symphony per-cell mapping distance.
  variants: score__knn_dist (mean kNN distance), pred__reject/score__reject (scHPL-style reject:
  mean kNN distance > 99th percentile of reference leave-one-out distances),
  pred__sp_knn (symphonypy transfer_labels_kNN, if it runs).
"""
import inspect
import numpy as np
from comp_common import (parse_args, run_main, load_std, save_result, Timer, versions, knn_transfer,
                         loo_mean_dist, nn_index, UNKNOWN)

METHOD = "Symphony"


def extra(ap):
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--n_pcs", type=int, default=30)
    ap.add_argument("--k", type=int, default=15)


def main(a):
    import scanpy as sc
    import symphonypy as sp
    T = Timer()
    np.random.seed(a.seed)
    ref, qry = load_std(a.ds, a.holdout_label)
    qnames = qry.obs_names.values.astype(str).copy()
    sc.pp.normalize_total(ref, target_sum=1e5)
    sc.pp.log1p(ref)
    try:
        sc.pp.highly_variable_genes(ref, batch_key="batch", n_top_genes=a.n_hvg)
    except Exception as e:  # noqa  (e.g. batches with too few cells)
        print("[Symphony] per-batch HVG failed, using pooled HVG:", repr(e), flush=True)
        sc.pp.highly_variable_genes(ref, n_top_genes=a.n_hvg)
    ref.raw = ref
    ref = ref[:, ref.var.highly_variable].copy()
    sc.pp.scale(ref, max_value=10)
    sc.pp.pca(ref, n_comps=a.n_pcs, zero_center=False, random_state=a.seed)
    hkw = {"random_state": a.seed} if "random_state" in inspect.signature(sp.pp.harmony_integrate).parameters else {}
    sp.pp.harmony_integrate(ref, key="batch", **hkw)
    T.mark("reference")
    sc.pp.normalize_total(qry, target_sum=1e5)
    sc.pp.log1p(qry)
    sp.tl.map_embedding(qry, ref, key="batch")
    T.mark("map_embedding")
    try:
        sp.tl.per_cell_confidence(qry, ref)
        col = next(c for c in qry.obs.columns if "per_cell" in c)
        conf = qry.obs[col].values.astype(float)
    except Exception as e:  # noqa
        print("[Symphony] per_cell_confidence failed:", repr(e), flush=True)
        conf = None
    T.mark("confidence")
    rep = "X_pca_harmony"
    if rep not in ref.obsm or rep not in qry.obsm:
        raise KeyError(f"{rep} missing: ref obsm {list(ref.obsm)}, query obsm {list(qry.obsm)}")
    Zr, Zq = np.asarray(ref.obsm[rep], np.float32), np.asarray(qry.obsm[rep], np.float32)
    yr = ref.obs.label.values.astype(str)
    idx = nn_index(Zr, a.k + 1)
    pred, P, d, classes = knn_transfer(Zr, yr, Zq, a.k, index=idx)
    thr = float(np.percentile(loo_mean_dist(Zr, a.k, index=idx), 99))
    rej = pred.astype(object).copy()
    rej[d > thr] = UNKNOWN
    T.mark("knn")
    variants = {"score__knn_dist": d, "pred__reject": rej.astype(str), "score__reject": d,
                "score__1-maxvote": 1 - P.max(1)}
    try:
        sp.tl.transfer_labels_kNN(qry, ref, "label")
        variants["pred__sp_knn"] = qry.obs["label"].values.astype(str)
    except Exception as e:  # noqa
        print("[Symphony] transfer_labels_kNN failed:", repr(e), flush=True)
    if not np.array_equal(qry.obs_names.values.astype(str), qnames):
        raise ValueError("query order changed during mapping")
    meta = dict(n_hvg=a.n_hvg, n_pcs=a.n_pcs, k=a.k, reject_threshold=thr,
                score="symphony per-cell mapping distance" if conf is not None else "mean kNN distance (fallback)",
                versions=versions(["symphonypy", "harmonypy", "scanpy", "anndata", "numpy"]))
    save_result(METHOD, a.tag, a.seed, qnames, pred, conf if conf is not None else d,
                variants=variants, meta=meta, timer=T)


if __name__ == "__main__":
    args = parse_args(__doc__, extra)
    run_main(METHOD, args, main)
