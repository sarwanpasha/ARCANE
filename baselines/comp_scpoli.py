"""(b) scPoli (De Donno et al. 2023, Nat Methods) via scArches: reference training with labels,
query mapping by architecture surgery, prototype-based label transfer with uncertainty.

Usage: python comp_scpoli.py <ds> <seed> [holdout] [--split full|cf0|cf1] [--epochs 50] [--pre 40]
Input : data/std/<ds>.h5ad; optionally emb/<tag>_seed<s>_<split>.npz (scVI/scANVI from train.py).
Output:
  split full: res/comp/scPoli_<tag>_seed<s>.npz
      pred = scPoli prototype classification (classify, get_prob=False); score = scPoli uncertainty
      (distance-to-prototype based, log_distance=True, unscaled).
      variants: pred__reject (unknown if uncertainty > 99th percentile of the reference cells'
      own uncertainties), score__proto_dist (Euclidean distance to the nearest class centroid of the
      reference scPoli latent).
  every split: emb/<tag>_scpoli_seed<s>_<split>.npz = the train.py npz of that split with ref_S /
      tgt_S replaced by the scPoli latent (ref_U/tgt_U stay scVI, scanvi_prob stays scANVI), so
      ARCANE can be run on the pair (scVI, scPoli) with
          python evaluate2.py <tag>_scpoli <seed>
      after running this script for --split full, cf0 and cf1.
      If the train.py npz is missing, emb/<tag>_scpoli-only_seed<s>_<split>.npz holds the scPoli
      latent alone (ref_S, tgt_S, y_ref, tgt_names, tgt_role).
Reference/target construction replicates train.py exactly (holdout removal, folds, 'cal_' name
prefix for held-out fold cells, 2000 seurat_v3 HVGs computed on the training cells).
"""
import inspect
import os
import numpy as np
import pandas as pd
from comp_common import (parse_args, run_main, save_result, Timer, versions, UNKNOWN, check_cwd)

METHOD = "scPoli"


def extra(ap):
    ap.add_argument("--split", default="full", choices=["full", "cf0", "cf1"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--pre", type=int, default=40, help="pretraining epochs (no prototype loss)")
    ap.add_argument("--latent", type=int, default=10)
    ap.add_argument("--eta_ref", type=float, default=5.0)
    ap.add_argument("--eta_query", type=float, default=10.0)


def build(a):
    """Replicates train.py: returns (train AnnData, target AnnData, target role array)."""
    import anndata as ad
    import scanpy as sc
    x = ad.read_h5ad(f"data/std/{a.ds}.h5ad")
    for c in ("label", "batch", "role", "fold"):
        if c not in x.obs:
            raise KeyError(f"data/std/{a.ds}.h5ad lacks obs column {c}")
    ref = x[x.obs.role == "ref"].copy()
    if a.holdout_label:
        ref = ref[ref.obs.label != a.holdout_label].copy()
    q_all = x[x.obs.role == "query"].copy()
    del x
    if a.split == "full":
        train, tgt = ref, q_all
        role = np.array(["query"] * tgt.n_obs)
    else:
        f = int(a.split[-1])
        train, held = ref[ref.obs.fold != f].copy(), ref[ref.obs.fold == f].copy()
        role = np.array(["cal"] * held.n_obs + ["query"] * q_all.n_obs)
        held.obs_names = ["cal_" + n for n in held.obs_names]
        tgt = ad.concat([held, q_all], join="outer", merge="first")
        tgt.obs["label"] = tgt.obs["label"].astype(str)
    sc.pp.highly_variable_genes(train, n_top_genes=2000, flavor="seurat_v3")
    hvg = train.var_names[train.var.highly_variable]
    train, tgt = train[:, hvg].copy(), tgt[:, hvg].copy()
    for z in (train, tgt):
        z.obs["batch"] = z.obs["batch"].astype(str)
        z.obs["label"] = z.obs["label"].astype(str)
        z.X = (z.X.toarray() if hasattr(z.X, "toarray") else np.asarray(z.X)).astype(np.float32)
    return train, tgt, role


def call(fn, **kw):
    """Call fn with the subset of kw it accepts (scArches versions differ)."""
    sig = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in sig.values()):
        return fn(**kw)
    return fn(**{k: v for k, v in kw.items() if k in sig})


def classify(model, adata):
    res = call(model.classify, adata=adata, get_prob=False, scale_uncertainties=False)
    if isinstance(res, tuple):  # very old scArches: (preds, uncert)
        return np.asarray(res[0]).astype(str), np.asarray(res[1], float)
    r = res["label_s"] if "label_s" in res else next(iter(res.values()))
    return np.asarray(r["preds"]).astype(str), np.asarray(r["uncert"], float).ravel()


def main(a):
    import torch
    from scarches.models.scpoli import scPoli
    T = Timer()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    train, tgt, role = build(a)
    T.mark("data")
    train.obs["label_s"] = train.obs["label"].values
    y_tgt = tgt.obs["label"].values.astype(str)
    tgt.obs["label_s"] = "Unknown"
    if "Unknown" in set(train.obs.label_s):
        raise ValueError("a reference label is literally 'Unknown'; rename it")
    print(f"[scPoli] {a.tag} {a.split}: train {train.shape}, target {tgt.shape}", flush=True)
    es = {"early_stopping_metric": "val_prototype_loss", "mode": "min", "threshold": 0, "patience": 20,
          "reduce_lr": True, "lr_patience": 13, "lr_factor": 0.1}
    m = call(scPoli, adata=train, condition_keys="batch", cell_type_keys="label_s", embedding_dims=5,
             latent_dim=a.latent, recon_loss="nb")
    call(m.train, n_epochs=a.epochs, pretraining_epochs=a.pre, early_stopping_kwargs=es, eta=a.eta_ref)
    T.mark("train_ref")
    try:
        qm = call(scPoli.load_query_data, adata=tgt, reference_model=m, labeled_indices=[],
                  unknown_ct_names=["Unknown"])
    except (KeyError, ValueError) as e:
        # some scArches versions cannot encode an unseen label even for unlabeled query cells;
        # the label column is ignored for training because labeled_indices=[]
        print("[scPoli] load_query_data with 'Unknown' failed (", repr(e), "); retrying with a reference label",
              flush=True)
        tgt.obs["label_s"] = train.obs["label_s"].value_counts().index[0]
        qm = call(scPoli.load_query_data, adata=tgt, reference_model=m, labeled_indices=[])
    if isinstance(qm, tuple):
        qm = qm[0]
    call(qm.train, n_epochs=min(a.epochs, 20), pretraining_epochs=min(a.pre, 15), eta=a.eta_query)
    T.mark("surgery")
    lat_ref = np.asarray(call(qm.get_latent, adata=train, mean=True), np.float32)
    lat_tgt = np.asarray(call(qm.get_latent, adata=tgt, mean=True), np.float32)
    T.mark("latent")
    y_ref = train.obs["label"].values.astype(str)
    # ------------------------------------------------------------------ ARCANE-compatible export
    os.makedirs("emb", exist_ok=True)
    src = f"emb/{a.tag}_seed{a.seed}_{a.split}.npz"
    if os.path.exists(src):
        d = dict(np.load(src, allow_pickle=True))
        ok = (len(d["y_ref"]) == len(y_ref) and np.array_equal(np.asarray(d["y_ref"]).astype(str), y_ref)
              and np.array_equal(np.asarray(d["tgt_names"]).astype(str), tgt.obs_names.values.astype(str)))
        if not ok:
            raise ValueError(f"cell order of {src} differs from the reconstructed train/target sets; "
                             "cannot build the (scVI, scPoli) pair")
        d["ref_S"], d["tgt_S"] = lat_ref, lat_tgt
        d["view_S"] = np.array("scPoli")
        dst = f"emb/{a.tag}_scpoli_seed{a.seed}_{a.split}.npz"
    else:
        print(f"[scPoli] WARNING {src} missing: writing scPoli-only latent", flush=True)
        d = dict(ref_S=lat_ref, tgt_S=lat_tgt, y_ref=y_ref, y_tgt=y_tgt, tgt_names=tgt.obs_names.values.astype(str),
                 tgt_role=role, b_ref=train.obs.batch.values.astype(str), b_tgt=tgt.obs.batch.values.astype(str))
        dst = f"emb/{a.tag}_scpoli-only_seed{a.seed}_{a.split}.npz"
    np.savez_compressed(dst, **d)
    print("[scPoli] wrote", dst, flush=True)
    if a.split != "full":
        save_meta_only(a, T)
        return
    # ------------------------------------------------------------------ predictions (full split)
    pred, unc = classify(qm, tgt)
    T.mark("classify_query")
    try:
        _, unc_ref = classify(qm, train)
        thr = float(np.percentile(unc_ref, 99))
    except Exception as e:  # noqa
        print("[scPoli] reference classify failed:", repr(e), flush=True)
        thr = np.inf
    classes = np.array(sorted(set(y_ref)))
    cent = np.stack([lat_ref[y_ref == c].mean(0) for c in classes])
    dcent = np.concatenate([np.sqrt(((lat_tgt[i:i + 20000, None, :] - cent[None]) ** 2).sum(-1)).min(1)
                            for i in range(0, len(lat_tgt), 20000)])
    rej = pred.astype(object).copy()
    rej[unc > thr] = UNKNOWN
    meta = dict(split=a.split, epochs=a.epochs, pretraining_epochs=a.pre, latent=a.latent,
                eta_ref=a.eta_ref, eta_query=a.eta_query, reject_threshold=thr, export=dst,
                versions=versions(["scarches", "torch", "scanpy", "anndata", "numpy"]))
    save_result(METHOD, a.tag, a.seed, tgt.obs_names.values, pred, unc,
                variants={"pred__reject": rej.astype(str), "score__proto_dist": dcent},
                meta=meta, timer=T)


def save_meta_only(a, T):
    import json
    from comp_common import resources, out_path
    os.makedirs("res/comp", exist_ok=True)
    json.dump(dict(method=METHOD, tag=a.tag, seed=a.seed, split=a.split, timing=T.marks, **resources()),
              open(out_path(METHOD, a.tag, a.seed, f"{a.split}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    args = parse_args(__doc__, extra)
    run_main(METHOD if args.split == "full" else f"{METHOD}-{args.split}", args, main)
