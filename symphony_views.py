"""ARCANE with a non-VAE unsupervised view: Symphony (Harmony reference + query mapping) replaces scVI as U.

Usage: python symphony_views.py <ds> <seed> [holdout]   (run in env_comp, which has symphonypy)
For each split (full, cf0, cf1) reads emb/<tag>_seed<s>_<split>.npz from train.py, builds a Symphony reference on
the same training cells, maps the same target cells (held-out fold + query for cf splits), and writes
emb/<tag>_symph_seed<s>_<split>.npz with ref_U / tgt_U replaced by the harmonized PCs. Then
`python evaluate2.py <tag>_symph <seed>` runs ARCANE on (Symphony, scANVI).
"""
import inspect
import os
import sys
import time
import numpy as np
import anndata as ad
import scanpy as sc
import symphonypy as sp

ds, seed = sys.argv[1], int(sys.argv[2])
holdout = sys.argv[3] if len(sys.argv) > 3 else None
tag = ds + (f"_ho-{holdout}" if holdout else "")
N_PCS = 20
a = ad.read_h5ad(f"data/std/{ds}.h5ad")
ref_all = a[a.obs.role == "ref"].copy()
if holdout:
    ref_all = ref_all[ref_all.obs.label != holdout.replace("~", " ")].copy()
q_all = a[a.obs.role == "query"].copy()
del a
for split in ["full", "cf0", "cf1"]:
    src = f"emb/{tag}_seed{seed}_{split}.npz"
    dst = f"emb/{tag}_symph_seed{seed}_{split}.npz"
    if os.path.exists(dst) or not os.path.exists(src):
        print("skip", dst, flush=True)
        continue
    t0 = time.time()
    d = dict(np.load(src, allow_pickle=True))
    if split == "full":
        train, tgt = ref_all.copy(), q_all.copy()
    else:
        f = int(split[2:])
        train = ref_all[ref_all.obs.fold != f].copy()
        held = ref_all[ref_all.obs.fold == f].copy()
        held.obs_names = ["cal_" + x for x in held.obs_names]
        tgt = ad.concat([held, q_all], join="outer", merge="first")
    assert train.n_obs == len(d["y_ref"]) and np.array_equal(train.obs.label.astype(str).values, d["y_ref"]), "reference order"
    assert np.array_equal(tgt.obs_names.values.astype(str), d["tgt_names"].astype(str)), "target order"
    np.random.seed(seed)
    sc.pp.normalize_total(train, target_sum=1e5)
    sc.pp.log1p(train)
    try:
        sc.pp.highly_variable_genes(train, batch_key="batch", n_top_genes=2000)
    except Exception:
        sc.pp.highly_variable_genes(train, n_top_genes=2000)
    train.raw = train
    train = train[:, train.var.highly_variable].copy()
    sc.pp.scale(train, max_value=10)
    sc.pp.pca(train, n_comps=N_PCS, zero_center=False, random_state=seed)
    hkw = {"random_state": seed} if "random_state" in inspect.signature(sp.pp.harmony_integrate).parameters else {}
    sp.pp.harmony_integrate(train, key="batch", **hkw)
    sc.pp.normalize_total(tgt, target_sum=1e5)
    sc.pp.log1p(tgt)
    sp.tl.map_embedding(tgt, train, key="batch")
    assert np.array_equal(tgt.obs_names.values.astype(str), d["tgt_names"].astype(str)), "target order after mapping"
    d["ref_U"] = np.asarray(train.obsm["X_pca_harmony"], np.float32)
    d["tgt_U"] = np.asarray(tgt.obsm["X_pca_harmony"], np.float32)
    d["runtime_symphony"] = np.array(time.time() - t0)
    np.savez_compressed(dst, **d)
    print("saved", dst, "minutes %.1f" % ((time.time() - t0) / 60), flush=True)
    del train, tgt
