"""CellTypist baseline: logistic regression on log-normalized reference expression.

Usage: python celltypist_run.py <ds> <seed> [holdout] -> emb/<tag>_seed<s>_celltypist.npz
"""
import os
import sys
import sys
import numpy as np
import scanpy as sc
import anndata as ad
import celltypist

ds, seed = sys.argv[1], int(sys.argv[2])
holdout = sys.argv[3] if len(sys.argv) > 3 else None
tag = ds + (f"_ho-{holdout}" if holdout else "")
tag = tag.replace(" ", "~").replace("/", "-")
out = f"emb/{tag}_seed{seed}_celltypist.npz"
a = ad.read_h5ad(f"data/std/{ds}.h5ad")
hvg = None
if (a.obs.role == "ref").sum() > 300000:
    # large references: after normalization, keep the 5000 most variable genes of the reference so that
    # the dense scaling inside CellTypist fits in memory
    hv = sc.pp.highly_variable_genes(a[a.obs.role == "ref"], n_top_genes=5000, flavor="seurat_v3", inplace=False)
    hvg = hv.index[hv.highly_variable]
if hvg is not None:
    # CellTypist requires log1p expression normalized to 10,000 counts per cell, so normalize within the genes kept
    a = a[:, hvg].copy()
    print("restricted to", a.n_vars, "genes", flush=True)
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
ref, qry = a[a.obs.role == "ref"].copy(), a[a.obs.role == "query"].copy()
if holdout:
    ref = ref[ref.obs.label != holdout.replace("~", " ")].copy()
np.random.seed(seed)
model = celltypist.train(ref, labels="label", feature_selection=True, use_SGD=True, mini_batch=True,
                         n_jobs=8, max_iter=20, epochs=10, balance_cell_type=False)
pred = celltypist.annotate(qry, model=model, majority_voting=False)
np.savez_compressed(out, pred=pred.predicted_labels["predicted_labels"].values.astype(str),
                    y_tgt=qry.obs.label.values.astype(str), tgt_names=qry.obs_names.values.astype(str))
print("saved", out)
