"""Train scVI and scANVI on a reference and map target cells with scArches surgery.

--split full : train on all reference cells, target = query cells
--split cf0/cf1 : train on reference fold 1/0, target = reference fold 0/1 (cross-fitting)
--holdout T : remove type T from all reference cells (it stays in the query)
--downsample T:N : keep N reference cells of type T
--noise uniform:R | swap:R : corrupt a fraction R of reference labels (uniform over other types, or
                            systematic swaps to a sibling type with the same parent)
--coarse : train scANVI on the parent labels (kNN evidence still uses the fine labels)
--partial F : keep reference labels for scANVI training on only a fraction F of reference cells
Symmetric cross-fitting: for cf0/cf1 the target is the held-out fold together with the query cells,
mapped in one surgery, so calibration and query cells are embedded by the same fold model.
Output npz: embeddings of training reference cells and target cells, labels, batches,
scANVI class probabilities on targets, target obs names.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scvi
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", default="full", help="full or cf<f>")
ap.add_argument("--nfolds", type=int, default=2, help="number of cross-fitting folds (folds of whole batches)")
ap.add_argument("--holdout", default=None)
ap.add_argument("--downsample", default=None)
ap.add_argument("--n_latent", type=int, default=10)
ap.add_argument("--max_epochs", type=int, default=200)
ap.add_argument("--surgery_epochs", type=int, default=0, help="0: min(100, max(30, round(4e6/n_target)))")
ap.add_argument("--outdir", default="emb")
ap.add_argument("--noise", default=None)
ap.add_argument("--coarse", action="store_true")
ap.add_argument("--partial", type=float, default=None)
ap.add_argument("--scanvi_frac", type=float, default=0.1, help="scANVI epochs as a fraction of scVI epochs")
ap.add_argument("--tagsuffix", default="")
ap.add_argument("--query_cond", default=None, help="keep only query cells with this condition (e.g. normal)")
ap.add_argument("--drop_isg", type=int, default=0, help="remove curated IFN/IEG genes plus the N most IFN-correlated genes")
A = ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)
tag = A.ds + (f"_ho-{A.holdout}" if A.holdout else "") + (f"_ds-{A.downsample}" if A.downsample else "")
tag += (f"_noise-{A.noise}" if A.noise else "") + ("_coarse" if A.coarse else "") + \
    (f"_partial-{A.partial}" if A.partial else "") + (f"_isgblind{A.drop_isg}" if A.drop_isg else "") + A.tagsuffix
tag = tag.replace(" ", "~").replace("/", "-").replace(":", "-")
out = f"{A.outdir}/{tag}_seed{A.seed}_{A.split}.npz"
if os.path.exists(out):
    raise SystemExit("exists " + out)
scvi.settings.seed = A.seed
torch.set_float32_matmul_precision("high")
t0 = time.time()

a = ad.read_h5ad(f"data/std/{A.ds}.h5ad")
ref = a[a.obs.role == "ref"].copy()
if A.holdout:
    A.holdout = A.holdout.replace("~", " ")
    ref = ref[ref.obs.label != A.holdout].copy()
if A.downsample:
    typ, n = A.downsample.replace("~", " ").rsplit(":", 1)
    rng = np.random.default_rng(A.seed)
    ii = np.where(ref.obs.label.values == typ)[0]
    drop = np.setdiff1d(ii, rng.choice(ii, min(int(n), len(ii)), replace=False))
    ref = ref[np.setdiff1d(np.arange(ref.n_obs), drop)].copy()
if A.noise:
    mode, rate = A.noise.split(":")
    rate = float(rate)
    rng = np.random.default_rng(1000 + A.seed)
    lab = ref.obs.label.astype(str).values.copy()
    par = ref.obs.parent.astype(str).values
    types = np.unique(lab)
    if mode == "uniform":
        flip = rng.random(len(lab)) < rate
        for i in np.where(flip)[0]:
            lab[i] = rng.choice(types[types != lab[i]])
    else:  # systematic: each type is mislabelled as one fixed sibling (same parent), else nearest-in-name
        sib = {}
        for t in types:
            pt = par[lab == t][0]
            cand = [u for u in types if u != t and par[lab == u][0] == pt] or [u for u in types if u != t]
            sib[t] = sorted(cand)[A.seed % len(cand)]
        flip = rng.random(len(lab)) < rate
        lab[flip] = [sib[t] for t in lab[flip]]
    ref.obs["label_clean"] = ref.obs.label.astype(str).values
    ref.obs["label"] = lab
    print("noise", A.noise, "flipped", int(flip.sum()), flush=True)
q_all = a[a.obs.role == "query"].copy()
if A.query_cond:
    q_all = q_all[q_all.obs.condition.astype(str) == A.query_cond].copy()
if A.split == "full":
    train, tgt = ref, q_all
    role = np.array(["query"] * tgt.n_obs)
else:
    f = int(A.split[2:])
    if A.nfolds != 2:
        # K folds of whole batches, assigned round-robin by decreasing batch size
        sizes = ref.obs.batch.value_counts()
        kf = {b: i % A.nfolds for i, b in enumerate(sizes.index)}
        ref.obs["fold"] = ref.obs.batch.map(kf).astype(int).values
    train, held = ref[ref.obs.fold != f].copy(), ref[ref.obs.fold == f].copy()
    role = np.array(["cal"] * held.n_obs + ["query"] * q_all.n_obs)
    held.obs_names = ["cal_" + x for x in held.obs_names]
    tgt = ad.concat([held, q_all], join="outer", merge="first")
    tgt.obs["label"] = tgt.obs["label"].astype(str)
del a
if A.drop_isg:
    # remove interferon, stress and immediate-early genes before gene selection: the curated sets plus the
    # A.drop_isg genes most correlated with an interferon score across query cells within labels
    import json
    import scipy.sparse as sp
    G = json.load(open("genesets.json"))
    curated = set()
    for k in ["ISG_typeI", "ISG_typeI_core_specific", "IFN_gamma_specific", "SIGLEC1_IFN_monocyte",
              "dissociation_IEG", "heat_shock", "IEG_core_specific"]:
        curated |= set(G[k])
    rng = np.random.default_rng(0)
    sub = q_all[rng.choice(q_all.n_obs, min(60000, q_all.n_obs), replace=False)]
    X = sp.csr_matrix(sub.X, dtype=np.float32)
    X = sp.diags((1e4 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)).astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    X = X.tocsr()
    names = np.array(sub.var_names)
    core = [i for i, g in enumerate(names) if g in set(G["ISG_typeI_core_specific"] + G["IFN_gamma_specific"])]
    Z = X[:, core].toarray()
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    s = Z.mean(1)
    labs = sub.obs.label.astype(str).values
    rs = s.copy()
    for l in np.unique(labs):
        rs[labs == l] -= s[labs == l].mean()
    cov = np.asarray(X.T @ rs).ravel()
    ss = np.asarray(X.multiply(X).sum(0)).ravel()
    for l in np.unique(labs):
        m = labs == l
        ss -= m.sum() * np.asarray(X[m].mean(0)).ravel() ** 2
    r = cov / np.sqrt(np.maximum(ss, 1e-9) * (rs ** 2).sum())
    top = set(names[np.argsort(-r)[:A.drop_isg]])
    drop = np.isin(train.var_names, list(curated | top))
    print("drop_isg: removed", int(drop.sum()), "genes; top correlated", list(names[np.argsort(-r)[:15]]), flush=True)
    np.savetxt(f"{A.outdir}/{tag}_seed{A.seed}_{A.split}_dropped.txt", np.array(train.var_names[drop]), fmt="%s")
    train, tgt = train[:, ~drop].copy(), tgt[:, ~drop].copy()
    del X, sub
sc.pp.highly_variable_genes(train, n_top_genes=2000, flavor="seurat_v3")
hvg = train.var_names[train.var.highly_variable]
train, tgt = train[:, hvg].copy(), tgt[:, hvg].copy()
ep = int(min(A.max_epochs, max(20, round(8e6 / train.n_obs))))
sep = A.surgery_epochs or int(min(100, max(30, round(4e6 / tgt.n_obs))))
bs = 512 if ref.n_obs > 50000 else 128  # larger minibatches for the 2 largest references
print(tag, A.split, "train", train.shape, "target", tgt.shape, "epochs", ep, "surgery", sep, "batch", bs, flush=True)

train.obs["label_s"] = train.obs["parent" if A.coarse else "label"].astype(str).values
if A.partial:
    rng = np.random.default_rng(2000 + A.seed)
    keep = rng.random(train.n_obs) < A.partial
    train.obs.loc[~keep, "label_s"] = "Unknown"
scvi.model.SCVI.setup_anndata(train, batch_key="batch", labels_key="label_s")
mU = scvi.model.SCVI(train, n_latent=A.n_latent)
mU.train(max_epochs=ep, batch_size=bs, enable_progress_bar=False)
t_u = time.time() - t0
mS = scvi.model.SCANVI.from_scvi_model(mU, unlabeled_category="Unknown", labels_key="label_s")
try:
    mS.train(max_epochs=max(20, int(round(ep * A.scanvi_frac))), batch_size=bs, enable_progress_bar=False)
except ValueError as e:  # a final minibatch of 1 labelled cell breaks batch norm; shift the batch size by 3
    if "more than 1 value per channel" not in str(e):
        raise
    print("retrying scANVI with batch size", bs + 3, flush=True)
    mS = scvi.model.SCANVI.from_scvi_model(mU, unlabeled_category="Unknown", labels_key="label_s")
    mS.train(max_epochs=max(20, int(round(ep * A.scanvi_frac))), batch_size=bs + 3, enable_progress_bar=False)
t_s = time.time() - t0 - t_u


def surgery(cls, model):
    q = tgt.copy()
    q.obs["label_s"] = "Unknown"
    cls.prepare_query_anndata(q, model)
    qm = cls.load_query_data(q, model)
    try:
        qm.train(max_epochs=sep, batch_size=bs, plan_kwargs=dict(weight_decay=0.0), enable_progress_bar=False)
    except ValueError as e:
        if "more than 1 value per channel" not in str(e):
            raise
        qm = cls.load_query_data(q, model)
        qm.train(max_epochs=sep, batch_size=bs + 3, plan_kwargs=dict(weight_decay=0.0), enable_progress_bar=False)
    return qm, q


qU, _ = surgery(scvi.model.SCVI, mU)
qS, qa = surgery(scvi.model.SCANVI, mS)
prob = qS.predict(qa, soft=True)
np.savez_compressed(
    out,
    ref_U=mU.get_latent_representation().astype(np.float32),
    ref_S=mS.get_latent_representation().astype(np.float32),
    tgt_U=qU.get_latent_representation().astype(np.float32),
    tgt_S=qS.get_latent_representation().astype(np.float32),
    y_ref=train.obs.label.values.astype(str), y_tgt=tgt.obs.label.values.astype(str),
    y_ref_clean=train.obs.get("label_clean", train.obs.label).values.astype(str), tgt_role=role,
    labeled_ref=(train.obs.label_s.values != "Unknown"),
    runtime=np.array([t_u, t_s, time.time() - t0]),
    gpu_peak_gb=np.array(torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0),
    ram_peak_gb=np.array(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1e6),
    parent_tgt=tgt.obs.parent.values.astype(str), parent_ref=train.obs.parent.values.astype(str),
    b_ref=train.obs.batch.values.astype(str), b_tgt=tgt.obs.batch.values.astype(str),
    cond_tgt=tgt.obs.condition.values.astype(str), tgt_names=tgt.obs_names.values.astype(str),
    scanvi_prob=prob.values.astype(np.float32), scanvi_classes=np.array(prob.columns, dtype=str),
)
print("saved", out, "minutes %.1f" % ((time.time() - t0) / 60), flush=True)
