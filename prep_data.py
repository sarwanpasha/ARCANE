"""Download public datasets and write standardized AnnData files to data/std/<name>.h5ad.

Standardized obs columns: label (fine type), parent (coarse type or label), batch, role
(ref/query), fold (0/1 for reference cells, -1 otherwise), condition (default 'none').
X holds raw counts. Usage: python prep_data.py <name>
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

os.makedirs("data/std", exist_ok=True)
name = sys.argv[1]


def assign_folds(a, ref_batches):
    """Split reference batches into two folds with similar cell counts (greedy)."""
    sizes = a.obs.loc[a.obs.batch.isin(ref_batches), "batch"].value_counts()
    load, fold = [0, 0], {}
    for b, n in sizes.sort_values(ascending=False).items():
        f = int(load[1] < load[0])
        fold[b] = f
        load[f] += n
    a.obs["fold"] = a.obs.batch.map(fold).fillna(-1).astype(int)


def finish(a, ref, qry, label, batch, parent=None, condition=None, extra=False):
    a.obs["label"] = a.obs[label].astype(str)
    a.obs["batch"] = a.obs[batch].astype(str)
    a.obs["parent"] = a.obs[parent].astype(str) if parent else a.obs["label"]
    a.obs["condition"] = a.obs[condition].astype(str) if condition else "none"
    a = a[~a.obs.label.isin(["nan", "NaN", "None", "NA", "", "unknown", "Unknown"])].copy()
    a.obs["role"] = np.where(a.obs.batch.isin(ref), "ref", np.where(a.obs.batch.isin(qry), "query", "none"))
    a = a[a.obs.role != "none"].copy()
    assign_folds(a, ref)
    if "counts" in a.layers:
        a.X = a.layers["counts"]
    a.X = sp.csr_matrix(a.X).astype(np.float32)
    keep = ["label", "parent", "batch", "role", "fold", "condition"]
    if extra:
        keep += [c for c in a.obs.columns if c not in keep]
    o = a.obs[keep].copy()
    for c in o.columns:
        if o[c].dtype == object or str(o[c].dtype) == "category":
            o[c] = o[c].astype(str)
    a = ad.AnnData(X=a.X, obs=o, var=pd.DataFrame(index=a.var_names.astype(str)))
    a.var_names_make_unique()
    a.write_h5ad(f"data/std/{name}.h5ad", compression="gzip")
    print(name, a.shape, a.obs.role.value_counts().to_dict(), "labels", a.obs.label.nunique(),
          "parents", a.obs.parent.nunique(), "folds", a.obs.groupby("fold").size().to_dict(), flush=True)


if name == "pbmc_fine":
    import scvi
    a = scvi.data.pbmc_seurat_v4_cite_seq(save_path="data/raw/")
    finish(a, ["P1", "P5", "P6", "P7", "P8"], ["P2", "P3", "P4"], "celltype.l2", "donor", parent="celltype.l1")
elif name == "spleen_lymph":
    import scvi
    a = scvi.data.spleen_lymph_cite_seq(save_path="data/raw/")
    finish(a, ["SLN111-D1", "SLN208-D1"], ["SLN111-D2", "SLN208-D2"], "cell_types", "batch")
elif name == "heart":
    import scvi
    a = scvi.data.heart_cell_atlas_subsampled(save_path="data/raw/")
    print(a.obs.columns.tolist(), flush=True)
    bcol = "donor" if "donor" in a.obs else "cell_source"
    donors = a.obs[bcol].value_counts()
    ref, qry, tot, run = [], [], donors.sum(), 0
    for d, n in donors.items():  # largest donors first; about 60% of cells to the reference
        (ref if run < 0.6 * tot else qry).append(d)
        if d in ref:
            run += n
    finish(a, ref, qry, "cell_type", bcol)
elif name == "lupus":
    a = ad.read_h5ad("data/raw/lupus.h5ad")
    a.obs["donor"] = a.obs.donor_id.astype(str)
    ref = sorted(a.obs.loc[a.obs.role_hint == "ref", "donor"].unique())
    qry = sorted(a.obs.loc[a.obs.role_hint == "query", "donor"].unique())
    counts = a.obs.cell_type.value_counts()
    a = a[a.obs.cell_type.isin(counts.index[counts >= 50])].copy()
    finish(a, ref, qry, "cell_type", "donor", condition="disease")
elif name == "hlca":
    a = ad.read_h5ad("data/raw/hlca_core.h5ad", backed="r")
    obs = a.obs
    print(obs.columns.tolist(), flush=True)
    lab = "ann_finest_level" if "ann_finest_level" in obs else "cell_type"
    par = "ann_level_2" if "ann_level_2" in obs else None
    bcol = "dataset" if "dataset" in obs else "study"
    sizes = obs[bcol].value_counts()
    chosen = sizes.index[:8].tolist()
    rng = np.random.default_rng(0)
    idx = []
    for b in chosen:
        ii = np.where(obs[bcol].values == b)[0]
        idx += sorted(rng.choice(ii, min(len(ii), 30000), replace=False).tolist())
    a = a[np.sort(idx)].to_memory()
    if "raw" in dir(a) and a.raw is not None:
        a = ad.AnnData(X=a.raw.X, obs=a.obs, var=a.raw.var)
    if "feature_name" in a.var:
        a.var_names = a.var["feature_name"].astype(str)
    finish(a, chosen[:5], chosen[5:8], lab, bcol, parent=par)
elif name == "hlca_full":
    a = ad.read_h5ad("data/raw/hlca_core.h5ad", backed="r")
    obs = a.obs
    lab = "ann_finest_level" if "ann_finest_level" in obs else "cell_type"
    par = "ann_level_2" if "ann_level_2" in obs else None
    bcol = "dataset" if "dataset" in obs else "study"
    sizes = obs[bcol].value_counts()
    # query: every fourth study by size (about a quarter of the cells), reference: the rest
    studies = sizes.index.tolist()
    qry = studies[1::4]
    ref = [s for s in studies if s not in qry]
    print("HLCA_FULL studies", len(studies), "query", qry, flush=True)
    a = a.to_memory()
    if a.raw is not None:
        a = ad.AnnData(X=a.raw.X, obs=a.obs, var=a.raw.var)
    if "feature_name" in a.var:
        a.var_names = a.var["feature_name"].astype(str)
    finish(a, ref, qry, lab, bcol, parent=par)
elif name == "lupusfull":
    a = ad.read_h5ad("data/raw/lupusfull.h5ad")
    o = a.obs
    if "ct_cov" in o and "author_cell_type" in o:  # fine lymphocyte labels, myeloid/other labels from author_cell_type
        ct = o["ct_cov"].astype(str)
        o["fine_label"] = np.where(ct.isin(["nan", "NaN", "None", ""]), o["author_cell_type"].astype(str), ct)
        fine = "fine_label"
    else:
        fine = next((c for c in ["author_cell_type", "cell_type_fine", "celltype", "cg_cov"] if c in o), "cell_type")
    print("fine label column", fine, o[fine].value_counts().to_dict(), flush=True)
    counts = o[fine].value_counts()
    a = a[o[fine].isin(counts.index[counts >= 200])].copy()
    o = a.obs
    o["donor"] = o.donor_id.astype(str)
    coh = next((c for c in ["Processing_Cohort", "processing_cohort", "batch_cov", "Batch", "batch"] if c in o), None)
    anc = next((c for c in ["pop_cov", "self_reported_ethnicity"] if c in o), None)
    dd = o.groupby("donor").agg(disease=("disease", "first"),
                                coh=(coh, "first") if coh else ("disease", "first"),
                                anc=(anc, "first") if anc else ("disease", "first")).reset_index()
    rng = np.random.default_rng(0)
    ref = []
    for _, g in dd[dd.disease == "normal"].groupby(["coh", "anc"]):
        d = g.donor.tolist()
        rng.shuffle(d)
        ref += d[: int(round(0.55 * len(d)))]
    qry = [d for d in dd.donor if d not in ref]
    print("lupusfull ref donors", len(ref), "query donors", len(qry), "cohort col", coh, "ancestry col", anc, flush=True)
    finish(a, ref, qry, fine, "donor", parent="cell_type", condition="disease", extra=True)
else:
    raise SystemExit("unknown dataset " + name)
