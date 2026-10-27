"""Full Perez et al. (2022) lupus cohort: source h5ad from CELLxGENE (keeps author metadata),
capped at 2000 cells per donor. Writes data/raw/lupusfull.h5ad with raw counts and all obs columns."""
import os
import numpy as np
import anndata as ad
import cellxgene_census

DID = "218acb0f-9f2f-4f76-b90b-15a4b7c7f629"
src = "data/raw/lupus_source.h5ad"
if not os.path.exists(src):
    cellxgene_census.download_source_h5ad(DID, to_path=src, census_version="2025-11-08")
a = ad.read_h5ad(src, backed="r")
obs = a.obs
print("OBS_COLUMNS", obs.columns.tolist(), flush=True)
for c in obs.columns:
    if obs[c].nunique() < 60:
        print("COL", c, obs[c].value_counts().head(40).to_dict(), flush=True)
rng = np.random.default_rng(0)
idx = []
for d, ii in obs.groupby("donor_id", observed=True).indices.items():
    idx += rng.choice(ii, min(2000, len(ii)), replace=False).tolist()
idx = np.sort(np.array(idx))
b = a[idx].to_memory()
if b.raw is not None:
    b = ad.AnnData(X=b.raw.X, obs=b.obs, var=b.raw.var)
if "feature_name" in b.var:
    b.var_names = b.var["feature_name"].astype(str).values
b.var.index.name = None
b.var_names_make_unique()
b.write_h5ad("data/raw/lupusfull.h5ad")
print("LUPUSFULL_OK", b.shape, b.obs.disease.value_counts().to_dict(), "donors", b.obs.donor_id.nunique(), flush=True)
