"""Download second-disease blood datasets from CELLxGENE and write data/raw/<name>.h5ad.

Usage (from ~/arcane):  python download_second.py [name ...] [--keep-source] [--cap 2000]
Default names: covid_stephenson covid_yoshida  (add ibd_blood / covid_wilk explicitly)

Why the source h5ad and not cellxgene_census.get_anndata: Census obs holds only the
standardized CELLxGENE schema columns (cell_type, disease, donor_id, sex, ...). The author
columns we need (fine author labels such as 'full_clustering', COVID severity, site, days
from onset, age group, outcome) exist only in the submitter's source h5ad. We therefore use
cellxgene_census.download_source_h5ad(dataset_id, census_version=...) (same approach as
lupusfull_download.py). If the dataset is absent from the requested Census release (e.g.
datasets published after it), we fall back to the CELLxGENE Discover curation API asset URL.
Census get_obs is used, when available, only to print a donor/disease summary up front.

Output: data/raw/<name>.h5ad with X = raw integer counts (sparse CSR, float32), var_names =
gene symbols (feature_name), var['ensembl_id'], ALL source obs columns, uns['source_dataset_id'].
At most --cap cells per donor (uniform random, seed 0). Fails loudly if counts are not integers.
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

DATASETS = {
    # Stephenson et al. 2021 Nat Med, doi 10.1038/s41591-021-01329-2 (Haniffa lab; Newcastle,
    # Cambridge, London). 647,366 PBMCs, ~120 donor_ids; COVID-19 asymptomatic->critical,
    # healthy, non-COVID respiratory illness, and LPS-stimulated healthy PBMC (dropped).
    "covid_stephenson": dict(
        dataset_id="c7775e88-49bf-4ba2-a03b-93f00447c958",
        collection_id="ddfad306-714d-4cc0-9985-d9072820c530",
        tissue=None),
    # Yoshida et al. 2022 Nature, doi 10.1038/s41586-021-04345-x (children and adults).
    # PBMC dataset: 422,220 cells, 75 donors; COVID-19, normal, post-COVID-19 disorder.
    "covid_yoshida": dict(
        dataset_id="2a498ace-872a-4935-984b-1afa70fd9886",
        collection_id="03f821b4-87be-4ff4-b65a-b5fc00061da7",
        tissue=None),
    # IBDverse 2026 Nature, doi 10.1038/s41586-026-10627-z. 1,852,681 cells, blood + ileum +
    # rectum; Crohn disease and normal (~421 individuals, 125 CD). Blood cells only are kept.
    "ibd_blood": dict(
        dataset_id="0e5c75da-7bc9-4a2c-83fc-ef1e378bd5f9",
        collection_id="7f7fdf50-aa0e-4a6f-979c-53655cac9d5d",
        tissue="blood"),
    # Backup only (small): Wilk et al. 2020 Nat Med, doi 10.1038/s41591-020-0944-y, 13 donors.
    "covid_wilk": dict(
        dataset_id="456e8b9b-f872-488b-871d-94534090a865",
        collection_id="a72afd53-ab92-4511-88da-252fb0e26b9a",
        tissue=None),
}
CENSUS_VERSIONS = ["2025-11-08", "stable", "latest"]
CURATION = "https://api.cellxgene.cziscience.com/curation/v1"

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*", default=["covid_stephenson", "covid_yoshida"])
ap.add_argument("--cap", type=int, default=2000, help="max cells per donor")
ap.add_argument("--keep-source", action="store_true", help="keep the multi-GB source h5ad")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
os.makedirs("data/raw", exist_ok=True)


def log(*x):
    print(*x, flush=True)


def census_summary(did):
    """Print disease x donor counts from Census (standard columns only). Returns the version that has did."""
    try:
        import cellxgene_census
    except ImportError:
        log("cellxgene_census not importable; skipping Census summary")
        return None
    for v in CENSUS_VERSIONS:
        try:
            with cellxgene_census.open_soma(census_version=v) as c:
                obs = cellxgene_census.get_obs(
                    c, "homo_sapiens", value_filter=f"dataset_id == '{did}'",
                    column_names=["donor_id", "disease", "tissue_general", "cell_type", "assay"])
            if len(obs) == 0:
                log(f"census {v}: dataset {did} not present")
                continue
            log(f"census {v}: {len(obs)} cells, {obs.donor_id.nunique()} donors")
            log(obs.groupby("disease", observed=True).donor_id.nunique().rename("donors").to_string())
            log(obs.tissue_general.value_counts().to_string())
            log(obs.assay.value_counts().to_string())
            return v
        except Exception as e:  # network or unknown version
            log(f"census {v}: failed ({type(e).__name__}: {e})")
    return None


def fetch_source(name, cfg):
    did, cid = cfg["dataset_id"], cfg["collection_id"]
    dst = f"data/raw/src_{name}.h5ad"
    if os.path.exists(dst):
        log("source exists", dst)
        return dst
    v = census_summary(did)
    if v is not None:
        import cellxgene_census
        try:
            cellxgene_census.download_source_h5ad(did, to_path=dst, census_version=v)
            return dst
        except Exception as e:
            log(f"download_source_h5ad failed ({e}); trying Discover curation API")
    url = f"{CURATION}/collections/{cid}/datasets/{did}"
    log("GET", url)
    with urllib.request.urlopen(url, timeout=120) as r:
        meta = json.load(r)
    assets = [x for x in meta.get("assets", []) if str(x.get("filetype", "")).upper() == "H5AD"]
    if not assets:
        raise SystemExit(f"no H5AD asset for {did}: {meta.get('assets')}")
    log("downloading", assets[0]["url"], "size", assets[0].get("filesize"))
    tmp = dst + ".part"
    urllib.request.urlretrieve(assets[0]["url"], tmp)
    os.replace(tmp, dst)
    return dst


def check_counts(X, what):
    s = X[: min(2000, X.shape[0])]
    d = s.data if sp.issparse(s) else np.asarray(s).ravel()
    if d.size == 0:
        raise SystemExit(f"{what}: empty matrix")
    frac = np.mean(np.abs(d - np.round(d)) < 1e-6)
    if frac < 0.999 or d.min() < 0:
        raise SystemExit(f"{what}: values are not raw counts (integer fraction {frac:.4f}, min {d.min()})")


for name in A.names:
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name}; choose from {list(DATASETS)}")
    cfg = DATASETS[name]
    out = f"data/raw/{name}.h5ad"
    if os.path.exists(out):
        log("exists", out)
        continue
    src = fetch_source(name, cfg)
    a = ad.read_h5ad(src, backed="r")
    obs = a.obs
    log(name, "source shape", a.shape)
    log("OBS_COLUMNS", obs.columns.tolist())
    for c in obs.columns:
        nu = obs[c].nunique()
        if nu < 80:
            log("COL", c, nu, obs[c].astype(str).value_counts().head(40).to_dict())
        else:
            log("COL", c, nu, "unique (first:", obs[c].astype(str).iloc[:3].tolist(), ")")
    if "donor_id" not in obs:
        raise SystemExit("donor_id missing from source obs")
    mask = np.ones(len(obs), bool)
    if cfg["tissue"]:
        tcol = next((c for c in ["tissue_general", "tissue"] if c in obs), None)
        if tcol is None:
            raise SystemExit("tissue column missing; cannot select blood")
        mask &= obs[tcol].astype(str).str.lower().eq(cfg["tissue"]).values
        log(f"tissue filter {tcol}=={cfg['tissue']}: {mask.sum()} cells")
        if mask.sum() == 0:
            raise SystemExit("no cells after tissue filter")
    rng = np.random.default_rng(A.seed)
    donors = obs["donor_id"].astype(str).values
    idx = []
    for d in pd.unique(donors[mask]):
        ii = np.where(mask & (donors == d))[0]
        idx.append(rng.choice(ii, min(A.cap, len(ii)), replace=False))
    idx = np.sort(np.concatenate(idx))
    log("subsampled", len(idx), "cells from", len(np.unique(donors[idx])), "donors")
    b = a[idx].to_memory()
    if b.raw is not None:
        X, var = b.raw.X, b.raw.var.copy()
        log("using .raw.X as counts")
    else:
        X, var = b.X, b.var.copy()
        log("no .raw; using .X as counts")
    X = sp.csr_matrix(X).astype(np.float32)
    check_counts(X, name)
    var["ensembl_id"] = var.index.astype(str)
    if "feature_name" not in var and "feature_name" in b.var:
        fn = b.var["feature_name"].astype(str)
        var["feature_name"] = [fn.get(g, g) for g in var.index.astype(str)]
    if "feature_name" not in var:
        raise SystemExit("var.feature_name missing; cannot set gene symbols")
    var.index = var["feature_name"].astype(str).values
    var.index.name = None
    o = b.obs.copy()
    for c in o.columns:  # categorical with unused levels -> clean strings, keep numerics
        if str(o[c].dtype) == "category":
            o[c] = o[c].astype(str)
    c = ad.AnnData(X=X, obs=o, var=var)
    c.var_names_make_unique()
    c.uns["source_dataset_id"] = cfg["dataset_id"]
    c.uns["source_collection_id"] = cfg["collection_id"]
    c.write_h5ad(out, compression="gzip")
    del a, b
    if not A.keep_source:
        os.remove(src)
    log("SECOND_OK", name, c.shape, c.obs.disease.value_counts().to_dict(), "donors", c.obs.donor_id.nunique())
