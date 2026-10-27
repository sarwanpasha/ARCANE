"""ARCANE tutorial: map lupus (SLE) PBMC onto a reference of healthy donors.

Data: Perez et al. 2022 (Science), CELLxGENE dataset 218acb0f-9f2f-4f76-b90b-15a4b7c7f629.
Healthy donors are split (within processing cohort and ancestry) into reference donors and healthy query
donors; the healthy query donors are the controls on which the unknown flag is calibrated.

    python tutorial/run_tutorial.py --download            # fetch the h5ad once (several GB)
    python tutorial/run_tutorial.py --max_cells_per_donor 1000 --n_jobs 16

Outputs in --outdir: cells.csv (ARCANE columns per query cell), donors.csv (per-donor flag rates),
summary.json (uns['arcane']), warnings.txt, and model/ (saved scVI/scANVI models).
"""
import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATASET_ID = "218acb0f-9f2f-4f76-b90b-15a4b7c7f629"


def download(path):
    import cellxgene_census
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cellxgene_census.download_source_h5ad(DATASET_ID, to_path=path)


def fine_labels(obs):
    """Fine labels as in prep_data.py: lymphocyte subtypes from ct_cov, other cells from author_cell_type."""
    if "ct_cov" in obs and "author_cell_type" in obs:
        ct = obs["ct_cov"].astype(str)
        return pd.Series(np.where(ct.isin(["nan", "NaN", "None", ""]), obs["author_cell_type"].astype(str), ct),
                         index=obs.index)
    col = next(c for c in ["author_cell_type", "cell_type"] if c in obs)
    return obs[col].astype(str)


def split_donors(obs, ref_frac=0.55, seed=0, donor="donor_id", disease="disease", healthy="normal"):
    """Reference = ref_frac of the healthy donors, drawn within processing cohort x ancestry strata."""
    coh = next((c for c in ["Processing_Cohort", "processing_cohort", "batch_cov"] if c in obs), None)
    anc = next((c for c in ["pop_cov", "self_reported_ethnicity"] if c in obs), None)
    dd = obs.groupby(donor, observed=True).agg(
        disease=(disease, "first"), coh=(coh or disease, "first"), anc=(anc or disease, "first")).reset_index()
    rng = np.random.default_rng(seed)
    ref = []
    for _, g in dd[dd.disease.astype(str) == healthy].groupby(["coh", "anc"], observed=True):
        d = sorted(g[donor].astype(str))
        rng.shuffle(d)
        ref += d[: int(round(ref_frac * len(d)))]
    return set(ref)


def load(path, max_cells_per_donor=None, seed=0):
    """Read the CELLxGENE h5ad (raw counts from .raw when present), optionally subsampling cells per donor."""
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    idx = np.arange(a.n_obs)
    if max_cells_per_donor:
        rng = np.random.default_rng(seed)
        don = obs["donor_id"].astype(str).values
        idx = np.concatenate([rng.choice(ii, min(len(ii), max_cells_per_donor), replace=False)
                              for ii in pd.Series(np.arange(len(don))).groupby(don).apply(np.array)])
    sub = a[np.sort(idx)].to_memory()
    if sub.raw is not None:
        sub = ad.AnnData(X=sub.raw.X, obs=sub.obs, var=sub.raw.var)
    for c in sub.obs.columns:  # categorical columns -> strings (simpler downstream)
        if str(sub.obs[c].dtype) == "category":
            sub.obs[c] = sub.obs[c].astype(str)
    return sub


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", default="data/raw/lupus_perez2022.h5ad")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--outdir", default="tutorial_out")
    ap.add_argument("--max_cells_per_donor", type=int, default=None, help="subsample for a quicker run")
    ap.add_argument("--min_cells", type=int, default=200, help="drop reference types with fewer cells")
    ap.add_argument("--ref_frac", type=float, default=0.55)
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--n_jobs", type=int, default=8)
    A = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if A.download or not os.path.exists(A.h5ad):
        download(A.h5ad)
    from arcane_tools import ARCANE
    os.makedirs(A.outdir, exist_ok=True)

    adata = load(A.h5ad, A.max_cells_per_donor)
    adata.obs["fine_label"] = fine_labels(adata.obs).values
    ref_donors = split_donors(adata.obs, A.ref_frac)
    is_ref = adata.obs["donor_id"].astype(str).isin(ref_donors).values
    ref, query = adata[is_ref].copy(), adata[~is_ref].copy()
    counts = ref.obs["fine_label"].value_counts()
    ref = ref[ref.obs["fine_label"].isin(counts.index[counts >= A.min_cells])].copy()
    print(f"reference: {ref.n_obs} cells, {len(ref_donors)} healthy donors, {ref.obs.fine_label.nunique()} types")
    print(f"query: {query.n_obs} cells, {query.obs.donor_id.nunique()} donors "
          f"({query.obs.loc[query.obs.disease == 'normal', 'donor_id'].nunique()} healthy)")

    model = ARCANE.fit(ref, batch_key="donor_id", labels_key="fine_label", parent_key="cell_type",
                       n_seeds=A.n_seeds, max_epochs=A.max_epochs, n_jobs=A.n_jobs)
    model.save(os.path.join(A.outdir, "model"), overwrite=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = model.map(query, control_key="disease", control_value="normal")

    cols = [c for c in query.obs.columns if c.startswith("arcane_")]
    query.obs[["donor_id", "disease"] + cols].to_csv(os.path.join(A.outdir, "cells.csv"))
    donors = model.donor_table()
    donors.to_csv(os.path.join(A.outdir, "donors.csv"))
    with open(os.path.join(A.outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(A.outdir, "warnings.txt"), "w") as fh:
        fh.write("\n".join(summary["warnings"]) + "\n")

    # donor-level comparison of the control-calibrated flag rate: donors, not cells, are the units
    from scipy.stats import mannwhitneyu
    if "flag_rate_ctrl" in donors:
        h, s = donors.loc[donors.control, "flag_rate_ctrl"], donors.loc[~donors.control, "flag_rate_ctrl"]
        p = mannwhitneyu(s, h, alternative="greater").pvalue if len(h) and len(s) else float("nan")
        print(f"KEY donor flag rate (control-calibrated): healthy median {h.median():.3f}, "
              f"SLE median {s.median():.3f}, one-sided Mann-Whitney p = {p:.2g}")
    print("KEY mean set size", round(summary["mean_set_size"], 3), "| empty sets", round(summary["empty_set_rate"], 3),
          "| parent-level calls", round(summary["parent_call_rate"], 3))
    for w in summary["warnings"]:
        print("KEY WARNING", w)
    print("KEY outputs in", A.outdir)


if __name__ == "__main__":
    main()
