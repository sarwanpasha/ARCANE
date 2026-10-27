"""Standardize a second-disease blood dataset for ARCANE (same contract as prep_data.py lupusfull).

Usage (from ~/arcane):  python prep_second.py <name> [--min_cells 50] [--ref_frac 0.55] [--label COL]
Reads data/raw/<name>.h5ad (from download_second.py), writes data/std/<name>.h5ad and
data/std/<name>_donors.csv (one row per donor: condition, severity, age, sex, site, role, fold, n).

Standard obs (as prep_data.finish): label (finest author label), parent (CELLxGENE cell_type),
batch (= donor), role ('ref'/'query'), fold (0/1 for reference donors, -1 otherwise),
condition (CELLxGENE disease string), plus ALL original obs columns, plus severity_std,
age_std, sex_std, site_std (copies of detected columns, 'NA' if absent).
Reference = ~55% of healthy ('normal') donors, stratified by site/cohort x sex (x age group when
present); query = remaining healthy donors + all patient donors. Labels: finest author column,
keeping labels with >= min_cells cells among reference cells (a label the reference cannot learn
is not a valid class); query cells whose label is not kept are dropped (counted in the log).
Excluded before splitting: in-vitro stimulated samples (any obs value containing 'LPS'),
doublets / unassigned / low-quality labels.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--min_cells", type=int, default=50)
ap.add_argument("--ref_frac", type=float, default=0.55)
ap.add_argument("--label", default=None, help="override fine label column")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
name = A.name
os.makedirs("data/std", exist_ok=True)


def log(*x):
    print(*x, flush=True)


# Candidate author columns, most specific first (case-insensitive match against obs columns).
FINE = {
    "covid_stephenson": ["full_clustering", "author_cell_type", "initial_clustering"],
    "covid_yoshida": ["annotation_detailed", "Annotation_detailed", "annotation_detailed_fullNames",
                      "author_cell_type", "annotation_broad"],
    "ibd_blood": ["label__machine", "label_machine", "category__machine", "author_cell_type", "celltype"],
}
GENERIC_FINE = ["author_cell_type", "cell_type_fine", "celltype_fine", "ct_fine", "fine_annotation",
                "annotation_fine", "annotation_detailed", "full_clustering", "cell_subtype", "celltype"]
SEVERITY = ["Status_on_day_collection_summary", "Worst_Clinical_Status", "COVID_severity",
            "covid_severity", "Severity", "severity", "disease_severity", "WHO_severity",
            "Status_on_day_collection", "COVID_status", "disease_status", "inflammation_status"]
AGE = ["Age_interval", "age", "Age", "Age_group", "age_group", "development_stage"]
SEX = ["sex", "Sex"]
SITE = ["Site", "site", "Processing_Cohort", "processing_cohort", "Centre", "centre", "center", "batch",
        "Batch", "sequencing_batch", "pool", "library_batch"]
BAD_LABEL = re.compile(r"doublet|unassigned|low.?quality|^nan$|^none$|^na$|^unknown$|^$", re.I)
SCHEMA = {"cell_type", "cell_type_ontology_term_id", "disease", "disease_ontology_term_id", "tissue",
          "tissue_ontology_term_id", "tissue_general", "assay", "assay_ontology_term_id", "sex",
          "sex_ontology_term_id", "development_stage", "development_stage_ontology_term_id",
          "self_reported_ethnicity", "self_reported_ethnicity_ontology_term_id", "organism",
          "organism_ontology_term_id", "donor_id", "suspension_type", "tissue_type", "is_primary_data",
          "observation_joinid"}


def find(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None


def assign_folds(o, ref_batches):
    """Greedy 2-fold split of reference donors with balanced cell counts (as prep_data.py)."""
    sizes = o.loc[o.batch.isin(ref_batches), "batch"].value_counts()
    load, fold = [0, 0], {}
    for b, n in sizes.sort_values(ascending=False).items():
        f = int(load[1] < load[0])
        fold[b] = f
        load[f] += n
    o["fold"] = o.batch.map(fold).fillna(-1).astype(int)
    return o


a = ad.read_h5ad(f"data/raw/{name}.h5ad")
o = a.obs
log(name, "raw", a.shape)
for req in ["donor_id", "disease", "cell_type"]:
    if req not in o:
        raise SystemExit(f"required obs column {req} missing; columns: {o.columns.tolist()}")

# 1. drop in-vitro stimulated samples (Stephenson LPS arm is labelled 'normal' disease!)
stim = np.zeros(a.n_obs, bool)
for c in o.columns:
    if o[c].dtype == object or str(o[c].dtype) == "category":
        v = o[c].astype(str)
        hit = v.str.contains(r"\bLPS\b", case=True, regex=True).values
        if 0 < hit.sum() < a.n_obs:
            log(f"stimulated marker in column {c}: {hit.sum()} cells ({v[hit].unique()[:5].tolist()})")
            stim |= hit
if stim.any():
    a = a[~stim].copy()
    o = a.obs
    log("dropped stimulated cells", int(stim.sum()))

# 2. tissue guard for multi-tissue sources
tcol = find(o.columns, ["tissue_general", "tissue"])
if tcol and o[tcol].astype(str).str.lower().nunique() > 1:
    keep = o[tcol].astype(str).str.lower().isin(["blood", "peripheral blood", "venous blood"]).values
    log("tissue values", o[tcol].astype(str).value_counts().to_dict(), "-> keeping blood", keep.sum())
    a = a[keep].copy()
    o = a.obs

# 3. fine label column
fine = A.label or find(o.columns, FINE.get(name, []) + GENERIC_FINE)
if fine is None:  # heuristic: author (non-schema) string column with the most labels, 5..150 levels
    cands = [c for c in o.columns if c not in SCHEMA and (o[c].dtype == object or str(o[c].dtype) == "category")
             and re.search(r"clust|annot|type|label|ct", c, re.I) and 5 <= o[c].nunique() <= 150]
    if not cands:
        raise SystemExit(f"no fine label column found; pass --label. columns: {o.columns.tolist()}")
    fine = max(cands, key=lambda c: o[c].nunique())
log("fine label column", fine, "n labels", o[fine].nunique())
log(o[fine].astype(str).value_counts().to_dict())
if o[fine].nunique() < o["cell_type"].nunique():
    log(f"WARNING: author label '{fine}' has fewer levels than cell_type; consider --label")
o["label_raw"] = o[fine].astype(str)
bad = o.label_raw.str.match(BAD_LABEL).values | o.label_raw.isin(["nan", "None", ""]).values
a = a[~bad].copy()
o = a.obs
log("dropped bad labels", int(bad.sum()))

# 4. donor table and stratified reference selection
o["donor"] = o.donor_id.astype(str)
sev, age, sex, site = (find(o.columns, x) for x in (SEVERITY, AGE, SEX, SITE))
log("severity col", sev, "| age col", age, "| sex col", sex, "| site col", site)
for std, col in [("severity_std", sev), ("age_std", age), ("sex_std", sex), ("site_std", site)]:
    o[std] = o[col].astype(str) if col else "NA"
nd = o.groupby("donor").disease.nunique()
if (nd > 1).any():
    log("WARNING donors with >1 disease value (kept, first used):", nd[nd > 1].index.tolist()[:10])
dd = o.groupby("donor").agg(condition=("disease", "first"), severity=("severity_std", "first"),
                            age=("age_std", "first"), sex=("sex_std", "first"), site=("site_std", "first"),
                            n=("disease", "size")).reset_index()
healthy = dd[dd.condition == "normal"]
if len(healthy) < 6:
    raise SystemExit(f"only {len(healthy)} healthy donors; cannot build a reference")
age_grp = healthy.age.where(healthy.age.str.len() < 20, "NA")  # use age only when it is a short group label
strata = healthy.site + "|" + healthy.sex + ("|" + age_grp if age_grp.nunique() <= 6 else "")
rng = np.random.default_rng(A.seed)
ref = []
for _, g in healthy.groupby(strata):
    d = g.donor.tolist()
    rng.shuffle(d)
    k = int(np.floor(A.ref_frac * len(d) + rng.random()))  # stochastic rounding keeps ~55% overall
    ref += d[:k]
if len(ref) < 4:
    raise SystemExit("reference has < 4 donors")
qry = [d for d in dd.donor if d not in set(ref)]
log(f"donors: healthy {len(healthy)}, ref {len(ref)}, query {len(qry)} "
    f"(healthy query {len(healthy) - len(ref)}, patients {len(dd) - len(healthy)})")

o["batch"] = o.donor
o["role"] = np.where(o.batch.isin(ref), "ref", "query")
o["condition"] = o.disease.astype(str)
o["parent"] = o.cell_type.astype(str)

# 5. labels with >= min_cells in the reference
rc = o.loc[o.role == "ref", "label_raw"].value_counts()
kept = rc.index[rc >= A.min_cells]
drop = ~o.label_raw.isin(kept)
log("labels kept", len(kept), "of", o.label_raw.nunique(),
    "| dropped cells by label:", o.loc[drop, "label_raw"].value_counts().head(30).to_dict())
a = a[~drop.values].copy()
o = a.obs
o["label"] = o.label_raw
# parent = majority cell_type of each fine label (one parent per label, needed by the hierarchy rule)
maj = o.groupby("label").parent.agg(lambda s: s.value_counts().index[0])
purity = o.groupby("label").apply(lambda g: (g.parent == maj[g.name]).mean())
log("label->parent purity (min/median):", round(purity.min(), 3), round(purity.median(), 3))
o["parent"] = o.label.map(maj).astype(str)
o = assign_folds(o, ref)

# 6. write
X = sp.csr_matrix(a.X).astype(np.float32)
d = X[: min(2000, X.shape[0])].data
if d.size and np.mean(np.abs(d - np.round(d)) < 1e-6) < 0.999:
    raise SystemExit("X is not raw counts")
keep = ["label", "parent", "batch", "role", "fold", "condition", "severity_std", "age_std", "sex_std", "site_std"]
obs = o[keep + [c for c in o.columns if c not in keep]].copy()
for c in obs.columns:
    if obs[c].dtype == object or str(obs[c].dtype) == "category":
        obs[c] = obs[c].astype(str)
b = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=a.var_names.astype(str)))
b.var_names_make_unique()
b.write_h5ad(f"data/std/{name}.h5ad", compression="gzip")
dd["role"] = np.where(dd.donor.isin(ref), "ref", "query")
fm = obs.groupby("batch").fold.first()
dd["fold"] = dd.donor.map(fm).fillna(-1).astype(int)
dd["n_kept"] = dd.donor.map(obs.batch.value_counts()).fillna(0).astype(int)
dd.to_csv(f"data/std/{name}_donors.csv", index=False)
log(name, b.shape, obs.role.value_counts().to_dict(), "labels", obs.label.nunique(), "parents",
    obs.parent.nunique(), "folds", obs.groupby("fold").size().to_dict())
log("condition x role donors:", dd.groupby(["condition", "role"]).size().to_dict())
log("severity x role donors:", dd.groupby(["severity", "role"]).size().to_dict())
log("PREP_SECOND_OK", name)
