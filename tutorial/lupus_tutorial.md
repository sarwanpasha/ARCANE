# ARCANE tutorial: annotating lupus PBMC against healthy donors

This tutorial maps blood cells (PBMC) from people with systemic lupus erythematosus (SLE) onto a reference
built from healthy donors of the same study, and shows how to read what ARCANE returns: a label for every
cell, a set of plausible labels, and a flag for cells that do not look like any reference cell type. The
full script is `tutorial/run_tutorial.py`; this page explains each step.

Data: Perez et al. 2022, *Science* ("Single-cell RNA-seq reveals cell type-specific molecular and genetic
associations to lupus"), CELLxGENE dataset `218acb0f-9f2f-4f76-b90b-15a4b7c7f629`.

## 1. What ARCANE gives you

ARCANE sits on top of two scvi-tools models of the same reference:

* **scVI** learns a map of the cells without looking at the labels.
* **scANVI** learns a map that is shaped by the labels.

Query cells are placed in both maps (scArches surgery), and each map votes for a label based on the nearest
reference cells. When the two votes disagree, ARCANE picks the one that has been right more often for that
kind of disagreement on reference donors the models never saw during training. On top of the label it reports:

* a **prediction set**: every reference type the cell is compatible with. On new donors that resemble the
  reference donors, the true type is in the set for at least about 90% of the cells of each type
  (`alpha=0.1`);
* an **unknown flag**: the cell is farther from the reference than almost all held-out reference cells. On
  such donors, about 5% of ordinary cells get flagged by chance (`alpha_unknown=0.05`); a clearly higher rate
  means something in the query is not in the reference.

The "held-out reference donors" come from cross-fitting: the reference donors are split into folds, one
extra pair of models is trained without each fold, and the left-out fold is mapped together with the query.
That is why `fit` trains `1 + n_folds` model pairs.

## 2. Install

```bash
pip install scvi-tools==1.3.3 scanpy scikit-misc cellxgene-census
```

`arcane_tools` lives in the ARCANE repository; run from the repository root or add it to `PYTHONPATH`.
A GPU is used automatically by scvi-tools when available.

## 3. Download the data

```python
import cellxgene_census
cellxgene_census.download_source_h5ad(
    "218acb0f-9f2f-4f76-b90b-15a4b7c7f629", to_path="data/raw/lupus_perez2022.h5ad")
```

or `python tutorial/run_tutorial.py --download`. The file is large; `--max_cells_per_donor` subsamples cells
(every donor is kept) for a quicker first run. ARCANE needs raw counts: the script takes them from `adata.raw`.

## 4. Build the reference and the query

Healthy donors are split into reference donors and healthy query donors, within each processing cohort and
ancestry group so that both halves look alike. Everything else (the remaining healthy donors and all SLE
donors) is the query.

```python
import sys; sys.path[:0] = [".", "tutorial"]               # repository root and tutorial/ on the path
from run_tutorial import load, fine_labels, split_donors
adata = load("data/raw/lupus_perez2022.h5ad", max_cells_per_donor=1000)
adata.obs["fine_label"] = fine_labels(adata.obs)          # fine lymphocyte subsets + myeloid types
ref_donors = split_donors(adata.obs, ref_frac=0.55)       # healthy donors only
is_ref = adata.obs["donor_id"].isin(ref_donors).values
ref, query = adata[is_ref].copy(), adata[~is_ref].copy()
```

Why keep healthy donors in the query? They are the **controls**: cells that should look ordinary. ARCANE uses
them to calibrate the unknown flag on the query study itself (section 7).

Types with very few reference cells are dropped (`--min_cells`). Each type must also appear in every fold's
training donors; otherwise `fit` stops with an explanation.

## 5. Fit

```python
import logging
from arcane_tools import ARCANE
logging.basicConfig(level=logging.INFO)   # one log line per step with its runtime

model = ARCANE.fit(ref, batch_key="donor_id", labels_key="fine_label", parent_key="cell_type",
                   n_folds=2, n_jobs=16)
model.save("tutorial_out/model")
# later: model = ARCANE.load("tutorial_out/model", ref)
```

* `batch_key` is the donor. Folds are made of whole donors, balanced by cell count. Pass
  `fold_key="my_column"` to choose the folds yourself (it must be constant within each donor).
* `parent_key` gives a coarser label (here the CELLxGENE `cell_type`), used for parent-level calls.
* `n_seeds=3` trains three model sets with different seeds and reports majority-vote calls; it takes three
  times longer.
* `max_epochs=None` uses the training schedule of the ARCANE paper; `n_jobs` sets the threads used for the
  nearest-neighbour searches.

## 6. Map the query

```python
summary = model.map(query, control_key="disease", control_value="normal")
```

This runs the surgeries (query into the full model; held-out fold + query into each fold model), scores the
cells, and writes the results into `query`:

| column | meaning |
|---|---|
| `arcane_label` | the fine label |
| `arcane_parent_label` | the parent type when the set contains several fine types that all share one parent (e.g. two T-cell subsets), otherwise the fine label |
| `arcane_set` | all compatible types, separated by `;` (empty when none fits) |
| `arcane_set_size` | number of types in the set |
| `arcane_p_unknown`, `arcane_unknown` | unknown p-value and flag calibrated on held-out **reference** donors |
| `arcane_p_unknown_ctrl`, `arcane_unknown_ctrl` | the same, calibrated on the **healthy query donors**, leaving each healthy donor out when scoring its own cells |
| `arcane_novelty`, `arcane_novelty_U` | distance to the reference (both maps / scVI map only); higher means farther |

The maps are in `query.obsm["X_arcane_scvi"]` and `query.obsm["X_arcane_scanvi"]` (for UMAPs or clustering),
and `query.uns["arcane"]` holds a summary: settings, fold assignment, calibration cell counts, flag rates,
per-donor flag rates and the warnings.

## 7. Reading the results

**Labels.** Start with `arcane_label`. Where `arcane_parent_label` differs from it, the data cannot separate
the fine subsets for that cell, and the parent is the safer call.

```python
import pandas as pd
query.obs[["arcane_label", "arcane_parent_label", "arcane_set"]].head()
pd.crosstab(query.obs["arcane_label"], query.obs["author_cell_type"])   # agreement with the authors' labels
```

**Sets.** A set with one type is a confident call. A set with several types means the cell sits between
reference types. An empty set means no reference type is compatible; such cells are usually also flagged.

**Which unknown flag to use.** Use `arcane_unknown_ctrl` whenever the query has healthy donors. The
reference-calibrated flag (`arcane_unknown`) assumes the query donors look like the reference donors apart
from disease; when the query comes from another lab, protocol or chemistry, many ordinary cells look
"far" and that flag fires too often. The control-calibrated flag compares SLE cells with healthy cells
processed in the same study, so technical differences cancel and what remains is mostly biology. A healthy
query donor's own cells never take part in its own calibration, so the flag rate on healthy donors is an
honest check: it should sit near `alpha_unknown`.

**Donor-level flag rates.** Donors, not cells, are the units of comparison: cells from one donor are not
independent. Compare per-donor rates, for example SLE versus healthy:

```python
from scipy.stats import mannwhitneyu
d = model.donor_table()        # n_cells, flag_rate_ref, flag_rate_ctrl, mean_set_size, control
mannwhitneyu(d.loc[~d.control, "flag_rate_ctrl"], d.loc[d.control, "flag_rate_ctrl"], alternative="greater")
```

A higher rate in SLE donors says that some SLE cells occupy states not seen in healthy blood (for example
strong interferon responses or disease-associated subsets). To see which, look at the flagged cells by
`arcane_label` and score them with marker genes; remember that the flag says "unlike the reference", not why.

## 8. Warnings and what to do

ARCANE checks its own assumptions and raises a Python `ArcaneWarning` for each problem; the same messages are
in `query.uns["arcane"]["warnings"]` (and `warnings.txt` from the script).

| code | what it means | what to do |
|---|---|---|
| `transfer_shift` | the reference-calibrated flag fires on more than twice `alpha_unknown` of the controls (or of all query cells without controls): the query differs systematically from the reference | rely on `arcane_unknown_ctrl`; treat set coverage with caution |
| `few_calibration_cells` | a type has too few held-out cells in a fold, so it is put in every set; its coverage holds only trivially | read sets for that type as uninformative; add donors or merge the type with its parent |
| `few_control_donors` | fewer than 20 healthy query donors: the control calibration rests on few independent units and can drift from the nominal rate (the donor-weighted variant, `hierarchical_ctrl=True`, may be unable to flag anything) | report donor counts; use the default equal-cells calibration |
| `few_controls` | fewer than 2 healthy query donors: leave-one-donor-out calibration is impossible, so the `_ctrl` columns are empty | use the reference-calibrated flag, aware of `transfer_shift` |
| `set_quality` | mean set size above 3, or empty sets above twice `alpha`, on the controls: reference types are poorly separated in this query, or cells match none of them | check gene identifiers, quality control and the reference labels |
| `shared_batches` | some query donors are also reference donors: the guarantees assume new donors | remove those donors from one side |
| `missing_genes` | many model genes are absent from the query and were set to zero | make `var_names` use the same gene identifiers |

No warning does not prove that everything is fine; it means none of these known failure modes was detected.

## 9. Running the whole tutorial

```bash
python tutorial/run_tutorial.py --download --max_cells_per_donor 1000 --n_jobs 16 --outdir tutorial_out
```

It prints a few `KEY` lines (donor-level comparison of flag rates, set sizes, warnings) and writes
`cells.csv`, `donors.csv`, `summary.json`, `warnings.txt` and the saved models to `--outdir`.
