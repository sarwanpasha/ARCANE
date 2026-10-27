# ARCANE

ARCANE (arbitration of complementary embeddings for annotation with error control) transfers cell-type labels from an annotated single-cell reference to a query by combining two views of the same cells: an unsupervised scVI embedding (U) and a supervised scANVI embedding (S).

With symmetric cross-fitting, models trained on one half of the reference donors map the other half together with all query cells, so that calibration and query cells are scored by the same model against the same neighbors. From the calibration cells, ARCANE

1. arbitrates between the kNN labels of the two views with the rule that maximizes balanced accuracy (a class-conditional behavior knowledge space rule), which does not depend on the query composition, and reports a near-tie bound on its regret (`bound.py`, `cal_bounds.py`);
2. returns Mondrian conformal prediction sets with coverage for every cell type (fold p-values combined by Bonferroni), and reports the parent type when a set contains several fine types with one parent;
3. calls cells unknown when they lie far from the reference, with a false-alarm rate calibrated on the reference folds or, when the query contains control donors, on those controls (leave one donor out).

## Layout

| Path | Purpose |
|---|---|
| `arcane.py` | The method (`run_arcane_sym`): kNN evidence, arbitration, fold p-values, sets, unknown flags |
| `train.py` | Trains scVI/scANVI on the full reference or one fold and maps fold + query by scArches surgery; options for label noise, coarse or partial labels, latent size, scANVI epochs, and ISG-blind gene removal |
| `evaluate2.py` | ARCANE and baselines for one run; writes metrics, per-type tables and per-cell outputs |
| `arcane_tools/` | Python interface for scvi-tools models: `ARCANE.fit`, `map`, `save`, `load`; writes labels, sets, unknown flags and warnings into the query AnnData |
| `tutorial/` | Tutorial for mapping patient cells onto healthy donors (`lupus_tutorial.md`, `run_tutorial.py`) |
| `aggregate2.py`, `aggregate3.py`, `cal_bounds.py`, `bound.py` | Summary tables and near-tie bounds |
| `prep_data.py`, `*_download.py`, `celltypist_run.py`, `make_tasks2.py`, `make_tasks3.py`, `download_nb.py`, `prep_nb.py` | Data preparation (including the independent SLE cohort), CellTypist baseline, task lists |
| `symphony_views.py`, `run_comp3.sh` | Symphony as the unsupervised view, and the competitor runs |
| `baselines/` | popV, scPoli (and ARCANE on scVI + scPoli), Symphony, scHPL-style rejection, a conformal annotator, scANVI with label-shift correction, and their evaluation |
| `analysis/` | Disease analyses: lupus deep analysis and controls, ISG-blind comparison, candidate states, interferon-low patients, view divergence, COVID-19 analysis, gene sets, disease-state benchmark with Milo and scPSS (`dstate_bench.py`), excess-rate simulation (`excess_sim.py`), independent-cohort validation (`nb_validate.py`), robustness of the lupus results (`lupus_robust.py`) |

## Usage

```python
from arcane_tools import ARCANE
model = ARCANE.fit(adata_ref, batch_key="donor_id", labels_key="cell_type")
model.map(adata_query, control_key="disease", control_value="normal")
adata_query.obs[["arcane_label", "arcane_set", "arcane_unknown_ctrl"]]
```

The lower-level function works on latents written by `train.py`:

```python
from arcane import run_arcane_sym
res = run_arcane_sym(full, [fold0, fold1], classes, k=15, alpha=0.1, alpha_unknown=0.05)
res["label"], res["sets_bonf"], res["unknown_bonf"], res["novelty"]
```

`full` and the fold dictionaries hold reference and target latents (`ref_U`, `ref_S`, `tgt_U`, `tgt_S`), reference labels (`y_ref`) and, for folds, the role of each target cell (`tgt_role`: `cal` or `query`), as written by `train.py`.

Requirements: scvi-tools 1.3.3, scanpy, anndata, numpy, pandas, scipy, scikit-learn, statsmodels.

## License

MIT (see LICENSE).
