"""(f) popV (Ergen et al. 2024, Nat Genet): ensemble of annotation methods with consensus score.

Usage: python comp_popv.py <ds> <seed> [holdout] [--methods M1,M2,...] [--n_per_label 300]
       [--scvi_epochs 0] [--hvg 4000]
Run in a separate venv (env_popv, see README): popV pins scikit-learn<1.8, harmonypy<0.1,
tensorflow<2.18 and pulls OnClass/transformers.
Settings: prediction_mode='retrain' (all models trained on our reference; no HuggingFace model
download), cl_obo_folder=False (no Cell Ontology -> ONCLASS disabled and the consensus is the
majority vote; ontology files would otherwise need downloading), unknown_celltype_label='unknown'.
Default methods (all train locally): KNN_SCVI, SCANVI_POPV, KNN_HARMONY, Support_Vector, XGboost,
CELLTYPIST. KNN_BBKNN and KNN_SCANORAMA can be added with --methods (slow on >100k cells).
--scvi_epochs N > 0 caps scVI/scANVI epochs through methods_kwargs (fast mode for large data).
Output: res/comp/popV_<tag>_seed<s>.npz
  pred = popv_prediction (majority vote); score = 1 - popv_prediction_score / n_methods
  (fraction of methods disagreeing with the consensus); variants pred__<method> per base method.
popV has no reject option, so no cell is called 'unknown'.
"""
import inspect
import os
import numpy as np
import pandas as pd
from comp_common import parse_args, run_main, load_std, save_result, Timer, versions

METHOD = "popV"
DEFAULT = "KNN_SCVI,SCANVI_POPV,KNN_HARMONY,Support_Vector,XGboost,CELLTYPIST"


def extra(ap):
    ap.add_argument("--methods", default=DEFAULT)
    ap.add_argument("--n_per_label", type=int, default=300)
    ap.add_argument("--scvi_epochs", type=int, default=0)
    ap.add_argument("--hvg", type=int, default=4000)


def slim(x, name_col):
    import anndata as ad
    obs = pd.DataFrame({"label": x.obs["label"].astype(str).values, "batch": x.obs["batch"].astype(str).values,
                        name_col: x.obs_names.astype(str).values}, index=x.obs_names.astype(str))
    return ad.AnnData(X=x.X.astype(np.float32), obs=obs, var=pd.DataFrame(index=x.var_names.astype(str)))


def main(a):
    import popv
    T = Timer()
    np.random.seed(a.seed)
    if hasattr(popv, "settings"):
        for k, v in (("seed", a.seed), ("random_seed", a.seed)):
            if hasattr(popv.settings, k):
                try:
                    setattr(popv.settings, k, v)
                except Exception:  # noqa
                    pass
    ref, qry = load_std(a.ds, a.holdout_label)
    from comp_common import uniq
    qnames = uniq(qry.obs_names.values.astype(str))
    qry.obs_names = qnames
    ref.obs_names = ["ref_" + x for x in uniq(ref.obs_names.values.astype(str))]
    ref, qry = slim(ref, "orig_name"), slim(qry, "orig_name")
    qry.obs["label"] = "unknown"  # never expose query labels to popV
    if "unknown" in set(ref.obs.label):
        raise ValueError("reference label 'unknown' collides with popV's unknown_celltype_label")
    save_dir = f"res/comp/popv_models/{a.tag}_seed{a.seed}/"
    os.makedirs(save_dir, exist_ok=True)
    kw = dict(query_adata=qry, ref_adata=ref, ref_labels_key="label", ref_batch_key="batch",
              query_batch_key="batch", cl_obo_folder=False, prediction_mode="retrain",
              unknown_celltype_label="unknown", n_samples_per_label=a.n_per_label,
              save_path_trained_models=save_dir, hvg=a.hvg)
    sig = inspect.signature(popv.preprocessing.Process_Query).parameters
    pq = popv.preprocessing.Process_Query(**{k: v for k, v in kw.items() if k in sig})
    adata = pq.adata
    T.mark("preprocess")
    methods = [m for m in a.methods.split(",") if m]
    avail = [m for m in methods if hasattr(popv.algorithms, m)]
    if len(avail) < len(methods):
        print("[popV] unavailable methods skipped:", sorted(set(methods) - set(avail)), flush=True)
    mk = {}
    if a.scvi_epochs > 0:
        for m in ("KNN_SCVI", "SCANVI_POPV"):
            if m in avail and "train_kwargs" in inspect.signature(getattr(popv.algorithms, m)).parameters:
                mk[m] = {"train_kwargs": {"max_epochs": a.scvi_epochs}}
    ak = dict(methods=avail, save_path=None, methods_kwargs=mk)
    asig = inspect.signature(popv.annotation.annotate_data).parameters
    popv.annotation.annotate_data(adata, **{k: v for k, v in ak.items() if k in asig})
    T.mark("annotate")
    obs = adata.obs
    qm = (obs["_dataset"].astype(str) == "query").values if "_dataset" in obs else np.isin(obs["orig_name"], qnames)
    o = obs[qm]
    if "orig_name" in o:
        o = o.set_index(o["orig_name"].astype(str))
    o = o[~o.index.duplicated()].reindex(qnames)
    if o["popv_prediction"].isna().any():
        raise ValueError(f"{o['popv_prediction'].isna().sum()} query cells lost by popV")
    pred = o["popv_prediction"].astype(str).values
    n_m = len(avail)
    score = 1.0 - o["popv_prediction_score"].astype(float).values / max(n_m, 1)
    variants = {}
    for c in o.columns:
        if c.endswith("_prediction") and c not in ("popv_prediction", "popv_majority_vote_prediction"):
            variants["pred__" + c.replace("popv_", "").replace("_prediction", "")] = o[c].astype(str).values
    meta = dict(methods=avail, methods_kwargs=mk, n_per_label=a.n_per_label, hvg=a.hvg,
                prediction_mode="retrain", ontology=False,
                versions=versions(["popv", "scvi", "scanpy", "anndata", "celltypist", "xgboost", "sklearn"]))
    save_result(METHOD, a.tag, a.seed, qnames, pred, score, variants=variants, meta=meta, timer=T)


if __name__ == "__main__":
    args = parse_args(__doc__, extra)
    run_main(METHOD, args, main)
