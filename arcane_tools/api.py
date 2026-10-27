"""User-facing ARCANE API on top of scvi-tools.

ARCANE transfers cell-type labels from an annotated reference to a query using two views of the same cells:
scVI (label-free, "U") and scANVI (label-shaped, "S"). It arbitrates between the two kNN labels, returns
class-conditional conformal prediction sets, and flags cells that match no reference type ("unknown").
The guarantees come from symmetric cross-fitting over folds of whole reference batches (donors): each fold
model is trained without one fold, and the held-out fold is mapped together with the query in one scArches
surgery, so held-out reference cells and query cells are scored in exactly the same way.

Typical use::

    from arcane_tools import ARCANE
    model = ARCANE.fit(adata_ref, batch_key="donor_id", labels_key="cell_type")
    model.map(adata_query, control_key="disease", control_value="normal")
    adata_query.obs[["arcane_label", "arcane_set", "arcane_unknown_ctrl"]]

The statistical core (``run_arcane_sym``) is imported unchanged from the research module ``arcane.py``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import warnings
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from arcane import run_arcane_sym, _fold_p  # noqa: E402  (research code, not modified)

logger = logging.getLogger("arcane_tools")
_BATCH, _LABEL, _UNLAB = "_arcane_batch", "_arcane_label", "_arcane_unlabeled"
_BAD_LABELS = {"nan", "NaN", "None", "NA", "", "none"}


class ArcaneWarning(UserWarning):
    """Warning raised when an ARCANE guarantee is likely not to hold for this query."""


@contextmanager
def _step(name, times):
    t0 = time.time()
    logger.info("ARCANE | %s ...", name)
    yield
    dt = time.time() - t0
    times[name] = times.get(name, 0.0) + dt
    logger.info("ARCANE | %s done in %.1f s", name, dt)


@contextmanager
def _threads(n_jobs):
    """Limit kNN / BLAS parallelism to n_jobs (sklearn reads the joblib context, BLAS/OpenMP via threadpoolctl)."""
    if n_jobs is None:
        yield
        return
    from joblib import parallel_backend
    from threadpoolctl import threadpool_limits
    n = os.cpu_count() if n_jobs == -1 else int(n_jobs)
    with parallel_backend("threading", n_jobs=n), threadpool_limits(limits=n):
        yield


def _counts(adata, layer):
    X = adata.layers[layer] if layer is not None else adata.X
    return sp.csr_matrix(X, dtype=np.float32)


def _align(X, src_genes, genes):
    """Columns of X reordered to `genes`; genes missing from src_genes are zero columns."""
    idx = pd.Index(np.asarray(src_genes, dtype=str)).get_indexer(np.asarray(genes, dtype=str))
    ok = idx >= 0
    sel = sp.csr_matrix((np.ones(ok.sum(), np.float32), (idx[ok], np.where(ok)[0])),
                        shape=(X.shape[1], len(genes)))
    return sp.csr_matrix(X @ sel), float(1.0 - ok.mean())


def _make_adata(X, names, batch, label, genes):
    import anndata as ad
    obs = pd.DataFrame({_BATCH: pd.Categorical(np.asarray(batch, dtype=str)),
                        _LABEL: pd.Categorical(np.asarray(label, dtype=str))},
                       index=pd.Index(np.asarray(names, dtype=str)))
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(np.asarray(genes, dtype=str))))


def _hvg(a, n_top):
    if n_top is None or n_top >= a.n_vars:
        return np.asarray(a.var_names, dtype=str)
    import scanpy as sc
    sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat_v3")
    return np.asarray(a.var_names[a.var["highly_variable"].values], dtype=str)


def greedy_folds(batch, n_folds=2):
    """Assign whole batches to folds by greedy size balancing (largest batch first, to the lightest fold).

    For two folds this is exactly prep_data.assign_folds. Returns {batch: fold}.
    """
    sizes = pd.Series(np.asarray(batch, dtype=str)).value_counts()
    load, fold = [0] * n_folds, {}
    for b, n in sizes.sort_values(ascending=False, kind="stable").items():
        f = int(np.argmin(load))
        fold[b] = f
        load[f] += int(n)
    return fold


def control_recal(nov_full, cond_ctrl, donor, seed=0, hierarchical=False):
    """Unknown-cell p-values calibrated on control donors of the query study, leave one donor out.

    Copied from evaluate2.control_recal. Each control donor contributes the same number of cells (per = 10th
    percentile of donor sizes, clipped to [50, 300]); a control cell is calibrated on the other control donors
    only, a non-control cell on all control donors. The hierarchical variant weights every calibration donor
    equally using all of its cells (p = (1 + sum_d mean_d[s_d >= s]) / (D + 1)).
    """
    ctrl = np.asarray(cond_ctrl, bool)
    donor = np.asarray(donor, dtype=str)
    rngc = np.random.default_rng(seed)
    dons = np.unique(donor[ctrl])
    per = int(max(50, min(300, np.percentile([(donor == d).sum() for d in dons], 10))))
    p = np.ones(len(nov_full))
    if hierarchical:
        srt = {d: np.sort(nov_full[(donor == d) & ctrl]) for d in dons}

        def hp(cal_dons, q):
            acc = np.zeros(len(q))
            for e in cal_dons:
                s_ = srt[e]
                acc += (len(s_) - np.searchsorted(s_, q, side="left")) / len(s_)
            return (1.0 + acc) / (len(cal_dons) + 1.0)
        p[~ctrl] = hp(dons, nov_full[~ctrl])
        for d in dons:
            m = donor == d
            p[m] = hp([e for e in dons if e != d], nov_full[m])
        return p, per, dons
    pick = {d: rngc.choice(np.where(donor == d)[0], min(per, (donor == d).sum()), replace=False) for d in dons}
    allcal = np.concatenate([nov_full[pick[d]] for d in dons])
    p[~ctrl] = _fold_p(allcal, nov_full[~ctrl])
    for d in dons:
        cal_d = np.concatenate([nov_full[pick[e]] for e in dons if e != d])
        m = donor == d
        p[m] = _fold_p(cal_d, nov_full[m])
    return p, per, dons


class ARCANE:
    """Label transfer with prediction sets and an unknown-cell flag, on scVI + scANVI.

    Create with :meth:`ARCANE.fit` (or :meth:`ARCANE.load`), then call :meth:`map` on each query.

    Attributes
    ----------
    models : dict
        ``models[seed][split] = (scvi.model.SCVI, scvi.model.SCANVI)`` with split ``"full"`` or ``"fold<f>"``;
        the fold-f model is trained on all reference batches except those of fold f.
    classes : ndarray of str
        Reference fine labels (sorted).
    fold_of_batch : dict
        Reference batch -> fold.
    """

    def __init__(self):
        self.models, self.genes, self.ref_latent, self.ref_index = {}, {}, {}, {}
        self.runtime = {}
        self.query_models = {}

    # ------------------------------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, adata_ref, batch_key, labels_key, n_latent=10, max_epochs=None, n_folds=2, seed=0,
            fold_key=None, n_seeds=1, parent_key=None, layer=None, n_top_genes=2000, scanvi_epochs=None,
            k=15, alpha=0.1, alpha_unknown=0.05, n_jobs=None):
        """Train scVI + scANVI on the full reference and on each cross-fitting fold.

        Parameters
        ----------
        adata_ref : AnnData
            Reference cells with raw integer counts in ``X`` (or ``layers[layer]``).
        batch_key : str
            obs column with the batch, normally the donor. Folds are made of whole batches, and the
            guarantees assume batches (donors) are exchangeable between reference and query.
        labels_key : str
            obs column with the fine cell-type labels. Cells with a missing label are dropped.
        n_latent : int
            Latent dimension of both models.
        max_epochs : int or None
            Cap on scVI epochs. Epochs = min(cap, max(20, round(8e6 / n_cells))), cap 200 by default (as train.py).
        n_folds : int
            Number of cross-fitting folds; batches are assigned by greedy size balancing (prep_data.assign_folds).
        seed : int
            Random seed of the first model set.
        fold_key : str or None
            obs column giving a user-defined fold per cell (must be constant within each batch).
        n_seeds : int
            Train ``n_seeds`` model sets (seeds seed, seed+1, ...); :meth:`map` then returns consensus calls.
        parent_key : str or None
            obs column with a coarser label; enables ``arcane_parent_label``.
        layer : str or None
            Layer holding raw counts (default ``X``).
        n_top_genes : int or None
            Highly variable genes (seurat_v3, computed on each training set, as train.py); None keeps all genes.
        scanvi_epochs : int or None
            scANVI epochs; default max(20, round(0.1 * scVI epochs)) as train.py.
        k, alpha, alpha_unknown : int, float, float
            kNN size, prediction-set miscoverage level, and level of the unknown flag.
        n_jobs : int or None
            Threads for kNN searches (and BLAS) in :meth:`map`; -1 uses all cores.
        """
        import scvi
        self = cls()
        t_all = time.time()
        obs = adata_ref.obs
        for key in [batch_key, labels_key] + [c for c in (parent_key, fold_key) if c is not None]:
            if key not in obs:
                raise KeyError(f"'{key}' not found in adata_ref.obs")
        lab = obs[labels_key].astype(str).values
        keep = ~np.isin(lab, list(_BAD_LABELS))
        if (~keep).any():
            logger.info("ARCANE | dropping %d reference cells without a label", int((~keep).sum()))
        if (lab[keep] == _UNLAB).any():
            raise ValueError(f"label '{_UNLAB}' is reserved")
        self.p = dict(batch_key=batch_key, labels_key=labels_key, n_latent=int(n_latent),
                      max_epochs=max_epochs, n_folds=int(n_folds), seed=int(seed), fold_key=fold_key,
                      n_seeds=int(n_seeds), parent_key=parent_key, layer=layer, n_top_genes=n_top_genes,
                      scanvi_epochs=scanvi_epochs, k=int(k), alpha=float(alpha), alpha_unknown=float(alpha_unknown),
                      n_jobs=n_jobs)
        self._setup_reference(adata_ref, keep)
        logger.info("ARCANE | reference %d cells, %d batches, %d types, %d folds, seeds %s",
                    self.n_ref, len(self.fold_of_batch), len(self.classes), self.n_folds, self.seeds)
        bs = 512 if self.n_ref > 50000 else 128
        self.p["batch_size"] = bs
        for s in self.seeds:
            self.models[s], self.genes[s], self.ref_latent[s] = {}, {}, {}
            for split in self.splits:
                tr = self.ref_index[split]
                with _step(f"seed {s} {split}: train scVI + scANVI on {len(tr)} cells", self.runtime):
                    a = self._ref_adata(tr)
                    genes = _hvg(a, n_top_genes)
                    a = a[:, genes].copy()
                    ep = int(min(max_epochs if max_epochs is not None else 200, max(20, round(8e6 / a.n_obs))))
                    sep = scanvi_epochs if scanvi_epochs is not None else max(20, int(round(ep * 0.1)))
                    scvi.settings.seed = s
                    scvi.model.SCVI.setup_anndata(a, batch_key=_BATCH, labels_key=_LABEL)
                    mU = scvi.model.SCVI(a, n_latent=n_latent)
                    mU.train(max_epochs=ep, batch_size=bs, enable_progress_bar=False)
                    mS = scvi.model.SCANVI.from_scvi_model(mU, unlabeled_category=_UNLAB, labels_key=_LABEL)
                    mS.train(max_epochs=sep, batch_size=bs, enable_progress_bar=False)
                    self.models[s][split], self.genes[s][split] = (mU, mS), genes
                    self.ref_latent[s][split] = self._latents(mU, mS)
        self.runtime["fit_total"] = time.time() - t_all
        return self

    def _setup_reference(self, adata_ref, keep, fold_of_batch=None):
        P = self.p
        self._ref = adata_ref
        self._keep = np.where(keep)[0]
        obs = adata_ref.obs.iloc[self._keep]
        self.ref_names = np.asarray(obs.index, dtype=str)
        self.ref_batch = obs[P["batch_key"]].astype(str).values
        self.ref_label = obs[P["labels_key"]].astype(str).values
        self.n_ref = len(self._keep)
        self.classes = np.array(sorted(set(self.ref_label)))
        if P["parent_key"] is not None:
            par = pd.DataFrame({"l": self.ref_label, "p": obs[P["parent_key"]].astype(str).values})
            self.parent_of = par.groupby("l")["p"].agg(lambda x: x.value_counts().index[0]).to_dict()
        else:
            self.parent_of = None
        if fold_of_batch is not None:
            self.fold_of_batch = {str(b): int(f) for b, f in fold_of_batch.items()}
            if set(self.ref_batch) - set(self.fold_of_batch):
                raise ValueError("adata_ref contains batches that were not in the fitted reference")
        elif P["fold_key"] is not None:
            fv = obs[P["fold_key"]].astype(str).values
            per_b = pd.Series(fv).groupby(self.ref_batch).nunique()
            if (per_b > 1).any():
                raise ValueError(f"fold_key must be constant within each batch; {list(per_b.index[per_b > 1])[:5]}")
            levels = sorted(set(fv))
            m = {v: i for i, v in enumerate(levels)}
            self.fold_of_batch = {b: m[f] for b, f in pd.Series(fv).groupby(self.ref_batch).first().items()}
        else:
            self.fold_of_batch = greedy_folds(self.ref_batch, P["n_folds"])
        self.n_folds = len(set(self.fold_of_batch.values()))
        if self.n_folds < 2:
            raise ValueError("at least 2 folds (and 2 reference batches) are needed for cross-fitting")
        self.p["n_folds"] = self.n_folds
        self.ref_fold = np.array([self.fold_of_batch[b] for b in self.ref_batch])
        self.splits = ["full"] + [f"fold{f}" for f in range(self.n_folds)]
        self.ref_index = {"full": np.arange(self.n_ref)}
        for f in range(self.n_folds):
            self.ref_index[f"fold{f}"] = np.where(self.ref_fold != f)[0]
            missing = sorted(set(self.classes) - set(self.ref_label[self.ref_fold != f]))
            if missing:
                raise ValueError(f"types {[str(x) for x in missing]} occur only in fold {f}, so the fold-{f} model never sees them; "
                                 "use fewer folds, a fold_key that spreads these types, or drop them")
        self.seeds = [self.p["seed"] + i for i in range(self.p["n_seeds"])]

    def _ref_adata(self, idx):
        X = _counts(self._ref, self.p["layer"])[self._keep[idx]]
        return _make_adata(X, self.ref_names[idx], self.ref_batch[idx], self.ref_label[idx],
                           np.asarray(self._ref.var_names, dtype=str))

    @staticmethod
    def _latents(mU, mS, adata=None):
        return (np.asarray(mU.get_latent_representation(adata), np.float32),
                np.asarray(mS.get_latent_representation(adata), np.float32))

    # ------------------------------------------------------------------------------------------ map
    def map(self, adata_query, control_key=None, control_value=None, surgery_epochs=None, n_jobs=None,
            hierarchical_ctrl=False, layer=None):
        """Map a query with scArches surgery and annotate it in place.

        The query is mapped into the full model; for each fold f, the held-out reference fold f and the query
        are mapped together into the fold-f model (one surgery), as train.py symmetric cross-fitting. Then
        ``arcane.run_arcane_sym`` computes labels, fold-Bonferroni prediction sets and unknown p-values.

        Parameters
        ----------
        adata_query : AnnData
            Query cells with raw counts and the same ``batch_key`` column as the reference (donor).
        control_key, control_value : str, str or list, optional
            obs column and value(s) marking control (e.g. healthy) query cells. When given, the unknown flag is
            also calibrated on the control donors, leaving each control donor out for its own cells.
        surgery_epochs : int or None
            Default min(100, max(30, round(4e6 / n_target))) per surgery (as train.py).
        n_jobs : int or None
            Overrides the n_jobs given at fit for the kNN searches.
        hierarchical_ctrl : bool
            Use the donor-weighted (hierarchical) control calibration instead of equal cells per donor.
        layer : str or None
            Query layer with raw counts (default ``X``).

        Writes ``adata_query.obs`` columns ``arcane_label``, ``arcane_parent_label``, ``arcane_set``,
        ``arcane_set_size``, ``arcane_p_unknown``, ``arcane_unknown``, ``arcane_p_unknown_ctrl``,
        ``arcane_unknown_ctrl``, ``arcane_novelty``, ``arcane_novelty_U``; ``obsm['X_arcane_scvi']``,
        ``obsm['X_arcane_scanvi']`` (full model, first seed); and ``uns['arcane']``. Returns the summary dict.
        """
        import anndata as ad
        import scvi
        P = self.p
        n_jobs = n_jobs if n_jobs is not None else P["n_jobs"]
        times, warn_list = {}, []
        t_all = time.time()
        bk = P["batch_key"]
        if bk not in adata_query.obs:
            raise KeyError(f"batch_key '{bk}' not found in adata_query.obs")
        q_batch = adata_query.obs[bk].astype(str).values
        nq = adata_query.n_obs
        Xq = _counts(adata_query, layer)
        q_names = np.array([f"q{i}" for i in range(nq)])
        ctrl = None
        if control_key is not None:
            if control_key not in adata_query.obs:
                raise KeyError(f"control_key '{control_key}' not found in adata_query.obs")
            vals = [control_value] if np.isscalar(control_value) or control_value is None else list(control_value)
            ctrl = np.isin(adata_query.obs[control_key].astype(str).values, [str(v) for v in vals])
        X_ref = _counts(self._ref, P["layer"])[self._keep]
        ref_genes = np.asarray(self._ref.var_names, dtype=str)
        q_genes = np.asarray(adata_query.var_names, dtype=str)
        per_seed, miss_frac = [], 0.0
        for s in self.seeds:
            dicts = {}
            for split in self.splits:
                mU, mS = self.models[s][split]
                genes = self.genes[s][split]
                Xa, miss = _align(Xq, q_genes, genes)
                miss_frac = max(miss_frac, miss)
                if split == "full":
                    tgt = _make_adata(Xa, q_names, q_batch, [_UNLAB] * nq, genes)
                    role = np.array(["query"] * nq)
                    y_tgt = np.array(["unlabeled"] * nq)
                else:
                    f = int(split[4:])
                    held = np.where(self.ref_fold == f)[0]
                    Xh, _ = _align(X_ref[held], ref_genes, genes)
                    tgt = _make_adata(sp.vstack([Xh, Xa]).tocsr(),
                                      np.concatenate([[f"cal{i}" for i in held], q_names]),
                                      np.concatenate([self.ref_batch[held], q_batch]), [_UNLAB] * (len(held) + nq), genes)
                    role = np.array(["cal"] * len(held) + ["query"] * nq)
                    y_tgt = np.concatenate([self.ref_label[held], ["unlabeled"] * nq])
                sep = surgery_epochs or int(min(100, max(30, round(4e6 / tgt.n_obs))))
                with _step(f"seed {s} {split}: surgery of {tgt.n_obs} cells ({sep} epochs)", times):
                    lat = []
                    for kls, ref_model in ((scvi.model.SCVI, mU), (scvi.model.SCANVI, mS)):
                        q = tgt.copy()
                        kls.prepare_query_anndata(q, ref_model)
                        qm = kls.load_query_data(q, ref_model)
                        qm.train(max_epochs=sep, batch_size=P["batch_size"], plan_kwargs=dict(weight_decay=0.0),
                                 enable_progress_bar=False)
                        lat.append(np.asarray(qm.get_latent_representation(), np.float32))
                        if split == "full" and s == self.seeds[0]:
                            self.query_models["scvi" if kls is scvi.model.SCVI else "scanvi"] = qm
                rU, rS = self.ref_latent[s][split]
                tr = self.ref_index[split]
                dicts[split] = dict(ref_U=rU, ref_S=rS, y_ref=self.ref_label[tr], tgt_U=lat[0], tgt_S=lat[1],
                                    y_tgt=y_tgt, tgt_role=role, b_tgt=np.asarray(tgt.obs[_BATCH].astype(str).values),
                                    tgt_names=np.asarray(tgt.obs_names, dtype=str))
            with _step(f"seed {s}: ARCANE scoring (kNN, n_jobs={n_jobs})", times), _threads(n_jobs):
                res = run_arcane_sym(dicts["full"], [dicts[f"fold{f}"] for f in range(self.n_folds)], self.classes,
                                     P["k"], P["alpha"], P["alpha_unknown"])
            out = dict(label=res["label"], sets=res["sets_bonf"], pvals=res["pvals_bonf"],
                       p_unknown=res["p_unknown_bonf"], novelty=res["novelty"], novelty_U=res["novelty_U"],
                       zU=dicts["full"]["tgt_U"], zS=dicts["full"]["tgt_S"])
            if ctrl is not None:
                with _step(f"seed {s}: control-calibrated unknown flag", times):
                    out["p_ctrl"] = self._control_p(res["novelty"], ctrl, q_batch, s, hierarchical_ctrl)
            per_seed.append(out)
        cons = self._consensus(per_seed)
        summary = self._write(adata_query, cons, per_seed, ctrl, q_batch, warn_list, miss_frac)
        times["map_total"] = time.time() - t_all
        summary["runtime_s"] = {k_: float(v) for k_, v in times.items()}
        adata_query.uns["arcane"] = summary
        logger.info("ARCANE | map done in %.1f s; %d warning(s)", times["map_total"], len(warn_list))
        return summary

    def _control_p(self, nov, ctrl, donor, seed, hierarchical):
        n_don = len(np.unique(donor[ctrl]))
        if n_don < 2:
            return None
        return control_recal(nov, ctrl, donor, seed=seed, hierarchical=hierarchical)[0]

    def _consensus(self, per_seed):
        """Combine seeds: majority vote of labels (ties -> first seed), of set membership and of flags;
        p-values are reported as the median across seeds."""
        if len(per_seed) == 1:
            r = dict(per_seed[0])
            r["unknown"] = r["p_unknown"] <= self.p["alpha_unknown"]
            if r.get("p_ctrl") is not None:
                r["unknown_ctrl"] = r["p_ctrl"] <= self.p["alpha_unknown"]
            return r
        n, T, au = len(per_seed), len(self.classes), self.p["alpha_unknown"]
        L = np.stack([r["label"] for r in per_seed])
        votes = np.zeros((L.shape[1], T))
        for row in L:
            votes[np.arange(L.shape[1]), row] += 1
        lab = np.where(votes[np.arange(L.shape[1]), L[0]] == votes.max(1), L[0], votes.argmax(1))
        r = dict(label=lab, sets=np.sum([x["sets"] for x in per_seed], 0) > n / 2,
                 pvals=np.median([x["pvals"] for x in per_seed], 0),
                 p_unknown=np.median([x["p_unknown"] for x in per_seed], 0),
                 unknown=np.sum([x["p_unknown"] <= au for x in per_seed], 0) > n / 2,
                 novelty=np.median([x["novelty"] for x in per_seed], 0),
                 novelty_U=np.median([x["novelty_U"] for x in per_seed], 0),
                 zU=per_seed[0]["zU"], zS=per_seed[0]["zS"])
        if per_seed[0].get("p_ctrl") is not None:
            r["p_ctrl"] = np.median([x["p_ctrl"] for x in per_seed], 0)
            r["unknown_ctrl"] = np.sum([x["p_ctrl"] <= au for x in per_seed], 0) > n / 2
        return r

    # -------------------------------------------------------------------------------- outputs + warnings
    def _write(self, adata, r, per_seed, ctrl, donor, warn_list, miss_frac):
        P, C = self.p, self.classes
        T, K = len(C), self.n_folds
        alpha, au = P["alpha"], P["alpha_unknown"]
        sets = np.asarray(r["sets"], bool)
        size = sets.sum(1)
        packed = np.packbits(sets, axis=1)
        uniq, inv = np.unique(packed, axis=0, return_inverse=True)
        strs = np.array([";".join(C[np.unpackbits(u)[:T].astype(bool)]) for u in uniq], dtype=object)
        set_str = strs[np.asarray(inv).ravel()]
        lab = C[r["label"]]
        if self.parent_of is not None:
            plev = np.array(sorted(set(self.parent_of.values())))
            pidx = np.array([np.searchsorted(plev, self.parent_of[c]) for c in C])
            lo = np.where(sets, pidx[None, :], len(plev)).min(1)
            hi = np.where(sets, pidx[None, :], -1).max(1)
            at_parent = (size > 1) & (lo == hi)
            parent_lab = np.where(at_parent, plev[np.clip(lo, 0, len(plev) - 1)], lab)
        else:
            at_parent = np.zeros(len(lab), bool)
            parent_lab = lab
        o = adata.obs
        o["arcane_label"] = pd.Categorical(lab)
        o["arcane_parent_label"] = pd.Categorical(parent_lab)
        o["arcane_set"] = set_str.astype(str)
        o["arcane_set_size"] = size.astype(int)
        o["arcane_p_unknown"] = r["p_unknown"].astype(float)
        o["arcane_unknown"] = np.asarray(r["unknown"], bool)
        has_ctrl = r.get("p_ctrl") is not None
        o["arcane_p_unknown_ctrl"] = r["p_ctrl"].astype(float) if has_ctrl else np.nan
        o["arcane_unknown_ctrl"] = pd.array(np.asarray(r["unknown_ctrl"], bool) if has_ctrl
                                            else [pd.NA] * len(lab), dtype="boolean")
        o["arcane_novelty"] = r["novelty"].astype(float)
        o["arcane_novelty_U"] = r["novelty_U"].astype(float)
        adata.obsm["X_arcane_scvi"] = r["zU"]
        adata.obsm["X_arcane_scanvi"] = r["zS"]

        def warn(code, msg):
            warn_list.append({"code": code, "message": msg})
            warnings.warn(f"[ARCANE {code}] {msg}", ArcaneWarning, stacklevel=3)

        diag = ctrl if (ctrl is not None and ctrl.any()) else np.ones(len(lab), bool)
        diag_name = "control query cells" if diag is ctrl else "query cells"
        # (5) donors shared with the reference
        shared = sorted(set(donor) & set(self.ref_batch))
        if shared:
            warn("shared_batches", f"{len(shared)} query batches also occur in the reference (e.g. {shared[:3]}); "
                 "the guarantees assume query donors are new donors exchangeable with reference donors, so error "
                 "rates for these cells are likely optimistic.")
        if miss_frac > 0.1:
            warn("missing_genes", f"up to {miss_frac:.0%} of the model genes are absent from the query and were set "
                 "to zero, which distorts the mapping; check that var_names use the same gene identifiers.")
        # (1) transfer shift
        rate = float(np.mean(np.asarray(r["unknown"], bool)[diag]))
        if rate > 2 * au:
            warn("transfer_shift", f"the reference-calibrated unknown flag marks {rate:.1%} of {diag_name} "
                 f"(expected about {au:.0%}); the query differs systematically from the reference (e.g. another "
                 "study or protocol), so the reference-calibrated flag and set coverage are not trustworthy here"
                 + ("; use arcane_unknown_ctrl." if has_ctrl else "; calibrate on query controls if available."))
        # (2) trivial coverage of rare types
        a_f = alpha / K
        min_f = int(math.ceil((1 - a_f) / a_f))
        cal_counts = {}
        for f in range(K):
            cnt = pd.Series(self.ref_label[self.ref_fold == f]).value_counts().reindex(C).fillna(0).astype(int)
            cal_counts[f"fold{f}"] = {c: int(v) for c, v in cnt.items()}
        small = sorted({c for f in cal_counts for c, v in cal_counts[f].items() if v < min_f})
        if small:
            warn("few_calibration_cells", f"types {small} have fewer than {min_f} held-out cells in some fold, so "
                 "they are included in every prediction set: their coverage holds trivially and sets say nothing "
                 "about them.")
        # (3) controls
        n_ctrl_don = int(len(np.unique(donor[ctrl]))) if ctrl is not None else 0
        if ctrl is not None:
            if n_ctrl_don < 2:
                warn("few_controls", f"only {n_ctrl_don} control donor(s): leave-one-donor-out calibration needs at "
                     "least 2, so arcane_p_unknown_ctrl / arcane_unknown_ctrl were not computed.")
            elif n_ctrl_don < 20:
                warn("few_control_donors", f"only {n_ctrl_don} control donors: control-calibrated p-values rest on "
                     f"few independent units, so the flag rate on controls can deviate noticeably from the nominal "
                     f"level (donor-weighted p-values, hierarchical_ctrl=True, cannot fall below 1/{n_ctrl_don + 1} "
                     "and then flag no cell at all when that exceeds alpha_unknown).")
        # (4) uninformative sets
        ms, emp = float(size[diag].mean()), float((size[diag] == 0).mean())
        if ms > 3 or emp > 2 * alpha:
            warn("set_quality", f"on {diag_name} the mean set size is {ms:.2f} and {emp:.1%} of sets are empty; "
                 "large sets mean the reference types are poorly separated in this query and many empty sets mean "
                 "cells look unlike every reference type, so labels should be read with caution.")
        # summary
        don = pd.DataFrame({"donor": donor, "flag_ref": np.asarray(r["unknown"], bool), "set_size": size})
        if has_ctrl:
            don["flag_ctrl"] = np.asarray(r["unknown_ctrl"], bool)
        if ctrl is not None:
            don["control"] = ctrl
        g = don.groupby("donor")
        dt = pd.DataFrame({"n_cells": g.size(), "flag_rate_ref": g["flag_ref"].mean(),
                           "mean_set_size": g["set_size"].mean()})
        if has_ctrl:
            dt["flag_rate_ctrl"] = g["flag_ctrl"].mean()
        if ctrl is not None:
            dt["control"] = g["control"].first()
        self.donor_table_ = dt
        summ = dict(
            params={k_: (v if v is not None else "None") for k_, v in P.items()},
            seeds=list(self.seeds), classes=C.tolist(), n_query=int(len(lab)),
            fold_of_batch={str(b): int(f) for b, f in self.fold_of_batch.items()},
            calibration_counts=cal_counts, min_cal_per_type=min_f,
            label_counts={str(k_): int(v) for k_, v in pd.Series(lab).value_counts().items()},
            mean_set_size=float(size.mean()), empty_set_rate=float((size == 0).mean()),
            singleton_rate=float((size == 1).mean()), parent_call_rate=float(at_parent.mean()),
            flag_rate_ref=float(np.mean(r["unknown"])),
            diagnostic_cells=diag_name, diagnostic_flag_rate_ref=rate,
            n_control_donors=n_ctrl_don, control_calibrated=bool(has_ctrl),
            donor_flag_rates={c: {str(i): (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v))
                                  for i, v in dt[c].items()} for c in dt.columns},
            warnings=[w["code"] + ": " + w["message"] for w in warn_list],
            warning_codes=[w["code"] for w in warn_list],
        )
        if ctrl is not None:
            summ["flag_rate_ref_controls"] = float(np.mean(np.asarray(r["unknown"], bool)[ctrl])) if ctrl.any() else float("nan")
        if has_ctrl:
            u = np.asarray(r["unknown_ctrl"], bool)
            summ["flag_rate_ctrl_controls"] = float(u[ctrl].mean())
            summ["flag_rate_ctrl_noncontrols"] = float(u[~ctrl].mean()) if (~ctrl).any() else float("nan")
            summ["excess_flag_rate_noncontrols"] = (summ["flag_rate_ctrl_noncontrols"] - au) / (1 - au)
        if len(per_seed) > 1:
            summ["seed_label_agreement"] = float(np.mean([np.mean(x["label"] == r["label"]) for x in per_seed]))
        return summ

    def donor_table(self):
        """Per-donor table of the last :meth:`map`: cells, flag rates (reference- and control-calibrated),
        mean set size, control status. Donors are the statistical units for any downstream comparison."""
        return self.donor_table_.copy()

    # ------------------------------------------------------------------------------------- save / load
    def save(self, dirname, overwrite=False):
        """Save all scVI/scANVI models (scvi ``model.save``, without AnnData) and the ARCANE settings."""
        d = Path(dirname)
        d.mkdir(parents=True, exist_ok=True)
        meta = dict(params=self.p, seeds=self.seeds, splits=self.splits, classes=self.classes.tolist(),
                    fold_of_batch=self.fold_of_batch, parent_of=self.parent_of, n_ref=self.n_ref,
                    ref_names_crc=zlib.crc32("\n".join(self.ref_names).encode()),
                    genes={str(s): {sp_: g.tolist() for sp_, g in self.genes[s].items()} for s in self.seeds})
        for s in self.seeds:
            for split, (mU, mS) in self.models[s].items():
                mU.save(str(d / f"seed{s}" / split / "scvi"), overwrite=overwrite, save_anndata=False)
                mS.save(str(d / f"seed{s}" / split / "scanvi"), overwrite=overwrite, save_anndata=False)
        with open(d / "arcane.json", "w") as fh:
            json.dump(meta, fh, indent=1, default=str)
        logger.info("ARCANE | saved to %s", d)

    @classmethod
    def load(cls, dirname, adata_ref):
        """Load a saved ARCANE model; ``adata_ref`` must be the reference used in :meth:`fit`."""
        import scvi
        d = Path(dirname)
        with open(d / "arcane.json") as fh:
            meta = json.load(fh)
        self = cls()
        self.p = meta["params"]
        lab = adata_ref.obs[self.p["labels_key"]].astype(str).values
        self._setup_reference(adata_ref, ~np.isin(lab, list(_BAD_LABELS)), fold_of_batch=meta["fold_of_batch"])
        if self.n_ref != meta["n_ref"] or zlib.crc32("\n".join(self.ref_names).encode()) != meta["ref_names_crc"] \
                or self.classes.tolist() != meta["classes"]:
            raise ValueError("adata_ref does not match the reference used to fit this model")
        self.splits = meta["splits"]
        self.seeds = [int(s) for s in meta["seeds"]]
        self.parent_of = meta["parent_of"]
        t0 = time.time()
        for s in self.seeds:
            self.models[s], self.genes[s], self.ref_latent[s] = {}, {}, {}
            for split in self.splits:
                genes = np.array(meta["genes"][str(s)][split])
                a = self._ref_adata(self.ref_index[split])
                a = a[:, genes].copy()
                mU = scvi.model.SCVI.load(str(d / f"seed{s}" / split / "scvi"), adata=a.copy())
                mS = scvi.model.SCANVI.load(str(d / f"seed{s}" / split / "scanvi"), adata=a.copy())
                self.models[s][split], self.genes[s][split] = (mU, mS), genes
                self.ref_latent[s][split] = self._latents(mU, mS)
        logger.info("ARCANE | loaded %d model pairs from %s in %.1f s",
                    len(self.seeds) * len(self.splits), d, time.time() - t0)
        return self

    def __repr__(self):
        return (f"ARCANE(types={len(self.classes)}, reference cells={self.n_ref}, folds={self.n_folds}, "
                f"seeds={self.seeds}, alpha={self.p['alpha']}, alpha_unknown={self.p['alpha_unknown']})")
