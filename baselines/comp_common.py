"""Shared helpers for the ARCANE competitor baselines (comp_*.py).

All competitor scripts are run from the project directory (~/arcane) as
    python comp_<name>.py <ds> <seed> [holdout]
and read the standardized AnnData data/std/<ds>.h5ad (raw counts; obs: label, parent, batch, role,
fold, condition) and/or the embedding files emb/<tag>_seed<s>_<split>.npz written by train.py.
The tag convention matches train.py / celltypist_run.py: <ds>[_ho-<Type>] where '~' stands for a
space in <Type> (the holdout type is removed from the reference only and stays in the query).

Every script writes res/comp/<method>_<tag>_seed<s>.npz with keys
    names   : query obs_names (target query cells)
    pred    : predicted label strings ('unknown' allowed)
    score   : float, higher = more novel / more uncertain (used for OOD AUROC)
    sets    : optional (n_cells, n_classes) bool prediction sets, with
    classes : class names (column order of sets)
    pred__<v>, score__<v>, sets__<v> : optional variants, evaluated as '<method>:<v>'
    meta    : JSON string (runtime, peak memory, settings, package versions)
and a sidecar res/comp/<method>_<tag>_seed<s>.json with the same meta. On failure, a
res/comp/<method>_<tag>_seed<s>.error.json with the traceback is written and the exit code is 1.
"""
import argparse
import json
import os
import resource
import sys
import time
import traceback

import numpy as np

COMP_DIR = "res/comp"
UNKNOWN = "unknown"


# ----------------------------------------------------------------------------------------------
# arguments, tags, I/O
# ----------------------------------------------------------------------------------------------
def make_tag(ds, holdout=None):
    tag = ds + (f"_ho-{holdout}" if holdout else "")
    return tag.replace(" ", "~").replace("/", "-").replace(":", "-")


def parse_args(desc, extra=None):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("ds", help="dataset name: data/std/<ds>.h5ad")
    ap.add_argument("seed", type=int)
    ap.add_argument("holdout", nargs="?", default=None,
                    help="reference type removed from the reference ('~' for spaces)")
    if extra is not None:
        extra(ap)
    a = ap.parse_args()
    if a.holdout in ("", "none", "None"):
        a.holdout = None
    a.tag = make_tag(a.ds, a.holdout)
    a.holdout_label = a.holdout.replace("~", " ") if a.holdout else None
    return a


def out_path(method, tag, seed, ext="npz"):
    return f"{COMP_DIR}/{method}_{tag}_seed{seed}.{ext}"


def check_cwd():
    if not os.path.isdir("data/std") and not os.path.isdir("emb"):
        raise SystemExit("Run from the project directory (~/arcane): neither data/std nor emb found in "
                         + os.getcwd())


def load_std(ds, holdout_label=None, backed=False):
    """Return (ref, qry) AnnData objects (in memory) from data/std/<ds>.h5ad.

    ref: role == 'ref' minus the holdout type; qry: role == 'query' (order as in the file)."""
    import anndata as ad
    path = f"data/std/{ds}.h5ad"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found (run prep_data.py {ds} first)")
    a = ad.read_h5ad(path, backed="r" if backed else None)
    need = ["label", "batch", "role"]
    miss = [c for c in need if c not in a.obs.columns]
    if miss:
        raise KeyError(f"{path}: obs is missing columns {miss}; has {a.obs.columns.tolist()[:40]}")
    lab = a.obs["label"].astype(str).values
    role = a.obs["role"].astype(str).values
    rmask = role == "ref"
    if holdout_label is not None:
        if not (lab == holdout_label).any():
            raise ValueError(f"holdout type '{holdout_label}' not among labels "
                             f"{sorted(set(lab))[:50]}")
        rmask &= lab != holdout_label
    qmask = role == "query"
    if rmask.sum() == 0 or qmask.sum() == 0:
        raise ValueError(f"empty reference ({rmask.sum()}) or query ({qmask.sum()})")
    if backed:
        ref, qry = a[rmask].to_memory(), a[qmask].to_memory()
    else:
        ref, qry = a[rmask].copy(), a[qmask].copy()
    del a
    for x in (ref, qry):
        x.obs["label"] = x.obs["label"].astype(str)
        x.obs["batch"] = x.obs["batch"].astype(str)
    print(f"[load_std] {ds}: ref {ref.shape}, query {qry.shape}, ref types {ref.obs.label.nunique()}, "
          f"holdout={holdout_label}", flush=True)
    return ref, qry


def load_obs(ds):
    """obs of data/std/<ds>.h5ad without loading X."""
    import anndata as ad
    path = f"data/std/{ds}.h5ad"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs.copy()
    if hasattr(a, "file"):
        a.file.close()
    return obs


def load_emb(tag, seed, split="full"):
    path = f"emb/{tag}_seed{seed}_{split}.npz"
    if not os.path.exists(path):
        avail = sorted(f for f in os.listdir("emb") if f.startswith(tag)) if os.path.isdir("emb") else []
        raise FileNotFoundError(f"{path} not found (run train.py first). Available for tag: {avail[:20]}")
    d = dict(np.load(path, allow_pickle=True))
    for key in ("ref_U", "tgt_U", "y_ref", "tgt_names"):
        if key not in d:
            raise KeyError(f"{path} lacks key {key}; keys: {sorted(d)}")
    if len(d["ref_U"]) != len(d["y_ref"]) or len(d["tgt_U"]) != len(d["tgt_names"]):
        raise ValueError(f"{path}: inconsistent array lengths")
    if "tgt_role" in d:  # full split: all targets are query cells; cf splits: keep query cells only
        pass
    print(f"[load_emb] {path}: ref {d['ref_U'].shape}, target {d['tgt_U'].shape}", flush=True)
    return d


def query_part(d):
    """Mask of query cells among the targets of an emb npz (all True for split full)."""
    if "tgt_role" in d:
        return np.asarray(d["tgt_role"]).astype(str) == "query"
    return np.ones(len(d["tgt_names"]), bool)


# ----------------------------------------------------------------------------------------------
# resources and saving
# ----------------------------------------------------------------------------------------------
class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.marks = {}

    def mark(self, name):
        self.marks[name] = round(time.time() - self.t0, 2)
        print(f"[time] {name}: {self.marks[name]:.1f}s", flush=True)


def resources():
    r_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    scale = 1e6 if sys.platform.startswith("linux") else 1e9  # Linux: KB, macOS: bytes
    out = dict(ram_peak_gb=r_self / scale, ram_peak_children_gb=r_child / scale,
               cpu_user_s=resource.getrusage(resource.RUSAGE_SELF).ru_utime)
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        try:
            if torch.cuda.is_available():
                out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                out["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return out


def versions(mods):
    v = {"python": sys.version.split()[0]}
    for m in mods:
        try:
            mod = __import__(m)
            v[m] = getattr(mod, "__version__", "?")
        except Exception as e:  # noqa
            v[m] = f"not importable ({type(e).__name__})"
    return v


def uniq(names):
    """Make cell names unique in order (k-th repeat gets suffix '-dupk'); unique arrays are unchanged."""
    names = np.asarray(names).astype(str)
    seen, out = {}, []
    for n in names:
        k = seen.get(n, 0)
        out.append(n if k == 0 else f"{n}-dup{k}")
        seen[n] = k + 1
    return np.array(out)


def _as_str(x):
    return np.asarray(x).astype(str)


def save_result(method, tag, seed, names, pred, score, sets=None, classes=None, variants=None,
                meta=None, timer=None, extra_arrays=None):
    """Validate and write res/comp/<method>_<tag>_seed<s>.npz (+ .json sidecar)."""
    os.makedirs(COMP_DIR, exist_ok=True)
    names, pred = _as_str(names), _as_str(pred)
    score = np.asarray(score, dtype=np.float64)
    n = len(names)
    if len(pred) != n or len(score) != n:
        raise ValueError(f"length mismatch names {n}, pred {len(pred)}, score {len(score)}")
    if len(set(names)) != n:
        names = uniq(names)
    bad = ~np.isfinite(score)
    if bad.any():
        print(f"[save] WARNING {bad.sum()} non-finite scores replaced by max finite + 1", flush=True)
        fin = score[~bad]
        score = score.copy()
        score[bad] = (fin.max() + 1.0) if len(fin) else 0.0
    arrs = dict(names=names, pred=pred, score=score.astype(np.float32))
    if sets is not None:
        sets = np.asarray(sets, bool)
        if classes is None or sets.shape != (n, len(classes)):
            raise ValueError("sets need classes with matching shape")
    if classes is not None:
        arrs["classes"] = _as_str(classes)
    if sets is not None:
        arrs["sets"] = sets
    for k, v in (variants or {}).items():
        if not (k.startswith("pred__") or k.startswith("score__") or k.startswith("sets__")):
            raise ValueError("variant keys must start with pred__/score__/sets__: " + k)
        v = np.asarray(v)
        if len(v) != n:
            raise ValueError(f"variant {k} has length {len(v)} != {n}")
        if k.startswith("pred__"):
            v = _as_str(v)
        elif k.startswith("score__"):
            v = np.nan_to_num(v.astype(np.float64), nan=np.nanmax(v) if np.isfinite(v).any() else 0.0,
                              posinf=np.nanmax(v[np.isfinite(v)]) + 1 if np.isfinite(v).any() else 0.0)
            v = v.astype(np.float32)
        else:
            v = v.astype(bool)
            if classes is None or v.shape[1] != len(classes):
                raise ValueError(f"{k}: sets need classes with matching width")
        arrs[k] = v
    for k, v in (extra_arrays or {}).items():
        arrs["x_" + k] = np.asarray(v)
    meta = dict(meta or {})
    meta.update(method=method, tag=tag, seed=seed, n_query=int(n),
                unknown_rate=float((pred == UNKNOWN).mean()), **resources())
    if timer is not None:
        meta["runtime_s"] = round(time.time() - timer.t0, 2)
        meta["timing"] = timer.marks
    arrs["meta"] = np.array(json.dumps(meta, default=str))
    p = out_path(method, tag, seed)
    np.savez_compressed(p, **arrs)
    json.dump(meta, open(out_path(method, tag, seed, "json"), "w"), indent=1, default=str)
    print(f"[save] {p}: n={n}, unknown={meta['unknown_rate']:.4f}, runtime={meta.get('runtime_s')}s, "
          f"ram_peak={meta['ram_peak_gb']:.2f}GB", flush=True)
    return p


def run_main(method, args, fn):
    """Run fn(args) with error capture; writes an .error.json and exits 1 on failure."""
    check_cwd()
    os.makedirs(COMP_DIR, exist_ok=True)
    err = out_path(method, args.tag, args.seed, "error.json")
    t0 = time.time()
    try:
        fn(args)
        if os.path.exists(err):
            os.remove(err)
    except SystemExit:
        raise
    except BaseException as e:  # noqa
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        json.dump(dict(method=method, tag=args.tag, seed=args.seed, error=repr(e), traceback=tb,
                       runtime_s=time.time() - t0, **resources()), open(err, "w"), indent=1, default=str)
        sys.exit(1)


# ----------------------------------------------------------------------------------------------
# kNN helpers
# ----------------------------------------------------------------------------------------------
def nn_index(Z, k):
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(Z)


def knn_transfer(Zr, yr, Zq, k=15, classes=None, index=None):
    """Majority-vote kNN label transfer. Returns (pred labels, vote fraction matrix, mean dist, classes)."""
    yr = _as_str(yr)
    classes = np.array(sorted(set(yr))) if classes is None else _as_str(classes)
    ci = {c: i for i, c in enumerate(classes)}
    yi = np.array([ci[c] for c in yr])
    index = index or nn_index(Zr, k)
    dist, idx = index.kneighbors(Zq, n_neighbors=k)
    P = np.zeros((len(Zq), len(classes)))
    np.add.at(P, (np.repeat(np.arange(len(Zq)), k), yi[idx].ravel()), 1.0 / k)
    return classes[P.argmax(1)], P, dist.mean(1), classes


def loo_mean_dist(Zr, k=15, index=None, max_n=None, seed=0):
    """Leave-one-out mean distance to the k nearest reference neighbours (self excluded)."""
    index = index or nn_index(Zr, k + 1)
    if max_n is not None and len(Zr) > max_n:
        sel = np.random.default_rng(seed).choice(len(Zr), max_n, replace=False)
    else:
        sel = np.arange(len(Zr))
    d, _ = index.kneighbors(Zr[sel], n_neighbors=k + 1)
    return d[:, 1:].mean(1)


# ----------------------------------------------------------------------------------------------
# conformal helpers (split conformal, standard = marginal and classwise = Mondrian)
# ----------------------------------------------------------------------------------------------
def aps_scores(P, rng):
    """Randomized APS nonconformity score for every (cell, class): sum of probabilities ranked above
    the class plus U * p_class (Romano et al. 2020; torchcp APS with randomization)."""
    order = np.argsort(-P, axis=1, kind="stable")
    Ps = np.take_along_axis(P, order, 1)
    cum_excl = np.cumsum(Ps, 1) - Ps
    U = rng.random(P.shape)
    Ss = cum_excl + U * Ps
    S = np.empty_like(Ss)
    np.put_along_axis(S, order, Ss, 1)
    return S


def lac_scores(P):
    return 1.0 - P


def conformal_quantile(s, alpha):
    s = np.sort(np.asarray(s))
    n = len(s)
    r = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or r > n:
        return np.inf
    return s[r - 1]


def split_conformal(S_cal, y_cal, S_test, alpha, n_classes, mode="standard"):
    """Prediction sets and p-values. S_*: (n, T) nonconformity matrices; y_cal integer labels.

    mode 'standard': one threshold from all calibration scores (marginal coverage);
    'classwise': per-class thresholds (class-conditional coverage); classes with too few
    calibration cells get an infinite threshold (always included; conservative)."""
    sc = S_cal[np.arange(len(y_cal)), y_cal]
    pv = np.ones_like(S_test, dtype=np.float64)
    if mode == "standard":
        q = conformal_quantile(sc, alpha)
        sets = S_test <= q
        s = np.sort(sc)
        ge = len(s) - np.searchsorted(s, S_test.ravel(), side="left")
        pv = ((1.0 + ge) / (len(s) + 1.0)).reshape(S_test.shape)
        qs = np.full(n_classes, q)
    else:
        qs = np.array([conformal_quantile(sc[y_cal == t], alpha) for t in range(n_classes)])
        sets = S_test <= qs[None, :]
        for t in range(n_classes):
            s = np.sort(sc[y_cal == t])
            ge = len(s) - np.searchsorted(s, S_test[:, t], side="left")
            pv[:, t] = (1.0 + ge) / (len(s) + 1.0)
    return sets, pv, qs


def marginal_pvalues(cal_scores, test_scores):
    """Conformal p-values for outlier detection (higher score = more anomalous)."""
    s = np.sort(np.asarray(cal_scores))
    ge = len(s) - np.searchsorted(s, np.asarray(test_scores), side="left")
    return (1.0 + ge) / (len(s) + 1.0)


def stratified_split(y, fracs, seed):
    """Split indices into consecutive stratified parts with the given fractions (sum 1)."""
    rng = np.random.default_rng(seed)
    y = _as_str(y)
    parts = [[] for _ in fracs]
    cum = np.cumsum(fracs)
    for c in np.unique(y):
        ii = rng.permutation(np.where(y == c)[0])
        cuts = np.round(cum * len(ii)).astype(int)
        start = 0
        for j, e in enumerate(cuts):
            parts[j].extend(ii[start:e].tolist())
            start = e
    return [np.sort(np.array(p, dtype=int)) for p in parts]


def log_normalize(adata, target_sum=1e4):
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata
