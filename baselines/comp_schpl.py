"""(d) scHPL-style kNN label transfer with distance rejection (Michielsen et al. 2021/2023).

Usage: python comp_schpl.py <ds> <seed> [holdout] [--k 15] [--pct 99] [--no_pkg]
Input : emb/<tag>_seed<s>_full.npz (train.py): ref_U, tgt_U, y_ref (scVI latent), ref_S/tgt_S.
Output: res/comp/scHPLknn_<tag>_seed<s>.npz
  Reimplementation of the scHPL distance rejection on a flat hierarchy: kNN (k=15) majority vote
  in the scVI latent; a query cell is rejected ('unknown') when its mean distance to its k nearest
  reference cells exceeds the <pct>-th percentile of the reference leave-one-out mean kNN
  distances (scHPL: _train_knn computes the distkNN=99 percentile of the training cells' mean
  neighbour distance, self excluded; predict_labels rejects if mean(dist) > maxDist).
  score = mean kNN distance (higher = more novel).
  variants: pred__S/score__S (same rule in the scANVI latent), pred__noreject (plain kNN-U),
            pred__pkg/score__pkg (the scHPL package itself, flat tree, knn classifier, k=15,
            dynamic_neighbors=False, useRE=False, distkNN=pct) if `pip install scHPL` works.
"""
import numpy as np
from comp_common import (parse_args, run_main, load_emb, save_result, Timer, versions, query_part,
                         knn_transfer, loo_mean_dist, nn_index, UNKNOWN)

METHOD = "scHPLknn"


def extra(ap):
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--no_pkg", action="store_true", help="skip the scHPL package run")


def reject_rule(Zr, yr, Zq, k, pct):
    idx = nn_index(Zr, k + 1)
    pred, P, d, classes = knn_transfer(Zr, yr, Zq, k, index=idx)
    loo = loo_mean_dist(Zr, k, index=idx)
    thr = float(np.percentile(loo, pct))
    out = pred.astype(object)
    out[d > thr] = UNKNOWN
    return out.astype(str), d, thr, pred


def run_pkg(Zr, yr, Zq, k, pct):
    """scHPL package on a flat tree. Labels are mapped to safe Newick ids."""
    from scHPL import utils as hutils
    try:
        from scHPL import train as htrain, predict as hpredict
    except ImportError:
        from scHPL import learn as htrain, predict as hpredict  # older layout
    classes = np.array(sorted(set(yr)))
    safe = {c: f"C{i}" for i, c in enumerate(classes)}
    back = {v: k_ for k_, v in safe.items()}
    tree = hutils.create_tree("(" + ",".join(safe[c] for c in classes) + ")root;")
    ys = np.array([safe[c] for c in yr])
    kw = dict(classifier="knn", dimred=False, useRE=False, n_neighbors=k, dynamic_neighbors=False,
              distkNN=int(pct))
    try:
        tree = htrain.train_tree(Zr, ys, tree, **kw)
    except TypeError:  # older signature without dynamic_neighbors/distkNN
        tree = htrain.train_tree(Zr, ys, tree, classifier="knn", dimred=False, useRE=False, n_neighbors=k)
    pred = np.asarray(hpredict.predict_labels(Zq, tree)).astype(str)
    out = np.array([back.get(p, UNKNOWN if p.lower().startswith("reject") or p == "root" else p)
                    for p in pred])
    return out, pred


def main(a):
    T = Timer()
    F = load_emb(a.tag, a.seed, "full")
    q = query_part(F)
    names = np.asarray(F["tgt_names"]).astype(str)[q]
    yr = np.asarray(F["y_ref"]).astype(str)
    Zr, Zq = F["ref_U"].astype(np.float32), F["tgt_U"][q].astype(np.float32)
    pred, d, thr, pred_nr = reject_rule(Zr, yr, Zq, a.k, a.pct)
    T.mark("knn_U")
    variants = {"pred__noreject": pred_nr, "score__noreject": d}
    meta = dict(k=a.k, pct=a.pct, threshold_U=thr, latent="scVI (U)")
    if "ref_S" in F:
        pS, dS, thrS, _ = reject_rule(F["ref_S"].astype(np.float32), yr, F["tgt_S"][q].astype(np.float32),
                                      a.k, a.pct)
        variants.update({"pred__S": pS, "score__S": dS})
        meta["threshold_S"] = thrS
        T.mark("knn_S")
    if not a.no_pkg:
        try:
            pk, raw = run_pkg(Zr, yr, Zq, a.k, a.pct)
            variants["pred__pkg"] = pk
            variants["score__pkg"] = (pk == UNKNOWN).astype(float) + 1e-6 * d  # binary call, ties by distance
            meta["pkg_raw_labels"] = sorted(set(raw))[:50]
            T.mark("scHPL_pkg")
        except Exception as e:  # noqa
            meta["pkg_error"] = repr(e)
            print("[scHPL] package run failed:", repr(e), flush=True)
    meta["versions"] = versions(["numpy", "sklearn", "scHPL"])
    save_result(METHOD, a.tag, a.seed, names, pred, d, variants=variants, meta=meta, timer=T)


if __name__ == "__main__":
    args = parse_args(__doc__, extra)
    run_main(METHOD, args, main)
