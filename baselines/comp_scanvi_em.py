"""(a) scANVI classifier with Saerens-Latinne-Decaestecker (2002) EM label-shift correction.

Usage: python comp_scanvi_em.py <ds> <seed> [holdout]
Input : emb/<tag>_seed<s>_full.npz (train.py): scanvi_prob (query x classes), scanvi_classes, y_ref.
Output: res/comp/scanviEM_<tag>_seed<s>.npz
  pred  = argmax of the EM-adapted posterior (priors re-estimated on the pooled query)
  score = 1 - max adapted posterior
  variants: pred__plain/score__plain (uncorrected scANVI), pred__perbatch/score__perbatch (EM run
            separately within each query batch = donor, since class priors differ by donor),
            score__entropy (entropy of the adapted posterior).
scANVI has no reject option, so no cell is called 'unknown'.
"""
import numpy as np
import pandas as pd
from comp_common import parse_args, run_main, load_emb, save_result, Timer, versions, query_part

METHOD = "scanviEM"


def saerens_em(prob, prior_train, max_iter=1000, tol=1e-8):
    """EM for new class priors given calibrated posteriors under training priors.

    p_new(y|x) ∝ p_train(y|x) * w_y / pi_y; w <- mean_x p_new(y|x). Returns (adapted posterior, w, n_iter)."""
    prior_train = np.clip(prior_train, 1e-8, None)
    prior_train = prior_train / prior_train.sum()
    w = prior_train.copy()
    post = prob
    for it in range(1, max_iter + 1):
        post = prob * (w / prior_train)[None, :]
        post /= np.clip(post.sum(1, keepdims=True), 1e-300, None)
        w_new = post.mean(0)
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w = w_new
    return post, w, it


def entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(1)


def main(a):
    T = Timer()
    F = load_emb(a.tag, a.seed, "full")
    for k in ("scanvi_prob", "scanvi_classes"):
        if k not in F:
            raise KeyError(f"emb npz lacks {k}; re-run train.py (needs scanvi_prob)")
    q = query_part(F)
    prob = F["scanvi_prob"].astype(np.float64)[q]
    pc = np.asarray(F["scanvi_classes"]).astype(str)
    names = np.asarray(F["tgt_names"]).astype(str)[q]
    yref = pd.Series(np.asarray(F["y_ref"]).astype(str))
    # scANVI was trained on the reference labels; its implicit training prior is the reference
    # label frequency (scANVI does not rebalance classes by default).
    missing = sorted(set(pc) - set(yref)) + sorted(set(yref) - set(pc))
    if missing:
        print("[scanviEM] WARNING classes not shared by scANVI and y_ref:", missing[:20], flush=True)
    prior = yref.value_counts(normalize=True).reindex(pc).fillna(1e-6).values
    prob = prob / prob.sum(1, keepdims=True)
    post, w, nit = saerens_em(prob, prior)
    T.mark("em_pooled")
    # per-batch (donor) EM
    b = np.asarray(F["b_tgt"]).astype(str)[q] if "b_tgt" in F else np.array(["all"] * len(names))
    post_b = np.zeros_like(post)
    for bb in np.unique(b):
        m = b == bb
        post_b[m] = saerens_em(prob[m], prior)[0]
    T.mark("em_perbatch")
    pred = pc[post.argmax(1)]
    variants = {"pred__plain": pc[prob.argmax(1)], "score__plain": 1 - prob.max(1),
                "pred__perbatch": pc[post_b.argmax(1)], "score__perbatch": 1 - post_b.max(1),
                "score__entropy": entropy(post)}
    meta = dict(n_em_iter=int(nit), prior_ref=dict(zip(pc, prior.round(5).tolist())),
                prior_query_em=dict(zip(pc, w.round(5).tolist())),
                versions=versions(["numpy", "pandas"]))
    save_result(METHOD, a.tag, a.seed, names, pred, 1 - post.max(1), variants=variants, meta=meta, timer=T)


if __name__ == "__main__":
    args = parse_args(__doc__)
    run_main(METHOD, args, main)
