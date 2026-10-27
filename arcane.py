"""ARCANE: arbitration of complementary embeddings for annotation with error control.

Given kNN label evidence from a label-free (unsupervised) embedding U and a label-shaped
(supervised) embedding S of the same cells, ARCANE
  1. arbitrates between the two kNN labels with a rule that maximizes held-out balanced accuracy,
     estimated on reference batches that were held out during training (cross-fitting), and
  2. returns class-conditional (Mondrian) conformal prediction sets whose empty sets flag cells
     that match no reference type.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_evidence(z_ref, y_ref, z_tgt, classes, k=15, exclude_self=False):
    """Return (n_tgt, n_classes) kNN label proportions and the mean neighbor distance."""
    nn = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(z_ref)
    dist, idx = nn.kneighbors(z_tgt)
    if exclude_self:
        dist, idx = dist[:, 1:], idx[:, 1:]
    cls_index = {c: i for i, c in enumerate(classes)}
    yi = np.array([cls_index[c] for c in y_ref])
    P = np.zeros((len(z_tgt), len(classes)))
    np.add.at(P, (np.repeat(np.arange(len(z_tgt)), k), yi[idx].ravel()), 1.0 / k)
    return P, dist.mean(1)


def centroid_scores(ref_U, ref_S, y_ref, tgt_U, tgt_S, classes):
    """Scaled distance to each reference type centroid, maximized over the two views.

    For view V and type t, d = ||z - mu_t|| / r_t, where r_t is the median distance of type-t
    reference cells to mu_t. A cell is typical of t only if it is typical in both views.
    """
    out = None
    for R, Z in ((ref_U, tgt_U), (ref_S, tgt_S)):
        mu = np.stack([R[y_ref == c].mean(0) for c in classes])
        r = np.array([np.median(np.linalg.norm(R[y_ref == c] - mu[i], axis=1)) + 1e-8 for i, c in enumerate(classes)])
        D = np.sqrt(((Z[:, None, :] - mu[None, :, :]) ** 2).sum(-1)) / r[None, :]
        out = D if out is None else np.maximum(out, D)
    return out


def hard(P):
    return P.argmax(1)


class Arbiter:
    """Balanced-accuracy-optimal arbitration between two label predictions.

    For an ordered disagreement (a, b), i.e. U predicts a and S predicts b, the rule outputs a if
    q_a(a, b) >= q_b(a, b), where q_t(a, b) = P(U = a, S = b | Y = t) is estimated on calibration
    cells. Pairs without calibration cells (an empirical tie) fall back to the label with the
    larger summed evidence. With objective="accuracy", class-conditional rates are replaced by
    joint counts, which maximizes overall accuracy instead of balanced accuracy (ablation).
    """

    def __init__(self, n_classes, min_count=1, objective="balanced"):
        self.T, self.min_count, self.objective = n_classes, min_count, objective

    def fit(self, yU, yS, y):
        T = self.T
        N = np.bincount(y, minlength=T).astype(float)
        cnt = np.zeros((T, T, T))  # cnt[a, b, t]
        np.add.at(cnt, (yU, yS, y), 1.0)
        self.cnt = cnt
        w = 1.0 / np.maximum(N, 1.0) if self.objective == "balanced" else np.ones(T)
        A = np.arange(T)
        qa = cnt[A[:, None], A[None, :], A[:, None]] * w[:, None]  # q_a(a, b)
        qb = cnt[A[:, None], A[None, :], A[None, :]] * w[None, :]  # q_b(a, b)
        support = cnt[A[:, None], A[None, :], A[:, None]] + cnt[A[:, None], A[None, :], A[None, :]]
        self.choose_U = qa >= qb
        self.known = support >= self.min_count
        return self

    def predict(self, PU, PS):
        yU, yS = hard(PU), hard(PS)
        out = yS.copy()
        agree = yU == yS
        d = ~agree
        cu = self.choose_U[yU[d], yS[d]]
        kn = self.known[yU[d], yS[d]]
        ev = PU + PS
        fallback_U = ev[np.where(d)[0], yU[d]] >= ev[np.where(d)[0], yS[d]]
        pick_U = np.where(kn, cu, fallback_U)
        out[d] = np.where(pick_U, yU[d], yS[d])
        return out


class MondrianConformal:
    """Class-conditional split conformal prediction on a score matrix (lower = more typical)."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, Sc, y, n_classes):
        self.cal = [np.sort(Sc[y == t, t]) for t in range(n_classes)]
        pooled = np.sort(Sc[np.arange(len(y)), y])
        self.pooled = pooled
        self.tau = np.ones(n_classes)
        self.pooled_types = []
        for t, s in enumerate(self.cal):
            n = len(s)
            r = int(np.ceil((n + 1) * (1 - self.alpha)))
            if 1 <= r <= n:
                self.tau[t] = s[r - 1]
            else:  # too few calibration cells of this type: use the threshold pooled over all types
                rp = int(np.ceil((len(pooled) + 1) * (1 - self.alpha)))
                self.tau[t] = pooled[min(rp, len(pooled)) - 1]
                self.pooled_types.append(t)
        return self

    def sets(self, Sc):
        return Sc <= self.tau[None, :]

    def pvalues(self, Sc):
        """Conformal p-value of each class for each cell (type-conditional)."""
        S = Sc
        out = np.ones_like(S)
        for t, s in enumerate(self.cal):
            if t in self.pooled_types:
                s = self.pooled
            # number of calibration scores >= test score
            ge = len(s) - np.searchsorted(s, S[:, t], side="left")
            out[:, t] = (1.0 + ge) / (len(s) + 1.0)
        return out


def loo_scale(R, k=15, n=5000, seed=0):
    """Median leave-one-out kNN distance among (a sample of) reference cells; makes distances scale-free."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(R), min(n, len(R)), replace=False)
    d = NearestNeighbors(n_neighbors=k + 1).fit(R).kneighbors(R[idx])[0][:, 1:].mean(1)
    return float(np.median(d)) + 1e-12


def run_arcane(ref_U, ref_S, y_ref, tgt_U, tgt_S, cal, classes, k=15, alpha=0.1,
               objective="balanced", score_type="hybrid", alpha_unknown=0.05):
    """Fit ARCANE on calibration data and annotate target cells.

    cal: list of dicts with keys ref_U, ref_S, y_ref, tgt_U, tgt_S, y_tgt (one per cross-fit fold).
    Returns a dict with arbitrated labels, prediction sets, novelty scores, and evidence.
    """
    T = len(classes)
    ci = {c: i for i, c in enumerate(classes)}
    def score(rU, rS, yr, tU, tS, PU, PS):
        if score_type == "knn":
            return 1.0 - 0.5 * (PU + PS)
        D = centroid_scores(rU, rS, yr, tU, tS, classes)
        if score_type == "centroid":
            return D
        # hybrid: kNN evidence, with ties (including zero evidence) broken by centroid distance;
        # the tie-breaker is smaller than one evidence step 1/(2k), so it never reorders evidence levels
        return 1.0 - 0.5 * (PU + PS) + (0.999 / (2 * k)) * D / (1.0 + D)

    yU_c, yS_c, y_c, S_c, D_c = [], [], [], [], []
    for f in cal:
        keep = np.isin(f["y_tgt"], classes)
        PU, du = knn_evidence(f["ref_U"], f["y_ref"], f["tgt_U"][keep], classes, k)
        PS, ds = knn_evidence(f["ref_S"], f["y_ref"], f["tgt_S"][keep], classes, k)
        D_c.append(0.5 * (du / loo_scale(f["ref_U"], k) + ds / loo_scale(f["ref_S"], k)))
        yU_c.append(hard(PU)); yS_c.append(hard(PS))
        S_c.append(score(f["ref_U"], f["ref_S"], f["y_ref"], f["tgt_U"][keep], f["tgt_S"][keep], PU, PS))
        y_c.append(np.array([ci[c] for c in f["y_tgt"][keep]]))
    yU_c, yS_c, y_c, S_c, D_c = map(np.concatenate, (yU_c, yS_c, y_c, S_c, D_c))
    D_sorted = np.sort(D_c)
    r_u = int(np.ceil((len(D_sorted) + 1) * (1 - alpha_unknown)))
    tau_u = D_sorted[min(r_u, len(D_sorted)) - 1]
    arb = Arbiter(T, objective=objective).fit(yU_c, yS_c, y_c)
    conf = MondrianConformal(alpha).fit(S_c, y_c, T)
    PU, dU = knn_evidence(ref_U, y_ref, tgt_U, classes, k)
    PS, dS = knn_evidence(ref_S, y_ref, tgt_S, classes, k)
    Sq = score(ref_U, ref_S, y_ref, tgt_U, tgt_S, PU, PS)
    D = 0.5 * (dU / loo_scale(ref_U, k) + dS / loo_scale(ref_S, k))
    return dict(label=arb.predict(PU, PS), yU=hard(PU), yS=hard(PS), yavg=hard(PU + PS),
                sets=conf.sets(Sq), novelty=D, unknown=D > tau_u, tau_unknown=tau_u,
                novelty_p=(1.0 + len(D_sorted) - np.searchsorted(D_sorted, D, side="left")) / (len(D_sorted) + 1.0),
                PU=PU, PS=PS, dU=dU, dS=dS,
                score=Sq, arbiter=arb, conformal=conf, cal_size=np.bincount(y_c, minlength=T))


# ---------------------------------------------------------------------------------------------
# Symmetric cross-fitting (revision): calibration cells (held-out reference fold) and query cells
# are embedded by the same fold model and scored against the same neighbour pool (the training
# fold), so per-fold calibration and query scores are exchangeable under batch exchangeability.
# Fold-wise ranks are pooled into cross-conformal p-values.
# ---------------------------------------------------------------------------------------------
def _fold_scores(f, classes, k, score_type="hybrid"):
    """Scores of all target cells (cal + query) of one fold model."""
    PU, du = knn_evidence(f["ref_U"], f["y_ref"], f["tgt_U"], classes, k)
    PS, ds = knn_evidence(f["ref_S"], f["y_ref"], f["tgt_S"], classes, k)
    if "PS_override" in f:  # supervised evidence from another source, e.g. the scANVI classifier
        PS = f["PS_override"]
    if score_type == "knn":
        S = 1.0 - 0.5 * (PU + PS)
    else:
        D = centroid_scores(f["ref_U"], f["ref_S"], f["y_ref"], f["tgt_U"], f["tgt_S"], classes)
        S = 1.0 - 0.5 * (PU + PS) + (0.999 / (2 * k)) * D / (1.0 + D)
    nu_u, nu_s = du / loo_scale(f["ref_U"], k), ds / loo_scale(f["ref_S"], k)
    return dict(PU=PU, PS=PS, S=S, nu=0.5 * (nu_u + nu_s), nu_u=nu_u, nu_s=nu_s)


def _cross_p(cal_scores, cal_groups, q_scores, q_groups, groups):
    """Cross-conformal p-values. cal_scores/q_scores: lists over folds; groups: group ids to evaluate.
    Returns p[q, g] = (1 + sum_f #{cal_f in g with score >= q score_f}) / (n_g + 1)."""
    nq = len(q_scores[0])
    out = np.ones((nq, len(groups)))
    for j, g in enumerate(groups):
        cnt, n = np.zeros(nq), 0
        for cs, cg, qs in zip(cal_scores, cal_groups, q_scores):
            s = np.sort(cs[cg == g] if cg is not None else cs)
            n += len(s)
            qv = qs[:, j] if qs.ndim == 2 else qs
            cnt += len(s) - np.searchsorted(s, qv, side="left")
        out[:, j] = (1.0 + cnt) / (n + 1.0)
    return out


def _fold_p(cal, q):
    """Split-conformal p-value of query scores q against calibration scores cal (higher = less conforming)."""
    s = np.sort(cal)
    return (1.0 + len(s) - np.searchsorted(s, q, side="left")) / (len(s) + 1.0)


def run_arcane_sym(full, folds, classes, k=15, alpha=0.1, alpha_unknown=0.05, objective="balanced",
                   min_type=None, labeled_mask=None):
    """ARCANE with symmetric cross-fitting.

    full: npz dict of the full model (ref_*, y_ref, tgt_* for query cells).
    folds: list of npz dicts of fold models whose targets are held-out fold cells (tgt_role 'cal')
           followed by the query cells (tgt_role 'query'), in the same query order as `full`.
    """
    T = len(classes)
    ci = {c: i for i, c in enumerate(classes)}
    min_type = min_type if min_type is not None else int(np.ceil((1 - alpha) / alpha))
    yU_c, yS_c, y_c = [], [], []
    calS, calY, qS, calNu, qNu, calNuU, calNuS, qNuU, qNuS = [], [], [], [], [], [], [], [], []
    fs = [_fold_scores(f, classes, k) for f in folds]
    cal_feat, cal_b = [], []
    for f, sc in zip(folds, fs):
        role = f["tgt_role"]
        cal = (role == "cal") & np.isin(f["y_tgt"], classes)
        if labeled_mask is not None:
            cal &= labeled_mask(f["tgt_names"])
        q = role == "query"
        yc = np.array([ci[c] for c in f["y_tgt"][cal]])
        yU_c.append(hard(sc["PU"][cal])); yS_c.append(hard(sc["PS"][cal])); y_c.append(yc)
        calS.append(sc["S"][cal][np.arange(cal.sum()), yc]); calY.append(yc); qS.append(sc["S"][q])
        calNu.append(sc["nu"][cal]); qNu.append(sc["nu"][q])
        calNuU.append(sc["nu_u"][cal]); qNuU.append(sc["nu_u"][q]); calNuS.append(sc["nu_s"][cal]); qNuS.append(sc["nu_s"][q])
        cal_feat.append(np.hstack([sc["PU"][cal], sc["PS"][cal], sc["nu_u"][cal][:, None], sc["nu_s"][cal][:, None]]))
        cal_b.append(np.asarray(f["b_tgt"])[cal] if "b_tgt" in f else np.array(["?"] * int(cal.sum())))
    yU_c, yS_c, y_c = map(np.concatenate, (yU_c, yS_c, y_c))
    arb = Arbiter(T, objective=objective).fit(yU_c, yS_c, y_c)
    # labels from the full model
    PU, dU = knn_evidence(full["ref_U"], full["y_ref"], full["tgt_U"], classes, k)
    PS, dS = knn_evidence(full["ref_S"], full["y_ref"], full["tgt_S"], classes, k)
    if "PS_override" in full:
        PS = full["PS_override"]
    label = arb.predict(PU, PS)
    # Mondrian cross-conformal p-values; types with too few calibration cells use pooled scores
    counts = np.bincount(y_c, minlength=T)
    P = _cross_p(calS, calY, qS, None, list(range(T)))
    pooled = np.where(counts < min_type)[0]
    if len(pooled):
        Pp = np.column_stack([_cross_p(calS, [None] * len(folds), [s[:, t] for s in qS], None, [0])[:, 0]
                              for t in pooled])
        P[:, pooled] = Pp
    sets = P > alpha
    # unknown flag: marginal cross-conformal p-value of the novelty score
    p_nu = _cross_p(calNu, [None] * len(folds), qNu, None, [0])[:, 0]
    p_nu_u = _cross_p(calNuU, [None] * len(folds), qNuU, None, [0])[:, 0]
    p_nu_s = _cross_p(calNuS, [None] * len(folds), qNuS, None, [0])[:, 0]
    unknown = p_nu <= alpha_unknown
    # type-conditional variant: group calibration and query cells by their fold-model arbitrated label
    cal_pred = [arb.predict(np.eye(T)[a], np.eye(T)[b]) for a, b in
                zip(np.split(yU_c, np.cumsum([len(x) for x in calY])[:-1]),
                    np.split(yS_c, np.cumsum([len(x) for x in calY])[:-1]))]
    q_pred = [arb.predict(sc["PU"][f["tgt_role"] == "query"], sc["PS"][f["tgt_role"] == "query"])
              for f, sc in zip(folds, fs)]
    p_nu_type = np.ones(len(label))
    for t in range(T):
        # use fold 0 predicted label to define the query group (both folds agree in most cells)
        qmask = q_pred[0] == t
        if not qmask.any():
            continue
        ncal = sum(int((cp == t).sum()) for cp in cal_pred)
        cs = [cn[cp == t] for cn, cp in zip(calNu, cal_pred)] if ncal >= min_type else calNu
        qs = [qn[qmask] for qn in qNu]
        p_nu_type[qmask] = _cross_p(cs, [None] * len(folds), qs, None, [0])[:, 0]
    unknown_type = p_nu_type <= alpha_unknown
    nu_full = 0.5 * (dU / loo_scale(full["ref_U"], k) + dS / loo_scale(full["ref_S"], k))
    # ---- fold-wise p-values combined by Bonferroni (2 x min) and by a random fold: exact 1-alpha guarantees
    nF = len(folds)
    a_f = alpha / nF
    min_f = int(np.ceil((1 - a_f) / a_f))
    Pf = []
    for cs, cy, qs in zip(calS, calY, qS):
        cnt_f = np.bincount(cy, minlength=T)
        p = np.empty((qs.shape[0], T))
        for t in range(T):
            # a type with fewer than (1 - a_f) / a_f calibration cells in the fold keeps its own cells, which
            # keeps the guarantee and includes the type in every set
            p[:, t] = _fold_p(cs[cy == t], qs[:, t])
        Pf.append(p)
    P_bonf = np.minimum(1.0, nF * np.min(Pf, axis=0))
    rng = np.random.default_rng(0)
    pick = rng.integers(0, nF, size=P_bonf.shape[0])
    P_rand = np.stack(Pf)[pick, np.arange(P_bonf.shape[0])]
    # marginal unknown flag, Bonferroni over folds
    au_f = alpha_unknown / nF
    p_nu_b = np.minimum(1.0, nF * np.min([_fold_p(c, q) for c, q in zip(calNu, qNu)], axis=0))
    # type-conditional unknown flag grouped by each fold's own scANVI-view evidence label (not fitted on
    # calibration cells); labels with few training-fold reference cells share one group
    min_u = int(np.ceil((1 - au_f) / au_f))
    pt = []
    for f, sc, cn, qn in zip(folds, fs, calNu, qNu):
        role = f["tgt_role"]
        cal = (role == "cal") & np.isin(f["y_tgt"], classes)
        if labeled_mask is not None:
            cal &= labeled_mask(f["tgt_names"])
        ev = hard(sc["PS"])
        nref = np.array([(f["y_ref"] == c).sum() for c in classes])
        grp = np.where(nref[ev] >= 2 * min_u, ev, -1)
        gc, gq = grp[cal], grp[role == "query"]
        p = np.ones(len(qn))
        for g in np.unique(gq):
            cg = cn[gc == g]
            if len(cg) < min_u:
                cg = cn
            p[gq == g] = _fold_p(cg, qn[gq == g])
        pt.append(p)
    p_nu_tb = np.minimum(1.0, nF * np.min(pt, axis=0))
    # unknown flag from the scVI view alone, fold-Bonferroni
    p_nu_ub = np.minimum(1.0, nF * np.min([_fold_p(c, q) for c, q in zip(calNuU, qNuU)], axis=0))
    sU, sS = loo_scale(full["ref_U"], k), loo_scale(full["ref_S"], k)
    q_feat = np.hstack([PU, PS, (dU / sU)[:, None], (dS / sS)[:, None]])
    return dict(pvals_bonf=P_bonf, sets_bonf=P_bonf > alpha, sets_rand=P_rand > alpha,
                p_unknown_U_bonf=p_nu_ub, unknown_U_bonf=p_nu_ub <= alpha_unknown,
                novelty_U=dU / sU, novelty_S=dS / sS, cal_feat=np.vstack(cal_feat), q_feat=q_feat,
                p_unknown_bonf=p_nu_b, unknown_bonf=p_nu_b <= alpha_unknown,
                p_unknown_type_bonf=p_nu_tb, unknown_type_bonf=p_nu_tb <= alpha_unknown,
                label=label, yU=hard(PU), yS=hard(PS), yavg=hard(PU + PS), PU=PU, PS=PS, dU=dU, dS=dS,
                pvals=P, sets=sets, p_unknown=p_nu, unknown=unknown, unknown_type=unknown_type,
                p_unknown_U=p_nu_u, p_unknown_S=p_nu_s, novelty=nu_full, arbiter=arb,
                cal_counts=counts, cal=(yU_c, yS_c, y_c), fold_cal_nu=calNu, fold_q_nu=qNu,
                fold_cal_nu_u=calNuU, fold_q_nu_u=qNuU, fold_cal_batch=cal_b)
