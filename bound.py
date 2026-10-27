"""Finite-sample guarantees for the balanced-accuracy arbitration of ARCANE (Theorem 2 of theory.tex).

Given calibration arrays yU (kNN-U label), yS (kNN-S label), y (true label), optionally fold (cross-fitting
fold of each calibration cell), this script computes, with probability >= 1 - delta,

  R_hat     : upper bound on the regret BA(g*) - BA(g_hat) of the plug-in rule g_hat (near-tie bound),
  LB_U/LB_S : lower bounds on BA(g_hat) - BA(U) and BA(g_hat) - BA(S) (certified gain over each view),
  LB_max    : lower bound on BA(g_hat) - max{BA(U), BA(S)} = max(min(LB_U, LB_S), -R_hat),
  old       : the uniform bound over all rules,  (2/T) sum_t sqrt((2T log2 + log(T/delta)) / (2 m_t)).

Confidence intervals for the class-conditional pair rates q_t(a,b) use, per fold, either the Chernoff/KL
inequality (Hoeffding 1963, Thm 1; default), the empirical Bernstein inequality (Maurer & Pontil 2009, Thm 4),
or Hoeffding's inequality, with a union bound over the N = 2T(T-1) (type, disagreeing pair) cells that can
enter the rule. The mass of disagreeing pairs never seen in calibration (where ARCANE falls back to the
evidence) is bounded by sample splitting (Lemma 3 of theory.tex).

Usage
  python bound.py                         # synthetic self-test (numpy only)
  python bound.py --npz FILE [--yU KEY --yS KEY --y KEY --fold KEY] [--delta 0.05] [--ci kl]
Keys are auto-detected when not given (e.g. yU/yU_c/cal_yU/pred_kNN-U, y/y_c/y_true, fold/cal_fold).
Labels may be integers or strings. The ARCANE rule g_hat is recomputed exactly as in arcane.Arbiter
(pooled class-conditional rates, ties -> U, unseen pairs -> evidence fallback).
"""
import argparse
import json
import sys

import numpy as np

LOG2 = np.log(2.0)


# --------------------------------------------------------------------------- confidence intervals
def _kl(p, q):
    p = np.clip(p, 0.0, 1.0)
    q = np.clip(q, 1e-15, 1 - 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log(p / q), 0.0)
        b = np.where(p < 1, (1 - p) * np.log((1 - p) / (1 - q)), 0.0)
    return a + b


def kl_upper(k, n, delta):
    """Largest q with n*kl(k/n || q) <= log(1/delta), q >= k/n (one-sided level delta)."""
    k, n = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float))
    p = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    lo, hi = p.copy(), np.ones_like(p)
    rhs = np.log(1.0 / delta) / np.maximum(n, 1)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ok = _kl(p, mid) <= rhs
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    return np.where(n > 0, hi, 1.0)


def kl_lower(k, n, delta):
    k, n = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float))
    return 1.0 - kl_upper(n - k, n, delta)


def interval(k, n, delta, method="kl"):
    """Two-sided interval [L, U] for a Bernoulli mean from k successes in n draws, level delta."""
    k, n = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float))
    p = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    if method == "kl":
        L, U = kl_lower(k, n, delta / 2), kl_upper(k, n, delta / 2)
    elif method == "hoeffding":
        w = np.sqrt(np.log(2.0 / delta) / (2 * np.maximum(n, 1)))
        L, U = p - w, p + w
    elif method == "bernstein":  # Maurer & Pontil (2009) Thm 4, applied to X and 1-X with delta/2 each
        nn = np.maximum(n, 2)
        V = nn / (nn - 1) * p * (1 - p)
        lg = np.log(4.0 / delta)
        w = np.sqrt(2 * V * lg / nn) + 7 * lg / (3 * (nn - 1))
        L, U = p - w, p + w
    else:
        raise ValueError(method)
    L = np.where(n > 0, np.clip(L, 0, 1), 0.0)
    U = np.where(n > 0, np.clip(U, 0, 1), 1.0)
    return L, U


# --------------------------------------------------------------------------- the ARCANE rule
def arcane_rule(yU, yS, y, T):
    """choose_U[a,b] and known[a,b] exactly as arcane.Arbiter(objective='balanced', min_count=1)."""
    N = np.bincount(y, minlength=T).astype(float)
    cnt = np.zeros((T, T, T))
    np.add.at(cnt, (yU, yS, y), 1.0)
    A = np.arange(T)
    w = 1.0 / np.maximum(N, 1.0)
    qa = cnt[A[:, None], A[None, :], A[:, None]] * w[:, None]
    qb = cnt[A[:, None], A[None, :], A[None, :]] * w[None, :]
    support = cnt[A[:, None], A[None, :], A[:, None]] + cnt[A[:, None], A[None, :], A[None, :]]
    return qa >= qb, support >= 1


def ba_of(pred, y, T):
    present = np.unique(y)
    return float(np.mean([(pred[y == t] == t).mean() for t in present]))


# --------------------------------------------------------------------------- bounds
def old_bound(m, delta):
    m = m[m > 0]
    T = len(m)
    eps = np.sqrt((2 * T * LOG2 + np.log(T / delta)) / (2 * m))
    return float(2.0 / T * eps.sum())


def compute_bounds(yU, yS, y, fold=None, delta=0.05, method="kl", seed=0, choose_U=None, known=None,
                   per_fold=False, rare_levels=(0, 1, 2, 4, 8)):
    """Bounds of Theorem 2 (theory.tex). Returns a dict.

    Estimand: q_t(a,b) = average class-conditional pair rate of the calibration cells of type t (pooled over
    folds, i.e. folds weighted by their share m_{t,f}/m_t; with per_fold=True folds are weighted equally and
    intervals are computed per fold). Cells must be independent given the fold models.
    Pairs (a,b) whose calibration counts of type a and of type b are both <= r are 'rare at level r'; their total
    mass per type is bounded by sample splitting (Lemma 3); the best level r in rare_levels is used (union bound).
    """
    yU, yS, y = (np.asarray(v).astype(int) for v in (yU, yS, y))
    T = int(max(yU.max(), yS.max(), y.max()) + 1)
    present = np.unique(y)
    Tp = len(present)
    if fold is None or not per_fold:
        fold_used = np.zeros(len(y), int)
    else:
        fold_used = np.asarray(fold)
    folds = np.unique(fold_used)
    nF = len(folds)
    if choose_U is None:
        choose_U, known = arcane_rule(yU, yS, y, T)
    m = np.bincount(y, minlength=T)
    R_lv = list(rare_levels)
    # (type t, disagreeing pair c in C_t) cells: t present, the other label any of the T labels (labels that are
    # predicted but never a true calibration label must be included: the fallback may pick them).
    N_cells = 2 * Tp * (T - 1)
    d_cell = delta / 2.0 / max(N_cells * nF, 1)                 # per (fold, type, pair), two-sided
    d_miss = delta / 2.0 / max(Tp * nF * len(R_lv), 1)          # per (fold, type, level), one-sided
    L = np.zeros((T, T, T)); U = np.zeros((T, T, T))           # [a, b, t]
    Mbar = np.zeros((len(R_lv), T))
    cnt_all = np.zeros((T, T, T)); np.add.at(cnt_all, (yU, yS, y), 1.0)
    rng = np.random.default_rng(seed)
    for t in present:
        fs = [f for f in folds if np.any((fold_used == f) & (y == t))]
        for f in fs:
            sel = (fold_used == f) & (y == t)
            n = int(sel.sum())
            c = np.zeros((T, T)); np.add.at(c, (yU[sel], yS[sel]), 1.0)
            l, u = interval(c, n, d_cell, method)
            L[:, :, t] += l / len(fs); U[:, :, t] += u / len(fs)
            idx = rng.permutation(np.where(sel)[0])
            h1, h2 = idx[: n // 2], idx[n // 2:]
            c1 = np.zeros((T, T)); np.add.at(c1, (yU[h1], yS[h1]), 1.0)
            dis2 = (yU[h2] != yS[h2]) & ((yU[h2] == t) | (yS[h2] == t))
            for j, r in enumerate(R_lv):
                inZ = dis2 & (c1[yU[h2], yS[h2]] <= r)
                Mbar[j, t] += float(kl_upper(inZ.sum(), len(h2), d_miss)) / len(fs)
    best = None
    for j, r in enumerate(R_lv):
        reg_obs = reg_cell = gU = gS = lU_cell = lS_cell = 0.0
        near = nobs = 0
        pres = np.zeros(T, bool); pres[present] = True
        for a in range(T):
            for b in range(T):
                if a == b or not (pres[a] or pres[b]):
                    continue   # pairs of two labels that are never a true label do not enter BA
                rare = (cnt_all[a, b, a] <= r) and (cnt_all[a, b, b] <= r)
                if known[a, b]:
                    if choose_U[a, b]:
                        loss = max(U[a, b, b] - L[a, b, a], 0.0); gs, gu = L[a, b, a] - U[a, b, b], 0.0
                    else:
                        loss = max(U[a, b, a] - L[a, b, b], 0.0); gs, gu = 0.0, L[a, b, b] - U[a, b, a]
                    lu, ls = -gu if gu < 0 else 0.0, -gs if gs < 0 else 0.0
                else:   # evidence fallback: any choice; loss <= max(q_a, q_b), gain >= -q_a (vs U), -q_b (vs S)
                    loss = max(U[a, b, a], U[a, b, b]); gs = gu = 0.0
                    lu, ls = U[a, b, a], U[a, b, b]
                if rare:
                    reg_cell += loss; lU_cell += lu; lS_cell += ls
                    gU += max(gu, 0.0); gS += max(gs, 0.0)
                else:
                    nobs += 1; near += loss > 0
                    reg_obs += loss; gU += gu; gS += gs
        mm = Mbar[j, present].sum()
        R = (reg_obs + min(reg_cell, mm)) / Tp
        LBU = (gU - min(lU_cell, mm)) / Tp
        LBS = (gS - min(lS_cell, mm)) / Tp
        cand = dict(level=r, R=R, R_main=reg_obs / Tp, R_rare_cellwise=reg_cell / Tp, R_rare_mass=mm / Tp,
                    n_pairs=nobs, n_near_tie=near, LBU=LBU, LBS=LBS)
        if best is None or R < best["R"]:
            best = cand
        if j == 0:
            bestLB = cand
        elif max(min(cand["LBU"], cand["LBS"]), -cand["R"]) > max(min(bestLB["LBU"], bestLB["LBS"]), -bestLB["R"]):
            bestLB = cand
    LBU, LBS = max(bestLB["LBU"], -1.0), max(bestLB["LBS"], -1.0)
    pred = np.where(yU == yS, yU, np.where(choose_U[yU, yS], yU, yS))
    cntw = cnt_all / np.maximum(m, 1)[None, None, :]
    bks = cntw.argmax(2)[yU, yS]
    third = (bks != yU) & (bks != yS)
    return dict(
        T=Tp, m_min=int(m[present].min()), m_median=float(np.median(m[present])), n_cal=int(len(y)),
        folds_used=int(nF), method=method, delta=delta,
        BA_U=ba_of(yU, y, T), BA_S=ba_of(yS, y, T), BA_ghat_insample=ba_of(pred, y, T),
        BA_BKS_unrestricted_insample=ba_of(bks, y, T), frac_cells_BKS_third_label=float(third.mean()),
        R_new=float(best["R"]), rare_level=int(best["level"]), R_new_main_part=float(best["R_main"]),
        R_new_rare_part=float(min(best["R_rare_cellwise"], best["R_rare_mass"])),
        n_nonrare_disagreeing_pairs=int(best["n_pairs"]), n_near_tie_pairs=int(best["n_near_tie"]),
        LB_gain_vs_U=float(LBU), LB_gain_vs_S=float(LBS),
        LB_gain_vs_max=float(max(min(bestLB["LBU"], bestLB["LBS"]), -best["R"])),
        old_bound=old_bound(m, delta),
    )


# --------------------------------------------------------------------------- synthetic test
def synth_q(T, rng, n_nb=3):
    """Class-conditional distributions of (A,B): each view is right with a type-specific accuracy and
    otherwise picks one of n_nb confusable neighbours; the two views err partly on the same cells."""
    q = np.zeros((T, T, T))
    for t in range(T):
        nb = rng.choice(np.delete(np.arange(T), t), n_nb, replace=False)
        w = rng.dirichlet(np.ones(n_nb))
        aU, aS = rng.uniform(0.6, 0.98), rng.uniform(0.6, 0.98)
        pU = np.zeros(T); pU[t] = aU; pU[nb] += (1 - aU) * w
        pS = np.zeros(T); pS[t] = aS; pS[nb] += (1 - aS) * rng.dirichlet(np.ones(n_nb))
        rho = 0.5  # shared errors
        joint = (1 - rho) * np.outer(pU, pS)
        shared = np.zeros((T, T))
        shared[t, t] = min(aU, aS)
        rest = 1 - min(aU, aS)
        for j, b in enumerate(nb):
            shared[b, b] += rest * w[j]
        joint += rho * shared
        # fix marginals approximately is unnecessary: any joint is a valid q_t
        q[:, :, t] = joint / joint.sum()
    return q


def true_ba(q, rule):
    T = q.shape[0]
    A = np.arange(T)
    out = 0.0
    for t in range(T):
        out += q[t, t, t]
        for a in range(T):
            for b in range(T):
                if a != b and rule(a, b) == t:
                    out += q[a, b, t]
    return out / T


def true_ba_present(q, rule, present):
    out = 0.0
    T = q.shape[0]
    for t in present:
        out += q[t, t, t]
        for a in range(T):
            for b in range(T):
                if a != b and rule(a, b) == t:
                    out += q[a, b, t]
    return out / len(present)


def sample(q, m, rng, nfold=2):
    T = q.shape[0]
    yU, yS, y, fold = [], [], [], []
    for t in range(T):
        idx = rng.choice(T * T, m[t], p=q[:, :, t].ravel())
        yU.append(idx // T); yS.append(idx % T); y.append(np.full(m[t], t)); fold.append(rng.integers(0, nfold, m[t]))
    return map(np.concatenate, (yU, yS, y, fold))


def selftest():
    rows = []
    for T, med in [(10, 2000), (30, 1000), (61, 1000), (61, 300)]:
        rng = np.random.default_rng(T + med)
        q = synth_q(T, rng)
        m = np.maximum(30, rng.lognormal(np.log(med), 1.0, T).astype(int))
        yU, yS, y, fold = sample(q, m, rng)
        choose_U, known = arcane_rule(yU, yS, y, T)
        gstar = lambda a, b: a if q[a, b, a] >= q[a, b, b] else b
        ghat = lambda a, b: (a if choose_U[a, b] else b) if known[a, b] else (a if q[a, b, a] < q[a, b, b] else b)  # adversarial fallback
        BA_star, BA_hat = true_ba(q, gstar), true_ba(q, ghat)
        BA_Uv, BA_Sv = true_ba(q, lambda a, b: a), true_ba(q, lambda a, b: b)
        res = {}
        for meth in ("kl", "bernstein", "hoeffding"):
            res[meth] = compute_bounds(yU, yS, y, fold, 0.05, meth, choose_U=choose_U, known=known)
        res["kl_perfold"] = compute_bounds(yU, yS, y, fold, 0.05, "kl", choose_U=choose_U, known=known, per_fold=True)
        r = res["kl"]
        rows.append(dict(T=T, m_median=int(np.median(m)), m_min=int(m.min()), true_regret=BA_star - BA_hat,
                         true_gain_vs_max=BA_hat - max(BA_Uv, BA_Sv), old=r["old_bound"],
                         R_kl=r["R_new"], R_kl_fold=res["kl_perfold"]["R_new"], R_bern=res["bernstein"]["R_new"],
                         R_hoeff=res["hoeffding"]["R_new"], R_main=r["R_new_main_part"], R_rare=r["R_new_rare_part"],
                         lvl=r["rare_level"], near_tie=r["n_near_tie_pairs"],
                         pairs=r["n_nonrare_disagreeing_pairs"], LB_max=r["LB_gain_vs_max"],
                         LB_U=r["LB_gain_vs_U"], LB_S=r["LB_gain_vs_S"]))
        assert BA_star - BA_hat <= r["R_new"] + 1e-12, "bound violated"
    hdr = list(rows[0].keys())
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]) for h in hdr))
    # coverage check of the bound: repeat sampling, count violations of regret and of the gain lower bound
    rng = np.random.default_rng(7)
    T = 20
    q = synth_q(T, rng)
    m = np.maximum(30, rng.lognormal(np.log(400), 1.0, T).astype(int))
    gstar = lambda a, b: a if q[a, b, a] >= q[a, b, b] else b
    BA_Uv, BA_Sv = true_ba(q, lambda a, b: a), true_ba(q, lambda a, b: b)
    viol_R = viol_LB = 0
    regrets, Rs = [], []
    for rep in range(40):
        yU, yS, y, fold = sample(q, m, rng)
        cu, kn = arcane_rule(yU, yS, y, T)
        gh = lambda a, b: (a if cu[a, b] else b) if kn[a, b] else (a if q[a, b, a] < q[a, b, b] else b)
        BAh = true_ba(q, gh)
        r = compute_bounds(yU, yS, y, fold, 0.05, "kl", seed=rep, choose_U=cu, known=kn)
        regrets.append(true_ba(q, gstar) - BAh); Rs.append(r["R_new"])
        viol_R += (true_ba(q, gstar) - BAh) > r["R_new"]
        viol_LB += (BAh - max(BA_Uv, BA_Sv)) < r["LB_gain_vs_max"]
    # labels predicted by a view but never a true calibration label (e.g. a type present in R_{-f} only):
    # pairs (a, b) with b absent must be charged too (bug in the first version of compute_bounds)
    rng = np.random.default_rng(11)
    T = 12
    q = synth_q(T, rng)
    absent = [10, 11]
    present = np.array([t for t in range(T) if t not in absent])
    m = np.zeros(T, int); m[present] = np.maximum(30, rng.lognormal(np.log(150), 1.0, len(present)).astype(int))
    viol_abs = 0
    for rep in range(40):
        yU, yS, y, fold = sample(q, m, rng)
        cu, kn = arcane_rule(yU, yS, y, T)
        gh = lambda a, b: (a if cu[a, b] else b) if kn[a, b] else (a if q[a, b, a] < q[a, b, b] else b)
        gs = lambda a, b: a if q[a, b, a] >= q[a, b, b] else b
        reg = (true_ba_present(q, gs, present) - true_ba_present(q, gh, present))
        r = compute_bounds(yU, yS, y, fold, 0.05, "kl", seed=rep, choose_U=cu, known=kn)
        viol_abs += reg > r["R_new"]
    print(f"absent-label check (T=12, 2 labels never true, 40 reps): violations of R_new {viol_abs}/40")
    print(f"\nrepeated sampling (T=20, 40 reps): mean true regret {np.mean(regrets):.4f}, mean R_new {np.mean(Rs):.4f}, "
          f"violations of R_new {viol_R}/40, of LB_max {viol_LB}/40")


# --------------------------------------------------------------------------- CLI for real data
CAND = dict(yU=["yU", "yU_c", "cal_yU", "pred_kNN-U", "kNN-U", "yU_cal"],
            yS=["yS", "yS_c", "cal_yS", "pred_kNN-S", "kNN-S", "yS_cal"],
            y=["y", "y_c", "cal_y", "y_true", "y_cal", "label"],
            fold=["fold", "cal_fold", "fold_c", "f"])


def _pick(z, key, name, required=True):
    if key:
        if key not in z:
            raise KeyError(f"key {key!r} not in npz (has {list(z.keys())})")
        return z[key]
    for k in CAND[name]:
        if k in z:
            return z[k]
    if required:
        raise KeyError(f"could not find {name}: tried {CAND[name]}; npz has {list(z.keys())}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz")
    ap.add_argument("--yU"); ap.add_argument("--yS"); ap.add_argument("--y"); ap.add_argument("--fold")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--ci", default="all", choices=["kl", "bernstein", "hoeffding", "all"])
    ap.add_argument("--per_fold", action="store_true",
                    help="intervals per fold, estimand = equal-weight mixture of the 2 fold models' pair rates")
    ap.add_argument("--mask_known", action="store_true", help="drop cells whose true label is not a predicted class")
    A = ap.parse_args()
    if not A.npz:
        selftest()
        return
    z = np.load(A.npz, allow_pickle=True)
    yU, yS, y = _pick(z, A.yU, "yU"), _pick(z, A.yS, "yS"), _pick(z, A.y, "y")
    fold = _pick(z, A.fold, "fold", required=False)
    if not (len(yU) == len(yS) == len(y)):
        raise ValueError("yU, yS, y lengths differ")
    if yU.dtype.kind in "USO" or y.dtype.kind in "USO":
        classes = np.array(sorted(set(map(str, yU)) | set(map(str, yS)) | set(map(str, y))))
        if "classes" in z:
            classes = np.array(list(map(str, z["classes"])))
        ci = {c: i for i, c in enumerate(classes)}
        keep = np.array([str(v) in ci for v in y]) & np.array([str(v) in ci for v in yU]) & np.array([str(v) in ci for v in yS])
        if (~keep).any():
            print(f"dropping {(~keep).sum()} cells whose label is not among the classes", file=sys.stderr)
        yU, yS, y = (np.array([ci[str(v)] for v in a[keep]]) for a in (yU, yS, y))
        fold = fold[keep] if fold is not None else None
    meths = ["kl", "bernstein", "hoeffding"] if A.ci == "all" else [A.ci]
    if A.per_fold and fold is None:
        raise SystemExit("--per_fold needs a fold array in the npz")
    out = {m: compute_bounds(yU, yS, y, fold, A.delta, m, per_fold=A.per_fold) for m in meths}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
