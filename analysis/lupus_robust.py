#!/usr/bin/env python
"""Robustness of the discovery-cohort (tag lupusfull) ARCANE findings. Run from ~/arcane:
    python analysis/lupus_robust.py --tag lupusfull --seeds 0 1 2
 (a) Seed stability of flags (p <= alpha) for p_unknown_ctrl, p_unknown_ctrl_U, p_unknown_bonf: per-cell pairwise
     Jaccard, Fleiss kappa, all-seed agreement; donor flag rate (composition-standardized) pairwise Spearman and
     ICC(2,1)/ICC(3,1); consensus (>= 2 of 3) flag rate SLE vs healthy-query donors (within processing-cohort
     permutation) and the false-alarm rate on healthy query donors.
 (b) Glucocorticoid sensitivity: donor GC score = mean z of glucocorticoid_core (vs reference cells of the same
     label) over all query cells; main flag excess and C1Q+ ncM / IFN-gamma-biased monocyte fractions (threshold =
     q99 of reference cells of the label) re-tested excluding SLE donors in the top GC quartile and adjusting for GC
     (+ processing cohort + ancestry).
 (c) ISG-adjusted SLE effect on donor flag rate: OLS (HC3) flag ~ SLE + ISG + cohort + ancestry + sex + age, with
     donor-bootstrap CI.
Donor = individual (donor_id); query cells of individuals that are also reference donors are excluded.
Cell scores use the true fine labels of lupusfull (as in the discovery analysis; --score_label arcane to use
label_arcane); composition standardization uses label_arcane.
Outputs in --out: summary.json, donor_table.csv, seed_pairs.csv, log.txt; KEY lines on stdout.
"""
import argparse
import json
import os
import sys
import traceback
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import val_common as vc  # noqa: E402
from val_common import log, fmt, eci  # noqa: E402

PCOLS = ["p_unknown_ctrl", "p_unknown_ctrl_U", "p_unknown_bonf"]
MAIN = "p_unknown_ctrl"


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tag", default="lupusfull")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--res_dir", default="res2")
    ap.add_argument("--h5ad", default=None)
    ap.add_argument("--genesets", default="genesets.json")
    ap.add_argument("--out", default="res3/lupus_robust")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--q", type=float, default=99.0)
    ap.add_argument("--score_label", choices=["true", "arcane"], default="true")
    ap.add_argument("--sd_floor", type=float, default=0.1)
    ap.add_argument("--zclip", type=float, default=10.0)
    ap.add_argument("--min_ref_cells", type=int, default=50)
    ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--min_donor_cells", type=int, default=50)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--disc_sign", default="C1Q_ncM=1,IFNg_mono=1")
    a = ap.parse_args(argv)
    a.seeds = [int(s) for x in a.seeds for s in str(x).split(",") if s.strip()]
    return a


A = None
R = {"errors": {}}


def guarded(name, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001
        R["errors"][name] = repr(e)
        log(f"!!! {name} failed: {e!r}\n{traceback.format_exc()}")
        return None


def save():
    with open(os.path.join(A.out, "summary.json"), "w") as f:
        json.dump(vc.clean(R), f, indent=1, default=str)


def build(gs):
    CL = vc.load_cells(A.root, A.res_dir, A.tag, A.seeds, PCOLS, A.alpha)
    R["seeds_used"] = CL["seeds"]
    isg_g, gc_g = gs["ISG_typeI"], gs.get("glucocorticoid_core", [])
    genes = list(dict.fromkeys(isg_g + gs.get("ISG_typeI_core_specific", []) + gc_g + vc.C1Q + vc.IFNG_UP))
    h5 = A.h5ad or os.path.join(A.root, "data", "std", f"{A.tag}.h5ad")
    want = dict(label=(["label"], []), role=(["role"], []), cond=(["condition", "disease"], [r"disease"]),
                indiv=(["donor_id", "ind_cov"], [r"individual"]), batch=(["batch"], []),
                cohort=(["Processing_Cohort", "processing_cohort", "cohort"], [r"process.*cohort", r"cohort"]),
                anc=(["self_reported_ethnicity", "pop_cov", "ancestry", "ethnicity", "race"], [r"ethnic", r"ancestry"]),
                sex=(["sex", "Sex", "sex_cov", "gender"], [r"^sex", r"gender"]),
                age=(["Age", "age", "age_cov", "age_years"], [r"^age"]),
                dev=(["development_stage"], [r"development_stage"]))
    O, present, E, found = vc.load_expression(h5, genes, A.chunk, want)
    R["obs_columns"] = found
    if found.get("indiv") is None:
        O["indiv"] = O["batch"]
    isref = O.role.values == "ref"
    isq = O.role.values == "query"
    names = CL["names"]
    qn = pd.Index(O.index[isq])
    if qn.isin(names).mean() > 0.99:
        pos = pd.Series(np.arange(len(names)), index=names).reindex(qn).values
    elif len(qn) == len(names):
        log("WARNING: aligning query cells by position")
        pos = np.arange(len(names), dtype=float)
    else:
        raise ValueError("cannot align query cells with the cells npz")
    qi = np.where(isq)[0]
    ok = np.isfinite(pos)
    pi = pos[ok].astype(int)
    lab_arc = np.full(len(O), "na", object)
    lab_arc[qi[ok]] = CL["label_arcane"][pi]
    lab = O.label.values.astype(object).copy()
    if A.score_label == "arcane":
        lab[qi[ok]] = CL["label_arcane"][pi]
    flags = {}
    for c, F in CL["flags"].items():
        arr = np.full((len(O), F.shape[1]), np.nan)
        arr[qi[ok]] = F.values[pi]
        flags[c] = arr
    lab = lab.astype(str)
    Z, nref = vc.z_by_label(E, lab, isref, A.sd_floor, A.zclip, A.min_ref_cells)
    del E
    gi = {g: i for i, g in enumerate(present)}
    cell = pd.DataFrame(index=O.index)
    cell["lab"], cell["lab_arc"], cell["isref"] = lab, lab_arc.astype(str), isref
    both = set(O.indiv.values[isref]) & set(O.indiv.values[isq])
    R["indiv_in_both_roles"] = sorted(map(str, both))[:50]
    R["n_indiv_in_both_roles"] = len(both)
    cell["isq"] = isq & np.isin(np.arange(len(O)), qi[ok]) & ~np.isin(O.indiv.values, list(both))
    for c in ["indiv", "cohort", "anc", "sex", "age", "dev"]:
        cell[c] = O[c].values
    cell["sle"] = [1 if vc.is_sle(x) else (0 if str(x).lower() in ("normal", "healthy", "control") else -1)
                   for x in O.cond.values]
    cell["isg"], _, _ = vc.gene_score(Z, gi, isg_g)
    g_, R["gc_genes"], _ = vc.gene_score(Z, gi, gc_g)
    cell["gc"] = g_ if g_ is not None else np.nan
    specs = vc.state_specs(gs, sorted(set(lab[isref])))
    for st in specs:
        s, st["up_used"], st["down_used"] = vc.gene_score(Z, gi, st["up"], st["down"])
        cell[f"s_{st['name']}"] = s if s is not None else np.nan
        rest = [g for g in isg_g if g not in set(st["up"]) | set(st["down"])]
        si, _, _ = vc.gene_score(Z, gi, rest)
        cell[f"isg_{st['name']}"] = si if si is not None else np.nan
        log(f"state {st['name']}: labels {st['labels']}")
    return cell, flags, specs


def donor_meta(cell):
    q = cell[cell.isq]
    g = q.groupby("indiv")
    D = g.agg(sle=("sle", "first"), n_cells=("lab", "size"))
    for c in ["cohort", "anc", "sex", "age", "dev"]:
        D[c] = g[c].agg(lambda x: x.value_counts().index[0]).astype(str)
    a = D.age.map(vc.parse_age)
    if a.notna().sum() < 0.5 * len(D):
        a = D.dev.map(vc.parse_age)
    D["age_num"] = a
    D["age_z"] = (a - a.mean()) / (a.std() if a.std() > 0 else 1)
    D["age_z"] = D.age_z.fillna(0.0)
    for c in ["cohort", "anc", "sex"]:
        D[c] = D[c].replace({"nan": "unknown", "na": "unknown", "": "unknown"})
    return D


def cats_available(D, cats):
    return [c for c in cats if D[c].nunique() > 1]


# ------------------------------------------------------------------------------------------------
def part_a(cell, flags, D, w):
    q = cell.isq.values & cell.indiv.isin(D.index[D.use]).values
    T, pairs = {}, []
    U = D[D.use]
    for c in PCOLS:
        if c not in flags:
            continue
        F = flags[c][q]
        ok = np.isfinite(F).all(1)
        Fo = F[ok]
        k = F.shape[1]
        res = dict(n_cells=int(ok.sum()), rate_per_seed=[float(v) for v in Fo.mean(0)])
        if k >= 2:
            res["fleiss_kappa"] = vc.fleiss_kappa_binary(Fo)
            res["all_seed_agree"] = float(((Fo.sum(1) == 0) | (Fo.sum(1) == k)).mean())
            jac, rhos = [], []
            dr = []
            for j in range(k):
                dr.append(vc.std_rate(F[:, j], cell.lab_arc.values[q], cell.indiv.values[q], w)["std"].reindex(U.index))
            for a_, b_ in combinations(range(k), 2):
                fa, fb = Fo[:, a_] == 1, Fo[:, b_] == 1
                ja = float((fa & fb).sum() / max((fa | fb).sum(), 1))
                rho = stats.spearmanr(dr[a_], dr[b_], nan_policy="omit")[0]
                jac.append(ja)
                rhos.append(rho)
                pairs.append(dict(pcol=c, seed_a=R["seeds_used"][a_], seed_b=R["seeds_used"][b_], jaccard=ja,
                                  donor_spearman=float(rho), cell_agree=float((fa == fb).mean())))
            res["jaccard_pairs"], res["jaccard_mean"] = jac, float(np.mean(jac))
            res["donor_spearman_pairs"], res["donor_spearman_min"] = rhos, float(np.min(rhos))
            res["icc"] = vc.icc(np.column_stack([d.values for d in dr]))
            for j in range(k):
                D.loc[U.index, f"flag_{c}_seed{R['seeds_used'][j]}"] = dr[j].values
        cons = vc.consensus(flags[c])
        cell[f"f_{c}"] = cons
        sr = vc.std_rate(cons[q], cell.lab_arc.values[q], cell.indiv.values[q], w)
        D[f"flag_{c}_std"], D[f"flag_{c}_raw"] = sr["std"].reindex(D.index), sr["raw"].reindex(D.index)
        U = D[D.use]
        res["consensus"] = vc.group_compare(U[f"flag_{c}_std"].values, U.sle.values, U.cohort.values, A.nboot, A.nperm)
        res["consensus"]["fa_healthy_donor_mean_raw"] = float(U.loc[U.sle == 0, f"flag_{c}_raw"].mean())
        res["consensus"]["fa_healthy_donor_mean_std"] = float(U.loc[U.sle == 0, f"flag_{c}_std"].mean())
        hq = q & (cell.sle.values == 0)
        res["consensus"]["fa_healthy_cells"] = float(np.nanmean(cons[hq])) if hq.any() else np.nan
        T[c] = res
        log(f"(a) {c}: kappa {fmt(res.get('fleiss_kappa'))} Jaccard {fmt(res.get('jaccard_mean'))} "
            f"ICC {fmt(res.get('icc', {}).get('icc2_1'))} diff {fmt(res['consensus'].get('diff'))}")
    R["a_seed_stability"] = T
    if pairs:
        pd.DataFrame(pairs).to_csv(os.path.join(A.out, "seed_pairs.csv"), index=False)


def state_frac(cell, st, pos, D):
    q = cell.isq.values & np.isin(cell.lab.values, st["labels"]) & np.isfinite(pos)
    d = pd.DataFrame(dict(indiv=cell.indiv.values[q], pos=pos[q]))
    g = d.groupby("indiv")
    S = pd.DataFrame(dict(k=g.pos.sum(), n=g.size())).reindex(D.index)
    S["frac"] = (S.k / S.n).where(S.n >= A.min_cells)
    S["logit"] = np.where(S.frac.notna(), vc.logit_frac(S.k.fillna(0), S.n.fillna(1)), np.nan)
    return S


def state_test(Dd, name, sign, cats, extra=(), sub=None):
    U = Dd[Dd.use & Dd[f"{name}_frac"].notna()]
    if sub is not None:
        U = U[U.index.isin(sub)]
    g = U.sle.values
    out = dict(n_sle=int(g.sum()), n_healthy=int((1 - g).sum()))
    if g.sum() < 2 or (1 - g).sum() < 2:
        return out
    out["hl"], out["hl_lo"], out["hl_hi"] = vc.hl_boot(U[f"{name}_frac"][g == 1], U[f"{name}_frac"][g == 0], A.nboot)
    pr = vc.strat_perm(U[f"{name}_logit"].values, g, U.cohort.values, A.nperm)
    out["p_perm"], out["p_one_sided"] = pr["p"], (pr["p_greater"] if sign > 0 else pr["p_less"])
    o = vc.ols_effect(U, f"{name}_logit", "sle", [f"isg_{name}"] + list(extra), cats, A.nboot)
    out.update({f"adj_{k}": v for k, v in o.items() if not isinstance(v, (list, dict))})
    out["adj_covariates"] = o.get("covariates")
    if "p_greater" in o:
        out["adj_p_one_sided"] = o["p_greater"] if sign > 0 else o["p_less"]
    return out


def part_bc(cell, D, specs, w, signs):
    q = cell.isq.values & cell.indiv.isin(D.index[D.use]).values
    D["isg"] = vc.std_rate(cell.isg.values[q], cell.lab_arc.values[q], cell.indiv.values[q], w)["std"].reindex(D.index)
    D["gc"] = pd.Series(cell.gc.values[q]).groupby(cell.indiv.values[q]).mean().reindex(D.index)
    U = D[D.use]
    y = f"flag_{MAIN}_std"
    cats_ca = cats_available(U, ["cohort", "anc"])
    cats_full = cats_available(U, ["cohort", "anc", "sex"])
    # (c) ISG-adjusted SLE effect on donor flag rate
    C = dict(unadjusted=vc.ols_effect(U, y, "sle", [], [], A.nboot),
             covariates_only=vc.ols_effect(U, y, "sle", ["age_z"], cats_full, A.nboot),
             isg_only=vc.ols_effect(U, y, "sle", ["isg"], [], A.nboot),
             full=vc.ols_effect(U, y, "sle", ["isg", "age_z"], cats_full, A.nboot),
             full_plus_gc=vc.ols_effect(U, y, "sle", ["isg", "age_z", "gc"], cats_full, A.nboot),
             spearman_flag_isg_sle=vc.spearman_ci(U.loc[U.sle == 1, y], U.loc[U.sle == 1, "isg"], A.nboot),
             lrt_flag_given_isg_covs=vc.logistic_lrt(U, "sle", ["isg", "age_z"], y, cats_full))
    f = C["full"]
    cf = f.get("coef", np.nan)
    word = "higher" if np.isfinite(cf) and cf >= 0 else "lower"
    C["statement"] = (f"Adjusted for donor ISG score, processing cohort, ancestry, sex and age, SLE donors had a "
                      f"flag rate {fmt(abs(cf))} {word} (absolute fraction, estimate {fmt(cf)}; 95% CI {fmt(f.get('lo'))} to "
                      f"{fmt(f.get('hi'))}, HC3 p = {fmt(f.get('p'), 4)}; donor bootstrap CI {fmt(f.get('boot_lo'))} to "
                      f"{fmt(f.get('boot_hi'))}; n = {f.get('n')} donors) than healthy donors; unadjusted difference "
                      f"{fmt(C['unadjusted'].get('coef'))}.")
    R["c_isg_adjusted"] = C
    log("(c)", C["statement"])
    # (b) glucocorticoid sensitivity
    B = dict(gc_sle_vs_healthy=vc.group_compare(U.gc.values, U.sle.values, U.cohort.values, A.nboot, A.nperm),
             spearman_gc_flag_sle=vc.spearman_ci(U.loc[U.sle == 1, "gc"], U.loc[U.sle == 1, y], A.nboot),
             spearman_gc_isg_sle=vc.spearman_ci(U.loc[U.sle == 1, "gc"], U.loc[U.sle == 1, "isg"], A.nboot))
    thr = U.loc[U.sle == 1, "gc"].quantile(0.75)
    excl = U.index[(U.sle == 1) & (U.gc >= thr)]
    keep = U.index.difference(excl)
    B["gc_q75_sle"], B["n_excluded"] = float(thr), int(len(excl))
    Uk = U.loc[keep]
    B["flag_all"] = vc.group_compare(U[y].values, U.sle.values, U.cohort.values, A.nboot, A.nperm)
    B["flag_excl"] = vc.group_compare(Uk[y].values, Uk.sle.values, Uk.cohort.values, A.nboot, A.nperm)
    B["flag_adj_gc"] = vc.ols_effect(U, y, "sle", ["gc"], cats_ca, A.nboot)
    B["flag_adj_gc_isg"] = vc.ols_effect(U, y, "sle", ["gc", "isg"], cats_ca, A.nboot)
    states = []
    for st in specs:
        name = st["name"]
        if not st["labels"] or cell[f"s_{name}"].isna().all():
            continue
        D[f"isg_{name}"] = vc.std_rate(cell[f"isg_{name}"].values[q], cell.lab_arc.values[q], cell.indiv.values[q],
                                       w)["std"].reindex(D.index)
        pos, thr_ = vc.threshold_ref(cell[f"s_{name}"].values, cell.lab.values, cell.isref.values, st["labels"], A.q)
        S = state_frac(cell, st, pos, D)
        D[f"{name}_frac"], D[f"{name}_logit"], D[f"{name}_n"] = S.frac, S.logit, S.n
        sign = signs.get(name, 1)
        r = dict(state=name, labels=st["labels"], thresholds=thr_,
                 all=state_test(D, name, sign, cats_ca), excl_topGC=state_test(D, name, sign, cats_ca, sub=keep),
                 adj_gc=state_test(D, name, sign, cats_ca, extra=("gc",)))
        hq = q & np.isin(cell.lab.values, st["labels"]) & (cell.sle.values == 0) & np.isfinite(pos)
        r["healthy_pos_rate_cells"] = float(np.nanmean(pos[hq])) if hq.any() else np.nan
        states.append(r)
        log(f"(b) {name}: all HL {fmt(r['all'].get('hl'), 4)} p1 {fmt(r['all'].get('p_one_sided'), 4)}; excl "
            f"{fmt(r['excl_topGC'].get('hl'), 4)} p1 {fmt(r['excl_topGC'].get('p_one_sided'), 4)}; adjGC "
            f"{fmt(r['adj_gc'].get('adj_coef'))} p1 {fmt(r['adj_gc'].get('adj_p_one_sided'), 4)}")
    B["states"] = states
    R["b_glucocorticoid"] = B


def key_lines():
    K = []
    for c, r in R.get("a_seed_stability", {}).items():
        cs = r.get("consensus", {})
        K.append(f"KEY a {c}: kappa={fmt(r.get('fleiss_kappa'))} Jaccard={fmt(r.get('jaccard_mean'))} "
                 f"agree_all={fmt(r.get('all_seed_agree'))} donor_rho_min={fmt(r.get('donor_spearman_min'))} "
                 f"ICC2,1={fmt(r.get('icc', {}).get('icc2_1'))} | consensus FA_healthy={fmt(cs.get('fa_healthy_donor_mean_raw'))}"
                 f" SLE-healthy={eci(cs.get('diff'), cs.get('diff_lo'), cs.get('diff_hi'))} cohort-strat p={fmt(cs.get('p_perm'), 4)}"
                 f" n={cs.get('n_sle')}/{cs.get('n_healthy')}")
    B = R.get("b_glucocorticoid")
    if B:
        e, a, al = B["flag_excl"], B["flag_adj_gc"], B["flag_all"]
        K.append(f"KEY b GC: SLE-healthy GC={fmt(B['gc_sle_vs_healthy'].get('diff'))} p={fmt(B['gc_sle_vs_healthy'].get('p_perm'), 4)}"
                 f" rho(GC,flag|SLE)={fmt(B['spearman_gc_flag_sle'].get('rho'))} | flag diff all={fmt(al.get('diff'))}"
                 f" excl top-GC (n={B['n_excluded']})={eci(e.get('diff'), e.get('diff_lo'), e.get('diff_hi'))} p={fmt(e.get('p_perm'), 4)}"
                 f" | GC+cohort+ancestry-adj={eci(a.get('coef'), a.get('lo'), a.get('hi'))} p={fmt(a.get('p'), 4)}")
        for r in B["states"]:
            x, ex, ag = r["all"], r["excl_topGC"], r["adj_gc"]
            K.append(f"KEY b {r['state']} ref_q99: HL={fmt(x.get('hl'), 4)} p1={fmt(x.get('p_one_sided'), 4)} "
                     f"ISGadj={fmt(x.get('adj_coef'), 2)} p1={fmt(x.get('adj_p_one_sided'), 4)} | excl top-GC HL="
                     f"{fmt(ex.get('hl'), 4)} p1={fmt(ex.get('p_one_sided'), 4)} adj p1={fmt(ex.get('adj_p_one_sided'), 4)}"
                     f" | +GC adj={eci(ag.get('adj_coef'), ag.get('adj_lo'), ag.get('adj_hi'), 2)} p1={fmt(ag.get('adj_p_one_sided'), 4)}")
    C = R.get("c_isg_adjusted")
    if C:
        f, l = C["full"], C["lrt_flag_given_isg_covs"]
        K.append(f"KEY c ISG-adjusted SLE effect on flag rate (ISG+cohort+ancestry+sex+age): {eci(f.get('coef'), f.get('lo'), f.get('hi'))}"
                 f" p={fmt(f.get('p'), 4)} boot[{fmt(f.get('boot_lo'))},{fmt(f.get('boot_hi'))}] n={f.get('n')} | unadj="
                 f"{fmt(C['unadjusted'].get('coef'))} ISG-only adj={fmt(C['isg_only'].get('coef'))} +GC={fmt(C['full_plus_gc'].get('coef'))}"
                 f" | LRT flag|ISG+covs p={fmt(l.get('p'), 4)}{' (separation)' if l.get('separation_warning') else ''}")
        K.append("KEY c statement: " + C["statement"])
    if R["errors"]:
        K.append(f"KEY errors: {list(R['errors'])}")
    return K


def main(argv=None):
    global A
    A = parse_args(argv)
    if not os.path.isabs(A.out):
        A.out = os.path.join(A.root, A.out)
    os.makedirs(A.out, exist_ok=True)
    vc.LOGF[0] = open(os.path.join(A.out, "log.txt"), "w")
    R["args"] = vars(A)
    gs = vc.load_genesets(A.root, A.genesets)
    signs = {k: float(v) for k, v in (x.split("=") for x in A.disc_sign.split(",") if "=" in x)}
    cell, flags, specs = build(gs)
    D = donor_meta(cell)
    D["use"] = (D.n_cells >= A.min_donor_cells) & D.sle.isin([0, 1])
    R["n_donors"] = dict(total=int(len(D)), used=int(D.use.sum()), sle=int((D.use & (D.sle == 1)).sum()),
                         healthy=int((D.use & (D.sle == 0)).sum()))
    log("donors:", R["n_donors"])
    h = cell.isq.values & cell.indiv.isin(D.index[D.use & (D.sle == 0)]).values
    w = vc.healthy_weights(cell.lab_arc.values, cell.indiv.values, h)
    R["healthy_composition"] = w.to_dict()
    guarded("a", part_a, cell, flags, D, w)
    save()
    guarded("bc", part_bc, cell, D, specs, w, signs)
    D.to_csv(os.path.join(A.out, "donor_table.csv"))
    K = key_lines()
    R["KEY"] = K
    save()
    print("\n".join(K), flush=True)
    log("LUPUS_ROBUST_DONE", A.out)


if __name__ == "__main__":
    main()
