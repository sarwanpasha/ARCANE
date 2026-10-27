#!/usr/bin/env python
"""Pre-registered validation of the discovery-cohort (lupusfull) findings in the independent SLE cohort GSE135779
(Nehar-Belaid 2020; tag lupus_nb). Run from ~/arcane:
    python analysis/nb_validate.py --tag lupus_nb --seeds 0 1 2
Donors (GSM samples) are the statistical units; age group (child / adult) is the stratum throughout.

Tests (markers / thresholds fixed in advance):
 T1 ARCANE flag rate per donor (consensus >= 2 of 3 seeds, p <= alpha), composition-standardized to the healthy
    query composition of label_arcane; SLE vs healthy (difference, stratified donor-bootstrap CI, within-age
    permutation p, per age group); false-alarm rate on healthy donors. For p_unknown_ctrl (LODO healthy
    calibration), p_unknown_ctrl_U (scVI view only) and p_unknown_bonf (reference-calibrated).
 T2 Interferon score (mean z of ISG_typeI vs reference cells of the same label; donor = composition-weighted mean).
    Spearman(flag rate, ISG) within SLE; OLS flag ~ SLE + ISG + age group (HC3 + bootstrap CI);
    logistic LRT SLE ~ ISG (+ age) + flag vs SLE ~ ISG (+ age).
 T3 Candidate states C1Q+ ncM and IFN-gamma-biased monocytes; thresholds (a) q99 of reference cells of the label,
    (b) q99 of healthy query cells (LODO for healthy donors). SLE vs healthy (HL shift, within-age permutation p on
    logit fraction), ISG-adjusted OLS, share of state+ cells flagged by ARCANE vs an ISG top-n detector, MH OR.
    Replicated = same sign as discovery and one-sided p < 0.05.
 T4 View divergence: U-only deviant (p_unknown_ctrl_U <= a and p_unknown_ctrl > a) and U-deviant fractions; AUROC.
 T5 Glucocorticoid sensitivity: donor GC score (mean z of glucocorticoid_core over all cells); T1/T3 excluding SLE
    donors in the top GC quartile and adjusting for GC.
Outputs in --out: summary.json, donor_table.csv, state_tests.csv, log.txt; KEY lines on stdout.
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import val_common as vc  # noqa: E402
from val_common import log, fmt, eci  # noqa: E402

PCOLS = ["p_unknown_ctrl", "p_unknown_ctrl_U", "p_unknown_bonf", "p_unknown_U_bonf"]
T1_COLS = ["p_unknown_ctrl", "p_unknown_ctrl_U", "p_unknown_bonf"]
MAIN = "p_unknown_ctrl"


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tag", default="lupus_nb")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--res_dir", default="res2")
    ap.add_argument("--emb_dir", default="emb")
    ap.add_argument("--h5ad", default=None, help="default <root>/data/std/<tag>.h5ad")
    ap.add_argument("--donors_csv", default=None, help="default <root>/data/std/<tag>_donors.csv")
    ap.add_argument("--genesets", default="genesets.json")
    ap.add_argument("--out", default="res3/nb_validate")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--q", type=float, default=99.0)
    ap.add_argument("--sd_floor", type=float, default=0.1)
    ap.add_argument("--zclip", type=float, default=10.0)
    ap.add_argument("--min_ref_cells", type=int, default=50)
    ap.add_argument("--min_cells", type=int, default=20, help="min state-label cells per donor")
    ap.add_argument("--min_donor_cells", type=int, default=50)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--disc_sign", default="C1Q_ncM=1,IFNg_mono=1",
                    help="direction of the SLE effect in the discovery cohort per state")
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


# ------------------------------------------------------------------------------------------------
def build(gs):
    root = A.root
    CL = vc.load_cells(root, A.res_dir, A.tag, A.seeds, PCOLS, A.alpha)
    R["seeds_used"] = CL["seeds"]
    R["label_arcane_seed_agreement"] = CL["label_seed_agreement"]
    isg_g = gs["ISG_typeI"]
    core = gs.get("ISG_typeI_core_specific", [])
    gc_g = gs.get("glucocorticoid_core", [])
    genes = list(dict.fromkeys(isg_g + core + gc_g + vc.C1Q + vc.IFNG_UP))
    h5 = A.h5ad or os.path.join(root, "data", "std", f"{A.tag}.h5ad")
    want = dict(label=(["label"], []), role=(["role"], []), batch=(["batch"], []),
                cond=(["condition"], [r"disease"]), age=(["age_group"], [r"age_group", r"^age"]))
    O, present, E, found = vc.load_expression(h5, genes, A.chunk, want)
    R["obs_columns"] = found
    R["genes_missing"] = sorted(set(genes) - set(present))
    isref = (O.role.values == "ref")
    isq = (O.role.values == "query")
    # reference labels: y_ref of the emb full npz (same order as the reference rows), else obs label
    lab = O.label.values.astype(object).copy()
    ef = os.path.join(root, A.emb_dir, f"{A.tag}_seed{CL['seeds'][0]}_full.npz")
    if os.path.exists(ef):
        yref = np.asarray(np.load(ef, allow_pickle=True)["y_ref"]).astype(str)
        if len(yref) == isref.sum():
            agree = float(np.mean(yref == lab[isref].astype(str)))
            lab[isref] = yref
            log(f"reference labels from {ef} (agreement with obs label {agree:.3f})")
            R["yref_obs_agreement"] = agree
        else:
            log(f"WARNING: y_ref length {len(yref)} != reference rows {isref.sum()}; using obs label")
    else:
        log(f"WARNING: {ef} missing; reference labels from obs")
    # query rows <-> cells npz by name
    qn = pd.Index(O.index[isq])
    names = CL["names"]
    if qn.isin(names).mean() > 0.99:
        pos = pd.Series(np.arange(len(names)), index=names).reindex(qn).values
    elif len(qn) == len(names):
        log("WARNING: query obs names do not match npz names; aligning by position")
        pos = np.arange(len(names), dtype=float)
    else:
        raise ValueError("cannot align query cells with the cells npz")
    R["query_cells_unmatched"] = int(np.isnan(pos).sum())
    qi = np.where(isq)[0]
    ok = np.isfinite(pos)
    pos_i = pos[ok].astype(int)
    keep = isref.copy()
    keep[qi[ok]] = True
    lab[qi[ok]] = CL["label_arcane"][pos_i]
    flags = {}
    for c, F in CL["flags"].items():
        arr = np.full((len(O), F.shape[1]), np.nan)
        arr[qi[ok]] = F.values[pos_i]
        flags[c] = arr
    lab = lab.astype(str)
    Z, nref = vc.z_by_label(E, lab, isref, A.sd_floor, A.zclip, A.min_ref_cells)
    del E
    R["ref_cells_per_label"] = nref
    gi = {g: i for i, g in enumerate(present)}
    cell = pd.DataFrame(index=O.index)
    cell["lab"], cell["isref"], cell["isq"] = lab, isref, isq & keep
    cell["donor"] = O.batch.values
    cell["sle"] = [1 if vc.is_sle(x) else (0 if str(x).lower() in ("normal", "healthy", "control") else -1)
                   for x in O.cond.values]
    cell["age"] = O.age.values
    cell["isg"], R["isg_genes"], _ = vc.gene_score(Z, gi, isg_g)
    gcs, R["gc_genes"], _ = vc.gene_score(Z, gi, gc_g)
    cell["gc"] = gcs if gcs is not None else np.nan
    specs = vc.state_specs(gs, sorted(set(lab[isref])))
    for st in specs:
        s, st["up_used"], st["down_used"] = vc.gene_score(Z, gi, st["up"], st["down"])
        cell[f"s_{st['name']}"] = s if s is not None else np.nan
        rest = [g for g in isg_g if g not in set(st["up"]) | set(st["down"])]
        si, _, _ = vc.gene_score(Z, gi, rest)
        cell[f"isg_{st['name']}"] = si if si is not None else np.nan
        st["isg_genes_adj"] = [g for g in rest if g in gi]
        log(f"state {st['name']}: labels {st['labels']} up {st['up_used']} down {st['down_used']}")
    del Z
    return cell, flags, specs


def donor_meta(cell):
    q = cell[cell.isq]
    D = q.groupby("donor").agg(sle=("sle", "first"), age=("age", "first"), n_cells=("lab", "size"))
    dcsv = A.donors_csv or os.path.join(A.root, "data", "std", f"{A.tag}_donors.csv")
    if (D.age.isin(["na", "nan", ""]).all()) and os.path.exists(dcsv):
        dc = pd.read_csv(dcsv, index_col=0)
        c = vc.find_col(dc.columns, ["age", "age_group"], [r"age"])
        if c:
            D["age"] = dc[c].reindex(D.index).astype(str).values
    D["age"] = D.age.astype(str).replace({"nan": "na", "": "na"})
    return D


# ------------------------------------------------------------------------------------------------
def test1(cell, flags, D, w):
    q = cell.isq.values
    T = {}
    for c in T1_COLS + ["p_unknown_U_bonf"]:
        if c not in flags:
            continue
        cons = vc.consensus(flags[c])
        cell[f"f_{c}"] = cons
        sr = vc.std_rate(cons[q], cell.lab.values[q], cell.donor.values[q], w)
        D[f"flag_{c}_std"] = sr["std"].reindex(D.index)
        D[f"flag_{c}_raw"] = sr["raw"].reindex(D.index)
        if c not in T1_COLS:
            continue
        U = D[D.use]
        g = U.sle.values
        res = vc.group_compare(U[f"flag_{c}_std"].values, g, U.age.values, A.nboot, A.nperm)
        h = cell[q & (cell.donor.isin(U.index[g == 0]))]
        res["fa_healthy_cells"] = float(np.nanmean(h[f"f_{c}"]))
        res["fa_healthy_donor_mean_raw"] = float(U.loc[g == 0, f"flag_{c}_raw"].mean())
        res["fa_healthy_donor_mean_std"] = float(U.loc[g == 0, f"flag_{c}_std"].mean())
        res["fa_healthy_donor_q90"] = float(U.loc[g == 0, f"flag_{c}_raw"].quantile(0.9))
        res["flag_sle_cells"] = float(np.nanmean(cell[q & cell.donor.isin(U.index[g == 1])][f"f_{c}"]))
        res["raw_diff"] = float(U.loc[g == 1, f"flag_{c}_raw"].mean() - U.loc[g == 0, f"flag_{c}_raw"].mean())
        per_seed = {}
        for j in range(flags[c].shape[1]):
            sr_j = vc.std_rate(flags[c][q, j], cell.lab.values[q], cell.donor.values[q], w)["std"].reindex(U.index)
            per_seed[str(R["seeds_used"][j])] = dict(
                diff=float(sr_j[g == 1].mean() - sr_j[g == 0].mean()), fa_healthy=float(sr_j[g == 0].mean()))
        res["per_seed"] = per_seed
        T[c] = res
        log(f"T1 {c}: FA healthy {fmt(res['fa_healthy_donor_mean_raw'])}, SLE-healthy std diff "
            f"{eci(res.get('diff'), res.get('diff_lo'), res.get('diff_hi'))}, p_perm(age) {fmt(res.get('p_perm'), 4)}")
    R["T1"] = T


def test2(cell, D, w):
    q = cell.isq.values
    sr = vc.std_rate(cell.isg.values[q], cell.lab.values[q], cell.donor.values[q], w)
    D["isg"] = sr["std"].reindex(D.index)
    U = D[D.use].copy()
    T = {}
    T["isg_sle_vs_healthy"] = vc.group_compare(U.isg.values, U.sle.values, U.age.values, A.nboot, A.nperm)
    for c in T1_COLS:
        y = f"flag_{c}_std"
        if y not in U:
            continue
        t = dict(spearman_sle=vc.spearman_ci(U.loc[U.sle == 1, y], U.loc[U.sle == 1, "isg"], A.nboot),
                 spearman_healthy=vc.spearman_ci(U.loc[U.sle == 0, y], U.loc[U.sle == 0, "isg"], A.nboot),
                 ols_unadj=vc.ols_effect(U, y, "sle", [], ["age"], A.nboot, strat_col="age"),
                 ols_isg=vc.ols_effect(U, y, "sle", ["isg"], ["age"], A.nboot, strat_col="age"),
                 lrt_flag_given_isg=vc.logistic_lrt(U, "sle", ["isg"], y),
                 lrt_flag_given_isg_age=vc.logistic_lrt(U, "sle", ["isg"], y, ["age"]))
        T[c] = t
    R["T2"] = T
    m = T.get(MAIN, {})
    log(f"T2 rho(flag,ISG|SLE) {m.get('spearman_sle', {}).get('rho')}, ISG-adj SLE {m.get('ols_isg', {}).get('coef')}")


def state_donor(cell, st, pos, D, isgcol):
    q = cell.isq.values & np.isin(cell.lab.values, st["labels"]) & np.isfinite(pos)
    d = pd.DataFrame(dict(donor=cell.donor.values[q], pos=pos[q], isg=cell[isgcol].values[q],
                          lab=cell.lab.values[q]))
    g = d.groupby("donor")
    S = pd.DataFrame(dict(k=g.pos.sum(), n=g.size())).reindex(D.index)
    S["frac"] = S.k / S.n
    S.loc[~(S.n >= A.min_cells), ["frac"]] = np.nan
    S["logit"] = np.where(S.frac.notna(), vc.logit_frac(S.k.fillna(0), S.n.fillna(1)), np.nan)
    return S


def state_tests(cell, D, st, pos, tname, sign, sub=None, extra_nums=()):
    """SLE vs healthy and ISG-adjusted effect of the state fraction. sub: donor mask (index) to restrict to."""
    name = st["name"]
    S = state_donor(cell, st, pos, D, f"isg_{name}")
    Dd = D.copy()
    Dd["frac"], Dd["logit"], Dd["k"], Dd["n"] = S.frac, S.logit, S.k, S.n
    # donor ISG for adjustment, state genes removed (composition-standardized like the main ISG score)
    U = Dd[Dd.use & Dd.frac.notna()]
    if sub is not None:
        U = U[U.index.isin(sub)]
    g = U.sle.values
    res = dict(state=name, threshold=tname, n_sle=int(g.sum()), n_healthy=int((1 - g).sum()),
               frac_sle_mean=float(U.frac[g == 1].mean()) if g.sum() else np.nan,
               frac_healthy_mean=float(U.frac[g == 0].mean()) if (1 - g).sum() else np.nan,
               frac_healthy_median=float(U.frac[g == 0].median()) if (1 - g).sum() else np.nan,
               frac_sle_median=float(U.frac[g == 1].median()) if g.sum() else np.nan)
    if g.sum() >= 2 and (1 - g).sum() >= 2:
        res["hl"], res["hl_lo"], res["hl_hi"] = vc.hl_boot(U.frac[g == 1], U.frac[g == 0], A.nboot)
        pr = vc.strat_perm(U.logit.values, g, U.age.values, A.nperm)
        res["perm_logit_stat"], res["p_perm"] = pr["stat"], pr["p"]
        res["p_one_sided"] = pr["p_greater"] if sign > 0 else pr["p_less"]
        o = vc.ols_effect(U, "logit", "sle", [f"isg_{name}"] + list(extra_nums), ["age"], A.nboot, strat_col="age")
        res.update({f"adj_{k}": v for k, v in o.items() if not isinstance(v, (list, dict))})
        res["adj_covariates"] = o.get("covariates")
        if "p_greater" in o:
            res["adj_p_one_sided"] = o["p_greater"] if sign > 0 else o["p_less"]
        res["replicated"] = bool(np.sign(res["hl"]) == sign and res["p_one_sided"] < 0.05)
        res["replicated_isg_adj"] = bool("coef" in o and np.sign(o["coef"]) == sign
                                         and res.get("adj_p_one_sided", 1) < 0.05)
        res["spearman_isg_sle"] = vc.spearman_ci(U.frac[g == 1], U[f"isg_{name}"][g == 1], A.nboot)
    return res, Dd


def test3(cell, D, specs, w, signs, sub=None, extra_nums=(), tag=""):
    q = cell.isq.values
    out = []
    flag = cell[f"f_{MAIN}"].values if f"f_{MAIN}" in cell else np.full(len(cell), np.nan)
    healthy = (cell.sle.values == 0)
    use_d = cell.donor.isin(D.index[D.use]).values
    for st in specs:
        name = st["name"]
        if not st["labels"] or cell[f"s_{name}"].isna().all():
            log(f"T3 {name}: skipped (labels {st['labels']})")
            continue
        sr = vc.std_rate(cell[f"isg_{name}"].values[q], cell.lab.values[q], cell.donor.values[q], w)
        D[f"isg_{name}"] = sr["std"].reindex(D.index)
        sc = cell[f"s_{name}"].values
        lab = cell.lab.values
        pa, thr_a = vc.threshold_ref(sc, lab, cell.isref.values, st["labels"], A.q)
        pb, thr_b = vc.threshold_healthy(sc, lab, q & use_d, healthy, cell.donor.values, st["labels"], A.q)
        inlab = q & use_d & np.isin(lab, st["labels"])
        isgdet = vc.isg_topn(cell[f"isg_{name}"].values, flag, lab, inlab, st["labels"])
        for tname, pos, thr in [("ref_q99", pa, thr_a), ("healthy_q99", pb, thr_b)]:
            res, Dd = state_tests(cell, D, st, pos, tname, signs.get(name, 1), sub, extra_nums)
            res["thresholds"] = thr
            res["labels"] = st["labels"]
            if not tag:
                D[f"{name}_{tname}_frac"], D[f"{name}_{tname}_n"] = Dd.frac, Dd.n
                for grp, gv in [("sle", 1), ("healthy", 0)]:
                    m = inlab & (cell.sle.values == gv) & (pos == 1)
                    res[f"npos_{grp}"] = int(m.sum())
                    res[f"pos_flag_share_{grp}"] = float(np.nanmean(flag[m])) if m.any() else np.nan
                    res[f"pos_isg_share_{grp}"] = float(np.nanmean(isgdet[m])) if m.any() else np.nan
                    mm = inlab & (cell.sle.values == gv)
                    res[f"flag_rate_labels_{grp}"] = float(np.nanmean(flag[mm])) if mm.any() else np.nan
                    for det, arr in [("flag", flag), ("isg", isgdet)]:
                        mh = vc.mh_by_donor(pos, arr, cell.donor.values, mm)
                        res[f"MH_{det}_{grp}"], res[f"MH_{det}_{grp}_lo"], res[f"MH_{det}_{grp}_hi"] = (
                            mh["OR"], mh["lo"], mh["hi"])
                hq = inlab & healthy & np.isfinite(pos)
                res["healthy_pos_rate_cells"] = float(np.nanmean(pos[hq])) if hq.any() else np.nan
            out.append(res)
            log(f"T3{tag} {name} {tname}: HL {eci(res.get('hl'), res.get('hl_lo'), res.get('hl_hi'), 4)} "
                f"p1 {fmt(res.get('p_one_sided'), 4)} adj {fmt(res.get('adj_coef'))} p1 {fmt(res.get('adj_p_one_sided'), 4)}"
                f" replicated {res.get('replicated')} / {res.get('replicated_isg_adj')}")
    return out


def test4(cell, flags, D, w):
    q = cell.isq.values
    T = {}
    if "p_unknown_ctrl_U" not in flags or MAIN not in flags:
        R["T4"] = dict(error="p_unknown_ctrl_U or p_unknown_ctrl missing")
        return
    FU, FB = flags["p_unknown_ctrl_U"], flags[MAIN]
    uonly = np.where(np.isfinite(FU) & np.isfinite(FB), ((FU == 1) & (FB == 0)).astype(float), np.nan)
    cell["u_only"] = vc.consensus(uonly)
    cell["u_dev"] = cell[f"f_p_unknown_ctrl_U"]
    for col in ["u_only", "u_dev"] + (["f_p_unknown_U_bonf"] if "f_p_unknown_U_bonf" in cell else []):
        sr = vc.std_rate(cell[col].values[q], cell.lab.values[q], cell.donor.values[q], w)
        D[f"{col}_raw"], D[f"{col}_std"] = sr["raw"].reindex(D.index), sr["std"].reindex(D.index)
        U = D[D.use]
        T[col] = dict(raw=vc.auroc_test(U[f"{col}_raw"].values, U.sle.values, A.nboot),
                      std=vc.auroc_test(U[f"{col}_std"].values, U.sle.values, A.nboot),
                      compare_std=vc.group_compare(U[f"{col}_std"].values, U.sle.values, U.age.values, A.nboot,
                                                   A.nperm),
                      healthy_mean_raw=float(U.loc[U.sle == 0, f"{col}_raw"].mean()),
                      sle_mean_raw=float(U.loc[U.sle == 1, f"{col}_raw"].mean()))
    U = D[D.use]
    T["main_flag_auroc"] = vc.auroc_test(U[f"flag_{MAIN}_std"].values, U.sle.values, A.nboot)
    R["T4"] = T


def test5(cell, D, specs, w, signs):
    q = cell.isq.values
    D["gc"] = pd.Series(cell.gc.values[q]).groupby(cell.donor.values[q]).mean().reindex(D.index)
    U = D[D.use]
    T = dict(gc_sle_vs_healthy=vc.group_compare(U.gc.values, U.sle.values, U.age.values, A.nboot, A.nperm),
             spearman_gc_flag_sle=vc.spearman_ci(U.loc[U.sle == 1, "gc"], U.loc[U.sle == 1, f"flag_{MAIN}_std"],
                                                 A.nboot),
             spearman_gc_isg_sle=vc.spearman_ci(U.loc[U.sle == 1, "gc"], U.loc[U.sle == 1, "isg"], A.nboot))
    thr = U.loc[U.sle == 1, "gc"].quantile(0.75)
    excl = U.index[(U.sle == 1) & (U.gc >= thr)]
    keep = U.index.difference(excl)
    T["gc_q75_sle"], T["n_excluded"] = float(thr), int(len(excl))
    T["excluded_donors"] = list(map(str, excl))
    Uk = U.loc[keep]
    y = f"flag_{MAIN}_std"
    T["T1_excl"] = vc.group_compare(Uk[y].values, Uk.sle.values, Uk.age.values, A.nboot, A.nperm)
    T["T1_adj_gc"] = vc.ols_effect(U, y, "sle", ["gc"], ["age"], A.nboot, strat_col="age")
    T["T1_adj_gc_isg"] = vc.ols_effect(U, y, "sle", ["gc", "isg"], ["age"], A.nboot, strat_col="age")
    T["T3_excl"] = test3(cell, D, specs, w, signs, sub=keep, tag=" excl-GC")
    T["T3_adj_gc"] = test3(cell, D, specs, w, signs, extra_nums=("gc",), tag=" adj-GC")
    R["T5"] = T


# ------------------------------------------------------------------------------------------------
def key_lines():
    K = []
    T1 = R.get("T1", {})
    for c, r in T1.items():
        bs = r.get("by_stratum", {})
        st = " ".join(f"{s}:{fmt(v['diff'])}[{fmt(v['lo'])},{fmt(v['hi'])}]p={fmt(v['p'], 4)}" for s, v in bs.items())
        K.append(f"KEY T1 {c}: FA_healthy={fmt(r.get('fa_healthy_donor_mean_raw'))} (cells {fmt(r.get('fa_healthy_cells'))})"
                 f" SLE_std={fmt(r.get('mean_sle'))} diff={eci(r.get('diff'), r.get('diff_lo'), r.get('diff_hi'))}"
                 f" agestrat={eci(r.get('strat_diff'), r.get('strat_lo'), r.get('strat_hi'))} p_perm={fmt(r.get('p_perm'), 4)}"
                 f" | {st} | n={r.get('n_sle')}/{r.get('n_healthy')}")
    t2 = R.get("T2", {}).get(MAIN, {})
    if t2:
        s, o, l = t2["spearman_sle"], t2["ols_isg"], t2["lrt_flag_given_isg"]
        u = t2["ols_unadj"]
        K.append(f"KEY T2 rho(flag,ISG|SLE)={eci(s['rho'], s['lo'], s['hi'])} n={s['n']} | SLE effect on flag: "
                 f"unadj={eci(u.get('coef'), u.get('lo'), u.get('hi'))} ISG-adj={eci(o.get('coef'), o.get('lo'), o.get('hi'))}"
                 f" p={fmt(o.get('p'), 4)} boot[{fmt(o.get('boot_lo'))},{fmt(o.get('boot_hi'))}] | LRT flag|ISG "
                 f"chi2={fmt(l.get('chi2'), 2)} p={fmt(l.get('p'), 4)}{' (separation)' if l.get('separation_warning') else ''}"
                 f" (age-adj p={fmt(t2['lrt_flag_given_isg_age'].get('p'), 4)})")
    for r in R.get("T3", []):
        K.append(f"KEY T3 {r['state']} {r['threshold']}: HL={eci(r.get('hl'), r.get('hl_lo'), r.get('hl_hi'), 4)} "
                 f"p1={fmt(r.get('p_one_sided'), 4)} ISGadj={eci(r.get('adj_coef'), r.get('adj_lo'), r.get('adj_hi'), 2)} "
                 f"p1={fmt(r.get('adj_p_one_sided'), 4)} REPL={'Y' if r.get('replicated') else 'N'}/"
                 f"{'Y' if r.get('replicated_isg_adj') else 'N'} | state+ flagged ARCANE={fmt(r.get('pos_flag_share_sle'), 2)}"
                 f" ISGtopn={fmt(r.get('pos_isg_share_sle'), 2)} MH_OR={fmt(r.get('MH_flag_sle'), 2)}"
                 f" healthy_pos={fmt(r.get('healthy_pos_rate_cells'), 4)}")
    t4 = R.get("T4", {})
    if "u_only" in t4:
        K.append(f"KEY T4 AUROC SLE vs healthy: U-only={eci(t4['u_only']['std'].get('auroc'), t4['u_only']['std'].get('lo'), t4['u_only']['std'].get('hi'), 2)}"
                 f" U-dev={eci(t4['u_dev']['std'].get('auroc'), t4['u_dev']['std'].get('lo'), t4['u_dev']['std'].get('hi'), 2)}"
                 f" both-view={eci(t4['main_flag_auroc'].get('auroc'), t4['main_flag_auroc'].get('lo'), t4['main_flag_auroc'].get('hi'), 2)}"
                 f" | U-only mean healthy/SLE={fmt(t4['u_only']['healthy_mean_raw'])}/{fmt(t4['u_only']['sle_mean_raw'])}")
    t5 = R.get("T5", {})
    if t5:
        e, a = t5["T1_excl"], t5["T1_adj_gc"]
        K.append(f"KEY T5 GC SLE-healthy={fmt(t5['gc_sle_vs_healthy'].get('diff'))} p={fmt(t5['gc_sle_vs_healthy'].get('p_perm'), 4)}"
                 f" | excl top-GC SLE (n={t5['n_excluded']}): flag diff={eci(e.get('diff'), e.get('diff_lo'), e.get('diff_hi'))}"
                 f" p={fmt(e.get('p_perm'), 4)} | GC-adj={eci(a.get('coef'), a.get('lo'), a.get('hi'))} p={fmt(a.get('p'), 4)}")
        for lbl, lst in [("excl", t5["T3_excl"]), ("adjGC", t5["T3_adj_gc"])]:
            K.append(f"KEY T5 states {lbl}: " + " ; ".join(
                f"{r['state']}/{r['threshold']} HL={fmt(r.get('hl'), 4)} p1={fmt(r.get('p_one_sided'), 4)} "
                f"adj={fmt(r.get('adj_coef'), 2)} p1={fmt(r.get('adj_p_one_sided'), 4)}" for r in lst))
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
    R["states"] = [{k: v for k, v in s.items()} for s in specs]
    D = donor_meta(cell)
    D["use"] = (D.n_cells >= A.min_donor_cells) & D.sle.isin([0, 1])
    R["n_donors"] = dict(total=int(len(D)), used=int(D.use.sum()),
                         by_group=D[D.use].groupby(["sle", "age"]).size().rename(lambda x: str(x)).to_dict())
    log("donors used:", R["n_donors"])
    q = cell.isq.values
    h = q & cell.donor.isin(D.index[D.use & (D.sle == 0)]).values
    w = vc.healthy_weights(cell.lab.values, cell.donor.values, h)
    R["healthy_composition"] = w.to_dict()
    guarded("T1", test1, cell, flags, D, w)
    save()
    guarded("T2", test2, cell, D, w)
    save()
    R["T3"] = guarded("T3", test3, cell, D, specs, w, signs) or []
    save()
    guarded("T4", test4, cell, flags, D, w)
    save()
    guarded("T5", test5, cell, D, specs, w, signs)
    D.to_csv(os.path.join(A.out, "donor_table.csv"))
    rows = R.get("T3", []) + [dict(r, variant="excl_topGC") for r in R.get("T5", {}).get("T3_excl", [])] + \
        [dict(r, variant="adj_GC") for r in R.get("T5", {}).get("T3_adj_gc", [])]
    if rows:
        pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]).to_csv(
            os.path.join(A.out, "state_tests.csv"), index=False)
    K = key_lines()
    R["KEY"] = K
    save()
    print("\n".join(K), flush=True)
    log("NB_VALIDATE_DONE", A.out)


if __name__ == "__main__":
    main()
