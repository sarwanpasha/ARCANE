"""Evaluate ARCANE (symmetric cross-fitting) and baselines for one run.

Usage: python evaluate2.py <tag> <seed> [k]
Reads emb/<tag>_seed<s>_{full,cf0,cf1}.npz (cf targets = held-out fold cells followed by query cells)
and, if present, emb/<tag>_seed<s>_celltypist.npz. Writes res2/<tag>_seed<s>[_k<k>].json, _types.csv,
_cells.npz.
"""
import json
import os
import sys
import time
import zlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from arcane import knn_evidence, hard, run_arcane, run_arcane_sym, _fold_p
from metrics import per_type, auroc

tag, seed = sys.argv[1], int(sys.argv[2])
K = int(sys.argv[3]) if len(sys.argv) > 3 else 15
ALPHA, ALPHA_U = 0.1, 0.05
OUT = "res2"
os.makedirs(OUT, exist_ok=True)
suffix = "" if K == 15 else f"_k{K}"
t0 = time.time()
L = lambda s: dict(np.load(f"emb/{tag}_seed{seed}_{s}.npz", allow_pickle=True))
import glob
FOLDS = sorted(os.path.basename(p).rsplit("_", 1)[1][:-4] for p in glob.glob(f"emb/{tag}_seed{seed}_cf*.npz")
               if not p.endswith("_dropped.txt"))
F, folds = L("full"), [L(s) for s in FOLDS]

# partial labels: only labelled reference cells serve as neighbours and calibration cells
partial = None
if "_partial-" in tag:
    partial = float(tag.split("_partial-")[1].split("_")[0])


def restrict(d):
    if partial is None or "labeled_ref" not in d:
        return d
    m = d["labeled_ref"].astype(bool)
    d = dict(d)
    for key in ("ref_U", "ref_S", "y_ref", "parent_ref", "b_ref"):
        if key in d:
            d[key] = d[key][m]
    return d


F = restrict(F)
folds = [restrict(f) for f in folds]
lab_mask = None
if partial is not None:
    lab_mask = lambda names: np.array([zlib.crc32(str(n).encode()) % 1000 < partial * 1000 for n in names])

# with partial labels the labelled subsets of the full and fold models differ; use the union of their classes
classes = np.array(sorted(set(F["y_ref"]).union(*[set(f["y_ref"]) for f in folds])))
T = len(classes)
ci = {c: i for i, c in enumerate(classes)}
y_tgt = F["y_tgt"]
known = np.isin(y_tgt, classes)
has_labels = bool(known.any())
y = np.array([ci.get(c, -1) for c in y_tgt])

res = run_arcane_sym(F, folds, classes, K, ALPHA, ALPHA_U, labeled_mask=lab_mask)
res_acc = run_arcane_sym(F, folds, classes, K, ALPHA, ALPHA_U, objective="accuracy", labeled_mask=lab_mask)
# previous (asymmetric) calibration for comparison: held-out fold only, full model for query
cal_old = []
for f in folds:
    m = f["tgt_role"] == "cal"
    cal_old.append(dict(ref_U=f["ref_U"], ref_S=f["ref_S"], y_ref=f["y_ref"], tgt_U=f["tgt_U"][m],
                        tgt_S=f["tgt_S"][m], y_tgt=f["y_tgt"][m]))
res_old = run_arcane(F["ref_U"], F["ref_S"], F["y_ref"], F["tgt_U"], F["tgt_S"], cal_old, classes, K, ALPHA)

pred = {"kNN-U": res["yU"], "kNN-S": res["yS"], "Average": res["yavg"], "ARCANE": res["label"],
        "ARCANE-acc": res_acc["label"]}
pc = list(F["scanvi_classes"])
prob = F["scanvi_prob"]
pred["scANVI-clf"] = np.array([ci.get(c, -1) for c in np.array(pc)[prob.argmax(1)]])
# Saerens et al. EM label-shift correction of the scANVI posterior
prior_ref = pd.Series(F["y_ref"]).value_counts(normalize=True).reindex(pc).fillna(1e-6).values
w = prior_ref.copy()
for _ in range(100):
    post = prob * (w / prior_ref)
    post /= post.sum(1, keepdims=True)
    w_new = post.mean(0)
    if np.abs(w_new - w).max() < 1e-6:
        break
    w = w_new
pred["scANVI-EM"] = np.array([ci.get(c, -1) for c in np.array(pc)[post.argmax(1)]])
sU, sS = StandardScaler().fit(F["ref_U"]), StandardScaler().fit(F["ref_S"])
Zr = np.hstack([sU.transform(F["ref_U"]), sS.transform(F["ref_S"])])
Zt = np.hstack([sU.transform(F["tgt_U"]), sS.transform(F["tgt_S"])])
pred["Concat"] = hard(knn_evidence(Zr, F["y_ref"], Zt, classes, K)[0])
ct = f"emb/{tag}_seed{seed}_celltypist.npz"
if os.path.exists(ct):
    C = np.load(ct, allow_pickle=True)
    m = pd.Series(C["pred"], index=C["tgt_names"]).reindex(F["tgt_names"]).values
    pred["CellTypist"] = np.array([ci.get(c, -1) for c in m])

# stacked meta-classifier: logistic regression on [p_U, p_S, scaled kNN distances], fitted on the
# cross-fitted calibration cells (same information as ARCANE)
from sklearn.linear_model import LogisticRegression
Xc, yc_all = res["cal_feat"], res["cal"][2]
rs_ = np.random.default_rng(seed)
sub_ = rs_.choice(len(yc_all), min(150000, len(yc_all)), replace=False)
present = np.unique(yc_all[sub_])
lr = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0).fit(Xc[sub_], yc_all[sub_])
pred["Stacking"] = lr.predict(res["q_feat"])


# ARCANE-clf: the scANVI classifier posterior as the supervised evidence (useful with partial labels)
def clf_prob(d):
    pcl = list(d["scanvi_classes"])
    M = np.zeros((d["scanvi_prob"].shape[0], T), np.float32)
    for j, c in enumerate(pcl):
        if c in ci:
            M[:, ci[c]] = d["scanvi_prob"][:, j]
    M /= np.maximum(M.sum(1, keepdims=True), 1e-9)
    return M


Fc = dict(F); Fc["PS_override"] = clf_prob(F)
fc = []
for f in folds:
    f2 = dict(f); f2["PS_override"] = clf_prob(f); fc.append(f2)
res_clf = run_arcane_sym(Fc, fc, classes, K, ALPHA, ALPHA_U, labeled_mask=lab_mask)
pred["ARCANE-clf"] = res_clf["label"]
n_ref = pd.Series(F["y_ref"]).value_counts().reindex(classes).fillna(0).values
n_q = np.array([(y == t).sum() for t in range(T)])
ev = np.where(n_q >= 10)[0]
pi = n_q[ev] / n_q[ev].sum()
order = np.argsort(pi)
pi_rev = np.empty_like(pi)
pi_rev[order] = pi[order[::-1]]          # reversed abundance ranking: heavy label shift
rows, summ = [], {"tag": tag, "seed": seed, "k": K, "n_classes": int(T), "n_query": int(len(y)),
                  "n_known": int(known.sum()), "methods": {}}
for mname, yh in (pred.items() if has_labels else []):
    rec, prec, f1, _ = per_type(y[known], yh[known], T)
    summ["methods"][mname] = dict(BA=float(np.nanmean(rec[ev])), macroF1=float(np.nanmean(f1[ev])),
                                  acc=float((yh[known] == y[known]).mean()),
                                  acc_reversed=float(np.nansum(pi_rev * rec[ev])))
    for t in ev:
        rows.append(dict(tag=tag, seed=seed, type=classes[t], n_query=int(n_q[t]), n_ref=int(n_ref[t]),
                         method=mname, recall=rec[t], precision=prec[t], f1=f1[t]))

S = res["sets"]
SB = res["sets_bonf"]


def conf_summary(sets, unknown, unknown_type, known=known):
    if not known.any():
        return dict(unknown_all=float(unknown.mean()), set_size=float(sets.sum(1).mean()))
    cov = [float(sets[known][y[known] == t, t].mean()) for t in ev if (y[known] == t).sum() >= 20]
    return dict(mean_cov=float(np.mean(cov)), min_cov=float(np.min(cov)), frac_types_cov085=float(np.mean(np.array(cov) >= 0.85)),
                per_type_cov=dict(zip([c for t, c in zip(ev, classes[ev]) if (y[known] == t).sum() >= 20], cov)),
                set_size=float(sets[known].sum(1).mean()), singleton=float((sets[known].sum(1) == 1).mean()),
                empty=float((sets[known].sum(1) == 0).mean()),
                unknown_known=float(unknown[known].mean()),
                unknown_novel=float(unknown[~known].mean()) if (~known).any() else None,
                unknown_type_known=float(unknown_type[known].mean()) if unknown_type is not None else None,
                unknown_type_novel=float(unknown_type[~known].mean()) if (unknown_type is not None and (~known).any()) else None)


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def bh_summary(p, mask, level=0.1):
    """BH discoveries among cells in mask; FDR and power when the novel cells are known."""
    q = bh(p[mask])
    disc = q <= level
    out = dict(level=level, n_tested=int(mask.sum()), n_disc=int(disc.sum()), frac_disc=float(disc.mean()))
    if has_labels and (~known[mask]).any():
        nv = ~known[mask]
        out.update(fdr=float((disc & ~nv).sum() / max(disc.sum(), 1)), power=float((disc & nv).sum() / nv.sum()))
    return out


if has_labels:
    summ["conformal"] = conf_summary(S, res["unknown"], res["unknown_type"])
    summ["conformal_old"] = conf_summary(res_old["sets"], res_old["unknown"], None)
    summ["conformal_bonf"] = conf_summary(SB, res["unknown_bonf"], res["unknown_type_bonf"])
    summ["conformal_rand"] = conf_summary(res["sets_rand"], res["unknown_bonf"], res["unknown_type_bonf"])
    summ["conformal_bonf_U"] = conf_summary(SB, res["unknown_U_bonf"], None)
    summ["conformal_clf"] = conf_summary(res_clf["sets_bonf"], res_clf["unknown_bonf"], None)
    # how often a point label falls outside the prediction set (the full-model labels are not covered by
    # the guarantee; the set is computed from the fold models)
    kk = known
    summ["label_outside_set"] = {m: float((~SB[kk][np.arange(kk.sum()), np.clip(pred[m][kk], 0, T - 1)]).mean())
                                 for m in ("ARCANE", "kNN-U", "kNN-S", "scANVI-clf")}
    summ["label_outside_set_wrong"] = {m: float((~SB[kk][np.arange(kk.sum()), np.clip(pred[m][kk], 0, T - 1)])[pred[m][kk] != y[kk]].mean())
                                       if (pred[m][kk] != y[kk]).any() else 0.0 for m in ("ARCANE", "kNN-U", "kNN-S")}
    summ["label_outside_set_right"] = {m: float((~SB[kk][np.arange(kk.sum()), np.clip(pred[m][kk], 0, T - 1)])[pred[m][kk] == y[kk]].mean())
                                       for m in ("ARCANE", "kNN-U", "kNN-S")}
summ["bh"] = {"ARCANE": bh_summary(res["p_unknown_bonf"], np.ones(len(y), bool)),
              "ARCANE U only": bh_summary(res["p_unknown_U_bonf"], np.ones(len(y), bool))}
# transfer diagnostic: under exchangeability the reference-calibrated flag rate on query cells of known
# types is about alpha_U; a much larger rate on the query (or on its controls) warns that guarantees may fail
cond = np.array([str(c) for c in F["cond_tgt"]]) if "cond_tgt" in F else None
donor = np.array([str(b) for b in F["b_tgt"]]) if "b_tgt" in F else None
has_ctrl = cond is not None and (cond == "normal").any() and (cond != "normal").any()
diag_mask = (cond == "normal") if has_ctrl else np.ones(len(y), bool)
rate = float(res["unknown_bonf"][diag_mask].mean())
summ["transfer_diagnostic"] = dict(flag_rate=rate, threshold=2 * ALPHA_U, warning=bool(rate > 2 * ALPHA_U),
                                   set_size=float(SB[diag_mask].sum(1).mean()),
                                   empty_sets=float((SB[diag_mask].sum(1) == 0).mean()))
# symmetry check: fold-model scores of the held-out reference cells against those of query cells that
# should be exchangeable with them (healthy query donors); donor-level permutation test of the medians
if has_ctrl:
    sym = {}
    for nm, cl, ql in (("nu", res["fold_cal_nu"], res["fold_q_nu"]), ("nu_U", res["fold_cal_nu_u"], res["fold_q_nu_u"])):
        stats = []
        for f_i, (cn, qn, cb) in enumerate(zip(cl, ql, res["fold_cal_batch"])):
            md_c = pd.Series(cn).groupby(cb).median()
            md_q = pd.Series(qn[cond == "normal"]).groupby(donor[cond == "normal"]).median()
            allm = np.concatenate([md_c.values, md_q.values])
            obs = md_q.mean() - md_c.mean()
            rp = np.random.default_rng(f_i)
            null = np.array([(lambda z: z[len(md_c):].mean() - z[:len(md_c)].mean())(rp.permutation(allm)) for _ in range(5000)])
            from scipy.stats import ks_2samp
            stats.append(dict(n_cal_donors=int(len(md_c)), n_ctrl_donors=int(len(md_q)), diff=float(obs),
                              perm_p=float((1 + (np.abs(null) >= abs(obs)).sum()) / 5001),
                              ks=float(ks_2samp(cn, qn[cond == "normal"]).statistic),
                              q95_cal=float(np.quantile(cn, 0.95)), q95_ctrl=float(np.quantile(qn[cond == "normal"], 0.95)),
                              fa_ctrl_fold=float((_fold_p(cn, qn[cond == "normal"]) <= ALPHA_U).mean())))
        sym[nm] = stats
    summ["symmetry"] = sym


def control_recal(nov_full, hierarchical=False):
    """Unknown flag calibrated on control donors of the query study, leave one donor out. Each control donor
    contributes the same number of cells; the hierarchical variant weights every calibration donor equally
    using all of its cells (p = (1 + sum_d mean_d[s_d >= s]) / (D + 1))."""
    ctrl = cond == "normal"
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


p_ctrl = p_ctrl_U = p_ctrl_h = None
if has_ctrl:
    ctrl = cond == "normal"
    p_ctrl, per, dons = control_recal(res["novelty"])
    p_ctrl_U, _, _ = control_recal(res["novelty_U"])
    p_ctrl_h, _, _ = control_recal(res["novelty"], hierarchical=True)
    u_ctrl = p_ctrl <= ALPHA_U
    don_fa = pd.Series(u_ctrl[ctrl]).groupby(donor[ctrl]).mean()
    don_pat = pd.Series(u_ctrl[~ctrl]).groupby(donor[~ctrl]).mean()
    summ["conformal_ctrl"] = conf_summary(S, res["unknown"], res["unknown_type"], known=known & ctrl)
    summ["conformal_bonf_ctrl"] = conf_summary(SB, res["unknown_bonf"], res["unknown_type_bonf"], known=known & ctrl)
    summ["conformal_old_ctrl"] = conf_summary(res_old["sets"], res_old["unknown"], None, known=known & ctrl)
    if has_labels and (known & ctrl).any():
        kc = known & ctrl
        summ["label_outside_set_ctrl"] = {m: float((~SB[kc][np.arange(kc.sum()), np.clip(pred[m][kc], 0, T - 1)]).mean())
                                          for m in ("ARCANE", "kNN-U", "kNN-S")}

    def fr(pv):
        u = pv <= ALPHA_U
        return dict(fa_ctrl=float(u[ctrl].mean()), flag_patient=float(u[~ctrl].mean()),
                    excess_patient=float((u[~ctrl].mean() - ALPHA_U) / (1 - ALPHA_U)),
                    fa_ctrl_donor_q90=float(pd.Series(u[ctrl]).groupby(donor[ctrl]).mean().quantile(0.9)))
    summ["control_recal"] = dict(n_ctrl_donors=int(len(dons)), cells_per_donor=per,
                                 fa_ctrl_lodo=float(u_ctrl[ctrl].mean()), fa_ctrl_donor_q90=float(don_fa.quantile(0.9)),
                                 fa_ctrl_donor_max=float(don_fa.max()),
                                 flag_patient=float(u_ctrl[~ctrl].mean()),
                                 excess_patient=float((u_ctrl[~ctrl].mean() - ALPHA_U) / (1 - ALPHA_U)),
                                 patient_donor_median=float(don_pat.median()),
                                 ref_cal_fa_ctrl=float(res["unknown"][ctrl].mean()),
                                 ref_cal_flag_patient=float(res["unknown"][~ctrl].mean()),
                                 bonf_fa_ctrl=float(res["unknown_bonf"][ctrl].mean()),
                                 bonf_flag_patient=float(res["unknown_bonf"][~ctrl].mean()),
                                 U_bonf_fa_ctrl=float(res["unknown_U_bonf"][ctrl].mean()),
                                 U_bonf_flag_patient=float(res["unknown_U_bonf"][~ctrl].mean()),
                                 U_ctrl=fr(p_ctrl_U), hierarchical=fr(p_ctrl_h))
    summ["bh"]["ARCANE control"] = bh_summary(p_ctrl, ~ctrl)
    summ["bh"]["ARCANE control U only"] = bh_summary(p_ctrl_U, ~ctrl)
    summ["bh"]["ARCANE control hierarchical"] = bh_summary(p_ctrl_h, ~ctrl)
    summ["bh"]["ARCANE control (controls)"] = bh_summary(p_ctrl, ctrl)
summ["calibration_counts"] = dict(zip(classes.tolist(), res["cal_counts"].tolist()))
if has_labels and (~known).any():
    nov = ~known
    PU, PS = res["PU"], res["PS"]
    pmax = lambda P: 1 - P.max(1)
    scores = {"ARCANE": 1 - res["p_unknown"], "ARCANE U only": 1 - res["p_unknown_U"],
              "ARCANE S only": 1 - res["p_unknown_S"], "ARCANE Bonferroni": 1 - res["p_unknown_bonf"],
              "ARCANE U Bonferroni": 1 - res["p_unknown_U_bonf"], "ARCANE-clf Bonferroni": 1 - res_clf["p_unknown_bonf"],
              "Stacking 1-max": 1 - lr.predict_proba(res["q_feat"]).max(1),
              "ARCANE type Bonferroni": 1 - res["p_unknown_type_bonf"], "ARCANE full-model distance": res["novelty"],
              "set p-value": 1.0 - res["pvals"].max(1), "1-max kNN-S": pmax(PS), "1-max kNN-U": pmax(PU),
              "1-max scANVI": 1 - prob.max(1), "kNN distance U": res["dU"], "kNN distance S": res["dS"],
              "U-S disagreement": (res["yU"] != res["yS"]).astype(float)}
    summ["novelty_auroc"] = {k: float(auroc(v, nov)) for k, v in scores.items()}

    def tpr_at(v, fpr=0.05):
        thr = np.quantile(v[~nov], 1 - fpr)
        return float((v[nov] > thr).mean())
    summ["novelty_tpr5"] = {k: tpr_at(np.asarray(v, float)) for k, v in scores.items()}
    summ["novel_frac"] = float(nov.mean())
    summ["novel_types"] = sorted(set(y_tgt[nov]))
# hierarchy: report the parent when the set holds several fine types that share one parent
par = {}
for d_ in folds + [F]:
    par.update(dict(zip(d_["y_ref"], d_["parent_ref"])))
if has_labels and any(par[c] != c for c in classes):
    lab_fine = classes[res["label"]]
    at_parent = np.zeros(len(y), bool)
    out_lab = lab_fine.astype(object).copy()
    for i in range(len(y)):
        members = classes[S[i]]
        parents = {par[c] for c in members}
        if len(members) > 1 and len(parents) == 1:
            out_lab[i] = parents.pop()
            at_parent[i] = True
    true_par = F["parent_tgt"]
    fine_ok = lab_fine == y_tgt
    par_ok = np.array([par.get(c, "") for c in lab_fine]) == true_par
    hier_ok = np.where(at_parent, out_lab == true_par, fine_ok)
    summ["hierarchy"] = dict(frac_parent=float(at_parent[known].mean()), hier_acc=float(hier_ok[known].mean()),
                             fine_acc=float(fine_ok[known].mean()), parent_acc_of_fine_label=float(par_ok[known].mean()),
                             fine_acc_on_parent_calls=float(fine_ok[known & at_parent].mean()) if (known & at_parent).any() else None,
                             parent_acc_on_parent_calls=float((out_lab == true_par)[known & at_parent].mean()) if (known & at_parent).any() else None,
                             fine_acc_on_fine_calls=float(fine_ok[known & ~at_parent].mean()))
summ["runtime_min"] = {s: (L(s)["runtime"] / 60).round(2).tolist() if "runtime" in L(s) else None for s in ["full"] + FOLDS}
summ["gpu_peak_gb"] = {s: float(L(s).get("gpu_peak_gb", np.nan)) for s in ["full"] + FOLDS}
summ["ram_peak_gb"] = {s: float(L(s).get("ram_peak_gb", np.nan)) for s in ["full"] + FOLDS}
summ["eval_minutes"] = (time.time() - t0) / 60
pd.DataFrame(rows).to_csv(f"{OUT}/{tag}_seed{seed}{suffix}_types.csv", index=False)
json.dump(summ, open(f"{OUT}/{tag}_seed{seed}{suffix}.json", "w"), indent=1)
if K == 15:
    np.savez_compressed(f"{OUT}/{tag}_seed{seed}_cells.npz", names=F["tgt_names"], y=y_tgt, cond=F["cond_tgt"],
                        batch=F["b_tgt"], novelty=res["novelty"], p_unknown=res["p_unknown"], unknown=res["unknown"],
                        unknown_type=res["unknown_type"], p_unknown_U=res["p_unknown_U"], p_unknown_S=res["p_unknown_S"],
                        set_size=S.sum(1), classes=classes, pvals=res["pvals"].astype(np.float32), sets=S,
                        scanvi_maxprob=prob.max(1), knn_dist_U=res["dU"], knn_dist_S=res["dS"],
                        pmax_S=res["PS"].max(1), pmax_U=res["PU"].max(1),
                        p_unknown_bonf=res["p_unknown_bonf"], p_unknown_type_bonf=res["p_unknown_type_bonf"],
                        sets_bonf=res["sets_bonf"], set_size_bonf=res["sets_bonf"].sum(1),
                        p_unknown_ctrl=p_ctrl if p_ctrl is not None else np.ones(len(y)),
                        p_unknown_ctrl_U=p_ctrl_U if p_ctrl_U is not None else np.ones(len(y)),
                        p_unknown_ctrl_h=p_ctrl_h if p_ctrl_h is not None else np.ones(len(y)),
                        p_unknown_U_bonf=res["p_unknown_U_bonf"], novelty_U=res["novelty_U"], novelty_S=res["novelty_S"],
                        label_arcane=classes[res["label"]],
                        **{f"pred_{k}": classes[np.clip(v, 0, T - 1)] for k, v in pred.items()})
print(tag, seed, K, {k: round(v["BA"], 3) for k, v in summ["methods"].items()}, summ.get("transfer_diagnostic"), flush=True)
