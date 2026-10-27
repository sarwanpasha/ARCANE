"""Compact summary of all res2 runs (evaluate2.py) and competitor summaries (comp_eval.py).
Writes summary3/*.csv and summary3/KEY.txt (short, for reading)."""
import glob
import json
import os
import re
import numpy as np
import pandas as pd

os.makedirs("summary3", exist_ok=True)
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)


def setting(tag):
    m = re.match(r"^(pbmc_fine|spleen_lymph|heart|hlca_full|hlca_loso|hlca|lupusfull|lupus_nb|covid_stephenson|covid_yoshida)(.*)$", tag)
    return (m.group(1), m.group(2) or "main") if m else (tag, "?")


J = []
for f in sorted(glob.glob("res2/*_seed*.json")):
    try:
        J.append(json.load(open(f)))
    except Exception:
        pass
rows, other = [], []
for j in J:
    ds, st = setting(j["tag"])
    for m, v in j.get("methods", {}).items():
        rows.append(dict(ds=ds, setting=st, k=j.get("k", 15), seed=j["seed"], method=m, **v))
    r = dict(ds=ds, setting=st, k=j.get("k", 15), seed=j["seed"])
    for blk in ["conformal_bonf", "conformal_bonf_U", "conformal_clf", "conformal_bonf_ctrl", "conformal"]:
        if blk in j:
            for kk, vv in j[blk].items():
                if isinstance(vv, (int, float)) or vv is None:
                    r[f"{blk}.{kk}"] = vv
    for blk in ["label_outside_set", "label_outside_set_wrong", "label_outside_set_right", "label_outside_set_ctrl",
                "novelty_auroc", "novelty_tpr5", "transfer_diagnostic", "hierarchy", "control_recal"]:
        for kk, vv in (j.get(blk) or {}).items():
            if isinstance(vv, (int, float, bool)):
                r[f"{blk}.{kk}"] = float(vv)
            elif isinstance(vv, dict):
                for k2, v2 in vv.items():
                    if isinstance(v2, (int, float)):
                        r[f"{blk}.{kk}.{k2}"] = v2
    for kk, vv in (j.get("bh") or {}).items():
        for k2, v2 in vv.items():
            if isinstance(v2, (int, float)):
                r[f"bh.{kk}.{k2}"] = v2
    for kk, vv in (j.get("symmetry") or {}).items():
        for i, d in enumerate(vv):
            for k2, v2 in d.items():
                r[f"sym.{kk}.f{i}.{k2}"] = v2
    rt = j.get("runtime_min") or {}
    r["runtime_total_min"] = float(sum((v[-1] if isinstance(v, list) else 0) for v in rt.values() if v))
    r["eval_min"] = j.get("eval_minutes")
    other.append(r)
M = pd.DataFrame(rows)
O = pd.DataFrame(other)
M.to_csv("summary3/methods.csv", index=False)
O.to_csv("summary3/other.csv", index=False)
K15 = M[M.k == 15]
meth = ["kNN-U", "kNN-S", "Average", "Concat", "Stacking", "scANVI-clf", "scANVI-EM", "CellTypist", "ARCANE", "ARCANE-acc", "ARCANE-clf"]
for metric in ["BA", "macroF1", "acc_reversed"]:
    t = K15.groupby(["ds", "setting", "method"])[metric].agg(["mean", "std", "count"])
    tab = t["mean"].unstack("method").reindex(columns=[m for m in meth if m in t.index.get_level_values(2)])
    sd = t["std"].unstack("method").reindex(columns=tab.columns)
    n = t["count"].unstack("method").reindex(columns=tab.columns)
    tab.round(4).to_csv(f"summary3/{metric}_mean.csv")
    sd.round(4).to_csv(f"summary3/{metric}_sd.csv")
    P(f"## {metric} mean (sd) n_seeds")
    for idx in tab.index:
        P(" ".join([f"{idx[0]}{idx[1]}"] + [f"{c}={tab.loc[idx, c]:.3f}({sd.loc[idx, c]:.3f},{int(n.loc[idx, c]) if n.loc[idx, c] == n.loc[idx, c] else 0})"
                                             for c in tab.columns if tab.loc[idx, c] == tab.loc[idx, c]]))
P("## k sensitivity BA (main)")
ks = M[M.setting.isin(["main", "_symph", "_scpoli"])].groupby(["ds", "setting", "k", "method"]).BA.mean().unstack("method")
for idx in ks.index:
    P(" ".join([f"{idx[0]}{idx[1]} k={idx[2]}"] + [f"{c}={ks.loc[idx, c]:.3f}" for c in ["kNN-U", "kNN-S", "ARCANE", "Stacking"] if c in ks.columns and ks.loc[idx, c] == ks.loc[idx, c]]))
# per-type no-harm vs better view
TY = []
for f in glob.glob("res2/*_types.csv"):
    if re.search(r"_seed\d+_k\d+_types", f):
        continue
    try:
        TY.append(pd.read_csv(f))
    except Exception:
        pass
if TY:
    TY = pd.concat(TY, ignore_index=True)
    TY["ds"], TY["setting"] = zip(*TY.tag.map(setting))
    P("## per-type recall minus better view: frac within .01 / mean / worst")
    for (ds, st), g in TY.groupby(["ds", "setting"]):
        piv = g.pivot_table(index=["seed", "type"], columns="method", values="recall")
        if not {"kNN-U", "kNN-S"} <= set(piv.columns):
            continue
        best = piv[["kNN-U", "kNN-S"]].max(axis=1)
        P(f"{ds}{st} " + " ".join(f"{m}={((piv[m] - best) >= -0.01).mean():.2f}/{(piv[m] - best).mean():+.4f}/{(piv[m] - best).min():+.3f}"
                                  for m in ["ARCANE", "Stacking", "Average", "Concat", "ARCANE-clf"] if m in piv))
# other metrics: mean over seeds
OM = O[O.k == 15].groupby(["ds", "setting"]).mean(numeric_only=True)
OM.to_csv("summary3/other_mean.csv")
OS = O[O.k == 15].groupby(["ds", "setting"]).std(numeric_only=True)
OS.to_csv("summary3/other_sd.csv")


def sel(prefix_list, title):
    P(f"## {title}")
    cols = [c for c in OM.columns if any(c.startswith(p) for p in prefix_list)]
    for idx in OM.index:
        short = lambda c: next((c[len(p):] or c for p in prefix_list if c.startswith(p)), c).lstrip(".")
        vals = [f"{short(c)}={OM.loc[idx, c]:.3f}" for c in cols if OM.loc[idx, c] == OM.loc[idx, c]]
        if vals:
            P(f"{idx[0]}{idx[1]} " + " ".join(vals))


sel(["conformal_bonf."], "sets fold-Bonferroni (reference calibration)")
sel(["conformal_bonf_U.unknown_known", "conformal_bonf_U.unknown_novel", "conformal_clf.mean_cov", "conformal_clf.min_cov", "conformal_clf.set_size"], "U-only flag; ARCANE-clf sets")
sel(["label_outside_set"], "label outside set")
sel(["novelty_auroc."], "novelty AUROC")
sel(["novelty_tpr5."], "novelty TPR at 5% FPR")
sel(["transfer_diagnostic."], "transfer diagnostic")
sel(["control_recal."], "control calibration")
sel(["bh."], "BH q=0.1")
sel(["sym."], "symmetry check (fold scores: calibration vs healthy query)")
sel(["hierarchy."], "hierarchy")
sel(["runtime_total_min", "eval_min"], "runtime (min, sum over full and fold models; eval)")
# competitor summaries
C = []
for f in glob.glob("res/comp/summary_*_seed*.json"):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    for m, v in j.get("methods", {}).items():
        if "-file:" in m:
            continue
        r = dict(tag=j["tag"], seed=j["seed"], method=m)
        for k in ["BA", "macroF1", "unknown_known", "unknown_novel", "auroc_novel_score", "auroc_error_score", "runtime_s",
                  "cov_mean", "cov_min", "set_size", "ram_peak_gb", "gpu_peak_gb"]:
            x = v.get(k)
            r[k] = x if isinstance(x, (int, float)) else None
        C.append(r)
if C:
    C = pd.DataFrame(C)
    C.to_csv("summary3/competitors.csv", index=False)
    for v in ["BA", "unknown_known", "auroc_error_score", "runtime_s", "unknown_novel", "auroc_novel_score"]:
        t = C.groupby(["tag", "method"])[v].agg(["mean", "count"]).dropna()
        P(f"## competitors {v} mean(n)")
        for tag, g in t.groupby(level=0):
            P(tag + " " + " ".join(f"{m}={r['mean']:.3f}({int(r['count'])})" for (_, m), r in g.iterrows()))
open("summary3/KEY.txt", "w").write("\n".join(out) + "\n")
print("WROTE summary3/KEY.txt", len(out), "lines;", len(J), "runs")
