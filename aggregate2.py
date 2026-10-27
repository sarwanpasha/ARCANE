"""Aggregate res2/ outputs of evaluate2.py into compact tables (summary2/*.csv and summary2/RESULTS.txt)."""
import glob
import json
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

os.makedirs("summary2", exist_ok=True)
J = [json.load(open(f)) for f in sorted(glob.glob("res2/*_seed*.json"))]
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


def setting(tag):
    m = re.match(r"^(pbmc_fine|spleen_lymph|heart|hlca_full|hlca|lupusfull|lupus|covid_stephenson|covid_yoshida)(.*)$", tag)
    return (m.group(1), m.group(2) or "main") if m else (tag, "?")


rows = []
for j in J:
    ds, st = setting(j["tag"])
    for m, v in j["methods"].items():
        rows.append(dict(ds=ds, setting=st, k=j.get("k", 15), seed=j["seed"], method=m, **v))
M = pd.DataFrame(rows)
M.to_csv("summary2/methods.csv", index=False)
for metric in ["BA", "macroF1", "acc", "acc_reversed"]:
    tab = M[M.k == 15].groupby(["ds", "setting", "method"])[metric].mean().unstack("method").round(4)
    tab.to_csv(f"summary2/{metric}.csv")
    P(f"\n== {metric} (mean over seeds, k=15) ==")
    P(tab.to_string())
P("\n== BA standard deviation over seeds ==")
P(M[M.k == 15].groupby(["ds", "setting", "method"]).BA.std().unstack("method").round(4).to_string())
P("\n== k sensitivity (BA) ==")
P(M[M.setting == "main"].groupby(["ds", "k", "method"]).BA.mean().unstack("method").round(4).to_string())

# per-type comparison vs better view, and paired tests, per setting
TY = pd.concat([pd.read_csv(f) for f in glob.glob("res2/*_types.csv") if "_k" not in os.path.basename(f).split("seed")[1]],
               ignore_index=True)
TY["ds"], TY["setting"] = zip(*TY.tag.map(setting))
TY.to_csv("summary2/types.csv", index=False)
P("\n== per-type recall vs better of kNN-U/kNN-S: within 0.01 (no_harm), mean change, worst ==")
for st, g in TY.groupby("setting"):
    if st.startswith("_ho-") or st.startswith("_ds-"):
        continue
    piv = g.pivot_table(index=["ds", "seed", "type"], columns="method", values="recall")
    best = piv[["kNN-U", "kNN-S"]].max(axis=1)
    line = []
    for m in ["kNN-U", "kNN-S", "Average", "Concat", "scANVI-clf", "scANVI-EM", "ARCANE", "ARCANE-acc"]:
        if m in piv:
            d = piv[m] - best
            line.append(f"{m}: nh={(d >= -0.01).mean():.3f} mean={d.mean():+.4f} worst={d.min():+.3f}")
    P(st, "|", " ; ".join(line))
P("\n== paired Wilcoxon on per-type F1 and recall: ARCANE minus method ==")
for st, g in TY.groupby("setting"):
    if st.startswith("_ho-") or st.startswith("_ds-"):
        continue
    for val in ["f1", "recall"]:
        pf = g.pivot_table(index=["ds", "seed", "type"], columns="method", values=val)
        for m in ["kNN-U", "kNN-S", "Average", "Concat", "scANVI-EM"]:
            if m in pf:
                d = (pf["ARCANE"] - pf[m]).dropna()
                p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                P(f"{st} {val} vs {m}: mean {d.mean():+.4f} W/T/L {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()} p={p:.1e}")
# conformal
C = []
for j in J:
    if j.get("k", 15) != 15:
        continue
    ds, st = setting(j["tag"])
    for name in ["conformal", "conformal_bonf", "conformal_rand", "conformal_old", "conformal_ctrl", "conformal_bonf_ctrl", "conformal_old_ctrl"]:
        if name not in j:
            continue
        c = {k: v for k, v in j[name].items() if k != "per_type_cov"}
        C.append(dict(ds=ds, setting=st, seed=j["seed"], calib=name, **c))
C = pd.DataFrame(C)
C.to_csv("summary2/conformal.csv", index=False)
P("\n== conformal: symmetric (conformal) vs previous (conformal_old) ==")
cols = [c for c in ["mean_cov", "min_cov", "frac_types_cov085", "set_size", "singleton", "unknown_known", "unknown_novel",
                    "unknown_type_known", "unknown_type_novel"] if c in C]
P(C.groupby(["ds", "setting", "calib"])[cols].mean().round(3).to_string())
CR = pd.DataFrame([dict(tag=j["tag"], seed=j["seed"], **j["control_recal"]) for j in J if "control_recal" in j and j.get("k", 15) == 15])
if len(CR):
    CR.to_csv("summary2/control_recal.csv", index=False)
    P("\n== unknown flag recalibrated on query control donors (leave one donor out) ==")
    P(CR.groupby("tag").mean(numeric_only=True).drop(columns="seed").round(4).T.to_string())
# novelty
N = pd.DataFrame([dict(tag=j["tag"], seed=j["seed"], **j["novelty_auroc"]) for j in J if "novelty_auroc" in j and j.get("k", 15) == 15])
if len(N):
    N.to_csv("summary2/novelty.csv", index=False)
    P("\n== novelty AUROC ==")
    P(N.groupby("tag").mean(numeric_only=True).drop(columns="seed").round(3).T.to_string())
# hierarchy
H = pd.DataFrame([dict(tag=j["tag"], seed=j["seed"], **j["hierarchy"]) for j in J if "hierarchy" in j and j.get("k", 15) == 15])
if len(H):
    H.to_csv("summary2/hierarchy.csv", index=False)
    P("\n== hierarchy rule ==")
    P(H.groupby("tag").mean(numeric_only=True).drop(columns="seed").round(3).to_string())
# runtime
R = []
for j in J:
    if j.get("k", 15) != 15:
        continue
    for s, v in j.get("runtime_min", {}).items():
        if v:
            R.append(dict(tag=j["tag"], seed=j["seed"], split=s, scvi_min=v[0], scanvi_min=v[1] - 0, total_min=v[2],
                          gpu_gb=j["gpu_peak_gb"].get(s), ram_gb=j["ram_peak_gb"].get(s), eval_min=j.get("eval_minutes")))
R = pd.DataFrame(R)
if len(R):
    R.to_csv("summary2/runtime.csv", index=False)
    P("\n== runtime (minutes) and peak memory (GB), main settings ==")
    R["ds"], R["setting"] = zip(*R.tag.map(setting))
    P(R[R.setting == "main"].groupby(["ds", "split"])[["scvi_min", "total_min", "gpu_gb", "ram_gb", "eval_min"]].mean().round(2).to_string())
# downsample
D = TY[TY.setting.str.startswith("_ds-")].copy()
if len(D):
    D["target"] = D.setting.str.extract(r"_ds-(.*)-\d+$")[0]
    D = D[D.type == D.target]
    P("\n== downsampled types ==")
    P(D.groupby(["setting", "method"])[["recall", "f1"]].mean().round(3).to_string())
open("summary2/RESULTS.txt", "w").write("\n".join(out))
print("AGG2_DONE")
