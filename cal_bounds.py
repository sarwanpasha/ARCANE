"""Near-tie bounds (Theorem 2) for every evaluated run: recomputes the calibration label pairs of the fold models
and writes summary2/bounds.csv. Usage: python cal_bounds.py"""
import glob
import json
import os
import re
import zlib
import numpy as np
import pandas as pd
from arcane import knn_evidence, hard
from bound import compute_bounds

rows = []
os.makedirs("summary2", exist_ok=True)
for j in sorted(glob.glob("res2/*_seed*.json")):
    if re.search(r"_k\d+\.json$", j):
        continue
    tag, seed = re.match(r"res2/(.*)_seed(\d+)\.json", j).groups()
    try:
        L = lambda s: dict(np.load(f"emb/{tag}_seed{seed}_{s}.npz", allow_pickle=True))
        F = L("full")
        partial = float(tag.split("_partial-")[1].split("_")[0]) if "_partial-" in tag else None
        classes = np.array(sorted(set(F["y_ref"] if partial is None else F["y_ref"][F["labeled_ref"].astype(bool)])))
        ci = {c: i for i, c in enumerate(classes)}
        yU, yS, y, fo = [], [], [], []
        for f in (0, 1):
            d = L(f"cf{f}")
            m = d["labeled_ref"].astype(bool) if partial is not None and "labeled_ref" in d else slice(None)
            rU, rS, yr = d["ref_U"][m], d["ref_S"][m], d["y_ref"][m]
            cal = (d["tgt_role"] == "cal") & np.isin(d["y_tgt"], classes)
            if partial is not None:
                cal &= np.array([zlib.crc32(str(n).encode()) % 1000 < partial * 1000 for n in d["tgt_names"]])
            yU.append(hard(knn_evidence(rU, yr, d["tgt_U"][cal], classes, 15)[0]))
            yS.append(hard(knn_evidence(rS, yr, d["tgt_S"][cal], classes, 15)[0]))
            y.append(np.array([ci[c] for c in d["y_tgt"][cal]]))
            fo.append(np.full(cal.sum(), f))
        yU, yS, y, fo = map(np.concatenate, (yU, yS, y, fo))
        for meth in ("kl", "bernstein"):
            b = compute_bounds(yU, yS, y, fo, 0.05, meth)
            flat = {k: v for k, v in b.items() if isinstance(v, (int, float))}
            rows.append(dict(tag=tag, seed=int(seed), ci=meth, n_types=len(classes), **flat))
        print(tag, seed, "ok", flush=True)
    except Exception as e:
        print(tag, seed, "ERR", repr(e)[:200], flush=True)
pd.DataFrame(rows).to_csv("summary2/bounds.csv", index=False)
print("BOUNDS_DONE")
