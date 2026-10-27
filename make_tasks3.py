"""Task lists for the full rerun. Usage: python make_tasks3.py A|B|C
Writes tasks3_<X>_{train,eval,ct,comp}.txt. Holdout labels are checked against data/std/<ds>.h5ad."""
import os
import sys
import anndata as ad

X = sys.argv[1]
runs = []  # (train args without split, tag, seed, nfolds)
_labels = {}


def labels(ds):
    if ds not in _labels:
        a = ad.read_h5ad(f"data/std/{ds}.h5ad", backed="r")
        _labels[ds] = set(a.obs.label.astype(str))
    return _labels[ds]


def match(ds, t):
    L = labels(ds)
    t0 = t.replace("~", " ")
    if t0 in L:
        return t
    for l in L:
        if l.lower() == t0.lower():
            return l.replace(" ", "~")
    print("holdout label not found", ds, t, sorted(L)[:60], flush=True)
    return None


def add(ds, seeds, extra="", tag=None, nfolds=2):
    for s in seeds:
        runs.append((f"--ds {ds} --seed {s} {extra}".strip(), tag or ds, s, nfolds))


def add_ho(ds, types, seeds):
    for t in types:
        m = match(ds, t)
        if m:
            add(ds, seeds, f"--holdout {m}", f"{ds}_ho-{m}")


comp, ct_extra, keval = [], [], []
if X == "A":
    for ds, n in [("pbmc_fine", 3), ("spleen_lymph", 5), ("heart", 5)]:
        add(ds, range(n))
    add_ho("pbmc_fine", ["pDC", "MAIT", "Plasmablast"], range(3))
    add_ho("spleen_lymph", ["pDCs", "Neutrophils"], range(3))
    for t in ["pDC:20", "MAIT:20"]:
        add("pbmc_fine", range(3), f"--downsample {t}", f"pbmc_fine_ds-{t.replace(':', '-')}")
    ds = "pbmc_fine"
    for nz in ["uniform:0.3", "swap:0.3"]:
        add(ds, range(3), f"--noise {nz}", f"{ds}_noise-{nz.replace(':', '-')}")
    add(ds, range(3), "--partial 0.05", f"{ds}_partial-0.05")
    add(ds, range(3), "--n_latent 30 --tagsuffix _nl30", f"{ds}_nl30")
    add(ds, range(3), "--scanvi_frac 1.0 --tagsuffix _sf1", f"{ds}_sf1")
    add(ds, range(3), "--coarse", f"{ds}_coarse")
    for ds in ["pbmc_fine", "spleen_lymph", "heart"]:
        for s in range(3):
            comp.append(f"{ds} {s}")
    for t in ["pDC", "MAIT", "Plasmablast"]:
        for s in range(2):
            comp.append(f"pbmc_fine {s} {t}")
    for t in ["pDCs", "Neutrophils"]:
        for s in range(2):
            comp.append(f"spleen_lymph {s} {t}")
    for ds, n in [("pbmc_fine", 3), ("spleen_lymph", 5), ("heart", 5)]:
        for s in range(n):
            keval += [f"{ds} {s} 5"] + ([f"{ds} {s} 50"] if ds != "heart" else [])
elif X == "B":
    add("hlca", range(3))
    add_ho("hlca", ["AT1", "Non-classical~monocytes"], range(3))
    add("hlca", range(3), "--coarse", "hlca_coarse")
    add("hlca", range(3), "--noise swap:0.3", "hlca_noise-swap-0.3")
    add("hlca", range(3), "--nfolds 5 --tagsuffix _loso", "hlca_loso", nfolds=5)
    add("hlca_full", range(3))
    ds = "spleen_lymph"
    for nz in ["uniform:0.3", "swap:0.3"]:
        add(ds, range(3), f"--noise {nz}", f"{ds}_noise-{nz.replace(':', '-')}")
    add(ds, range(3), "--partial 0.05", f"{ds}_partial-0.05")
    add(ds, range(3), "--n_latent 30 --tagsuffix _nl30", f"{ds}_nl30")
    add(ds, range(3), "--scanvi_frac 1.0 --tagsuffix _sf1", f"{ds}_sf1")
    add(ds, range(3), "--coarse", f"{ds}_coarse")
    for ds in ["covid_stephenson", "covid_yoshida"]:
        if os.path.exists(f"data/std/{ds}.h5ad"):
            add(ds, range(3))
    for s in range(3):
        comp.append(f"hlca {s}")
    for t in ["AT1", "Non-classical~monocytes"]:
        for s in range(2):
            comp.append(f"hlca {s} {t}")
    for ds in ["hlca", "hlca_full", "covid_stephenson", "covid_yoshida"]:
        for s in range(3):
            keval.append(f"{ds} {s} 5")
elif X == "C":
    add("lupusfull", range(3))
    add("lupusfull", range(3), "--drop_isg 300", "lupusfull_isgblind300")
    add_ho("lupusfull", ["ncM", "pDC"], range(3))
    if os.path.exists("data/std/lupus_nb.h5ad"):
        add("lupus_nb", range(3))
    for s in range(3):
        keval.append(f"lupusfull {s} 5")
with open(f"tasks3_{X}_train.txt", "w") as f:
    for a, t, s, nf in runs:
        if nf == 2:
            sp = ["full", "cf0", "cf1"]
        else:
            sp = [f"cf{i}" for i in range(nf)]  # the full model is shared with the 2-fold run (symlinked)
        for x in sp:
            f.write(f"{a} --split {x}\n")
with open(f"tasks3_{X}_eval.txt", "w") as f:
    for a, t, s, nf in runs:
        f.write(f"{t} {s}\n")
    for k in keval:
        f.write(k + "\n")
with open(f"tasks3_{X}_ct.txt", "w") as f:
    for a, t, s, nf in runs:
        if any(x in a for x in ("--downsample", "--noise", "--partial", "_nl30", "_sf1", "--coarse", "--drop_isg", "_loso")):
            continue
        ds = a.split()[1]
        ho = a.split("--holdout ")[1].split()[0] if "--holdout" in a else ""
        f.write(f"{ds} {s} {ho}".strip() + "\n")
with open(f"tasks3_{X}_comp.txt", "w") as f:
    for c in comp:
        f.write(c + "\n")
print(X, len(runs), "runs", len(comp), "competitor runs")
