"""Task lists for the experiments. Usage: python make_tasks2.py A|B|C  -> tasks2_<X>_{train,eval,ct}.txt"""
import sys
X = sys.argv[1]
runs = []  # (train args without split, tag, seeds)


def add(ds, seeds, extra="", tag=None):
    for s in seeds:
        runs.append((f"--ds {ds} --seed {s} {extra}".strip(), tag or ds, s))


if X == "A":
    for ds, n in [("pbmc_fine", 3), ("spleen_lymph", 5), ("heart", 5)]:
        add(ds, range(n))
    for t in ["pDC", "MAIT", "Plasmablast"]:
        add("pbmc_fine", range(3), f"--holdout {t}", f"pbmc_fine_ho-{t}")
    for t in ["pDCs", "Neutrophils"]:
        add("spleen_lymph", range(3), f"--holdout {t}", f"spleen_lymph_ho-{t}")
    for t in ["pDC:20", "MAIT:20"]:
        add("pbmc_fine", range(3), f"--downsample {t}", f"pbmc_fine_ds-{t.replace(':', '-')}")
    for ds in ["pbmc_fine", "spleen_lymph"]:
        for nz in ["uniform:0.3", "swap:0.3"]:
            add(ds, range(3), f"--noise {nz}", f"{ds}_noise-{nz.replace(':', '-')}")
        add(ds, range(3), "--partial 0.05", f"{ds}_partial-0.05")
        add(ds, range(3), "--n_latent 30 --tagsuffix _nl30", f"{ds}_nl30")
        add(ds, range(3), "--scanvi_frac 1.0 --tagsuffix _sf1", f"{ds}_sf1")
    add("pbmc_fine", range(3), "--coarse", "pbmc_fine_coarse")
elif X == "B":
    add("hlca", range(3))
    for t in ["AT1", "Non-classical~monocytes"]:
        add("hlca", range(3), f"--holdout {t}", f"hlca_ho-{t}")
    add("hlca", range(3), "--coarse", "hlca_coarse")
    add("hlca", range(3), "--noise swap:0.3", "hlca_noise-swap-0.3")
    add("hlca_full", range(2))
elif X == "C":
    add("lupusfull", range(3))
with open(f"tasks2_{X}_train.txt", "w") as f:
    for a, t, s in runs:
        for sp in ["full", "cf0", "cf1"]:
            f.write(f"{a} --split {sp}\n")
with open(f"tasks2_{X}_eval.txt", "w") as f:
    for a, t, s in runs:
        f.write(f"{t} {s}\n")
    if X == "A":
        for ds, n in [("pbmc_fine", 3), ("spleen_lymph", 5)]:
            for s in range(n):
                for k in (5, 50):
                    f.write(f"{ds} {s} {k}\n")
with open(f"tasks2_{X}_ct.txt", "w") as f:
    for a, t, s in runs:
        if "--downsample" in a or "--noise" in a or "--partial" in a or "_nl30" in a or "_sf1" in a or "--coarse" in a:
            continue
        ds = a.split()[1]
        ho = a.split("--holdout ")[1].split()[0] if "--holdout" in a else ""
        f.write(f"{ds} {s} {ho}".strip() + "\n")
print(X, len(runs), "runs")
