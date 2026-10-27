"""Build data/std/lupus_nb.h5ad: reference = the healthy reference donors of lupusfull (labels, folds), query =
all GSE135779 donors (unlabeled), condition 'normal' for healthy donors. Basic QC and at most 2000 cells per donor."""
import gzip, os, re, glob
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp, anndata as ad
R = "data/raw/nb/raw"
fs = sorted(os.listdir(R))
print("raw files", len(fs), fs[:12], flush=True)
# sample metadata from the series matrix
meta = {}
for mf in glob.glob("data/raw/nb/*series_matrix*.txt.gz"):
    rows = {}
    for line in gzip.open(mf, "rt", errors="ignore"):
        if line.startswith("!Sample_"):
            k, *v = line.rstrip("\n").split("\t")
            rows.setdefault(k, []).append([x.strip('"') for x in v])
    gsm = rows["!Sample_geo_accession"][0]
    title = rows["!Sample_title"][0]
    ch = [" | ".join(x) for x in zip(*rows.get("!Sample_characteristics_ch1", [[""] * len(gsm)]))]
    src = rows.get("!Sample_source_name_ch1", [[""] * len(gsm)])[0]
    for g, t, c, s in zip(gsm, title, ch, src):
        meta[g] = dict(title=t, chars=c, source=s)
print("samples in matrix", len(meta), list(meta.items())[:4], flush=True)
# group 10x files by GSM
groups = {}
for f in fs:
    m = re.match(r"(GSM\d+)", f)
    if m:
        groups.setdefault(m.group(1), []).append(f)
rng = np.random.default_rng(0)
parts = []
for g, ff in sorted(groups.items()):
    try:
        h5 = [f for f in ff if f.endswith(".h5")]
        if h5:
            import scanpy as sc
            a = sc.read_10x_h5(os.path.join(R, h5[0]))
        else:
            mtx = [f for f in ff if "matrix" in f and f.endswith((".mtx.gz", ".mtx"))][0]
            bc = [f for f in ff if "barcodes" in f][0]
            gl = [f for f in ff if ("genes" in f or "features" in f)]
            gn = gl[0] if gl else None
            X = scipy.io.mmread(os.path.join(R, mtx)).T.tocsr().astype(np.float32)
            b = pd.read_csv(os.path.join(R, bc), header=None, sep="\t")[0].astype(str).values
            gpath = os.path.join(R, gn) if gn else sorted(glob.glob("data/raw/nb/*genes*.tsv*") + glob.glob("data/raw/nb/*features*.tsv*"))[0]
            G = pd.read_csv(gpath, header=None, sep="\t")
            sym = G[1].astype(str).values if G.shape[1] > 1 else G[0].astype(str).values
            a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"{g}_{x}" for x in b]), var=pd.DataFrame(index=sym))
        a.var_names_make_unique()
        a.obs_names = [f"{g}_{x}" if not x.startswith(g) else x for x in a.obs_names]
        X = a.X.tocsr()
        ng = np.asarray((X > 0).sum(1)).ravel()
        mt = np.asarray(X[:, a.var_names.str.upper().str.startswith("MT-")].sum(1)).ravel() / np.maximum(np.asarray(X.sum(1)).ravel(), 1)
        keep = np.where((ng >= 300) & (ng <= 5000) & (mt < 0.15))[0]
        if len(keep) > 2000:
            keep = np.sort(rng.choice(keep, 2000, replace=False))
        a = a[keep].copy()
        md = meta.get(g, dict(title=g, chars="", source=""))
        txt = (md["title"] + " " + md["chars"] + " " + md["source"]).upper()
        healthy = bool(re.search(r"\bHD|HEALTHY|CONTROL|\bCHD|\bAHD", txt)) and not re.search(r"SLE\d|CSLE|ASLE|LUPUS", md["title"].upper())
        a.obs["batch"] = g
        a.obs["sample_title"] = md["title"]
        a.obs["sample_chars"] = md["chars"]
        a.obs["condition"] = "normal" if healthy else "systemic lupus erythematosus"
        a.obs["age_group"] = "child" if re.search(r"\bC(SLE|HD)|CHILD|PEDIATRIC|\bC\d", md["title"].upper()) else "adult"
        parts.append(a)
        print(g, md["title"], a.obs.condition.iloc[0] if a.n_obs else "", a.n_obs, flush=True)
    except Exception as e:
        print("skip", g, ff, repr(e), flush=True)
q = ad.concat(parts, join="inner")
q.obs["label"] = "unlabeled"; q.obs["parent"] = "unlabeled"; q.obs["role"] = "query"; q.obs["fold"] = -1
L = ad.read_h5ad("data/std/lupusfull.h5ad")
ref = L[L.obs.role == "ref"].copy()
del L
genes = ref.var_names.intersection(q.var_names)
print("common genes", len(genes), "ref", ref.shape, "query", q.shape,
      "donors", q.obs.batch.nunique(), q.obs.groupby("condition").batch.nunique().to_dict(), flush=True)
keep = ["label", "parent", "batch", "role", "fold", "condition"]
ref.obs["age_group"] = "adult"
q = q[:, genes].copy(); ref = ref[:, genes].copy()
o = pd.concat([ref.obs[keep + ["age_group"]], q.obs[keep + ["age_group", "sample_title"]]])
X = sp.vstack([sp.csr_matrix(ref.X), sp.csr_matrix(q.X)]).tocsr().astype(np.float32)
for c in o.columns:
    o[c] = o[c].astype(str)
o["fold"] = o["fold"].astype(int)
a = ad.AnnData(X=X, obs=o, var=pd.DataFrame(index=genes))
a.write_h5ad("data/std/lupus_nb.h5ad", compression="gzip")
a.obs[a.obs.role == "query"].groupby("batch").agg(condition=("condition", "first"), age=("age_group", "first"),
    title=("sample_title", "first"), n=("label", "size")).to_csv("data/std/lupus_nb_donors.csv")
print("NB_PREP_DONE", a.shape, flush=True)
