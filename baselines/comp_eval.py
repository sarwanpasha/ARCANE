"""Evaluate all competitor outputs (and ARCANE, if present) for one tag and seed.

Usage: python comp_eval.py <tag> <seed> [--min_n 10]
  tag = <ds> or <ds>_ho-<Type> ('~' for spaces in <Type>), as used by train.py / comp_*.py.
Reads  res/comp/<method>_<tag>_seed<s>.npz (keys names, pred, score, [sets, classes],
       [pred__v, score__v, sets__v], meta), plus ARCANE cells files if present:
       res2/<tag>_seed<s>_cells.npz or res/<tag>_seed<s>_cells.npz  -> 'ARCANE', 'ARCANE-noflag'
       res2/<tag>_scpoli_seed<s>_cells.npz (ARCANE on the (scVI, scPoli) pair) -> 'ARCANE[scVI+scPoli]'
Truth: data/std/<ds>.h5ad obs (label, condition, role); fallback emb/<tag>_seed<s>_full.npz.
Writes res/comp/summary_<tag>_seed<s>.json, res/comp/summary_<tag>_seed<s>_types.csv (per-type
recall/precision/F1/set coverage) and res/comp/summary_<tag>_seed<s>_donor.csv (per donor/batch and
condition: unknown-call rate and mean score, e.g. for lupus donor-level analyses).

Metrics (known types = reference types after holdout removal; evaluated types = known types with
>= min_n query cells; 'unknown' predictions count as errors for BA / F1 / recall):
  BA, macroF1, acc, per-type recall/precision/F1; BA_accepted (recall among non-unknown calls);
  unknown rate in known cells (overall and per condition), in all cells per condition, in novel cells;
  novelty AUROC of score (held-out type cells vs known-type cells) and of the binary unknown call;
  error-detection AUROC of score on known cells (misclassified vs correct);
  per-condition BA; prediction-set mean/min per-type coverage, mean size, singleton and empty rates.
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from comp_common import uniq

UNKNOWN = "unknown"


def auroc(score, pos):
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def parse_tag(tag):
    if "_ho-" in tag:
        ds, ho = tag.split("_ho-", 1)
        return ds, ho.replace("~", " ")
    return tag, None


def load_truth(tag, seed):
    ds, ho = parse_tag(tag)
    path = f"data/std/{ds}.h5ad"
    if os.path.exists(path):
        import anndata as ad
        a = ad.read_h5ad(path, backed="r")
        obs = a.obs
        for c in ("label", "role"):
            if c not in obs:
                raise KeyError(f"{path} lacks obs column {c}")
        lab, role = obs["label"].astype(str).values, obs["role"].astype(str).values
        cond = obs["condition"].astype(str).values if "condition" in obs else np.array(["none"] * len(obs))
        batch = obs["batch"].astype(str).values if "batch" in obs else np.array(["none"] * len(obs))
        ref_lab = lab[role == "ref"]
        q = role == "query"
        names = uniq(obs.index.values.astype(str)[q])
        truth = pd.DataFrame({"y": lab[q], "cond": cond[q], "batch": batch[q]}, index=names)
        src = path
    else:
        e = f"emb/{tag}_seed{seed}_full.npz"
        if not os.path.exists(e):
            raise FileNotFoundError(f"neither {path} nor {e} exists")
        d = np.load(e, allow_pickle=True)
        m = d["tgt_role"].astype(str) == "query" if "tgt_role" in d else slice(None)
        truth = pd.DataFrame({"y": d["y_tgt"].astype(str)[m],
                              "cond": d["cond_tgt"].astype(str)[m] if "cond_tgt" in d else "none",
                              "batch": d["b_tgt"].astype(str)[m] if "b_tgt" in d else "none"},
                             index=uniq(d["tgt_names"].astype(str)[m]))
        ref_lab = d["y_ref"].astype(str)
        src = e
    classes = np.array(sorted(set(ref_lab) - ({ho} if ho else set())))
    if ho and ho not in set(truth.y):
        print(f"[eval] WARNING holdout type '{ho}' absent from query truth", flush=True)
    return truth, classes, ho, src


def collect(tag, seed):
    """Return {method: dict(names, pred, score, sets, classes, meta)} including variants."""
    out, errors = {}, {}
    suffix = f"_{tag}_seed{seed}.npz"
    for f in sorted(glob.glob(f"res/comp/*{suffix}")):
        meth = os.path.basename(f)[: -len(suffix)]
        if meth.startswith("summary") or "_" in meth and meth.split("_")[-1].startswith("ho-"):
            continue
        try:
            d = dict(np.load(f, allow_pickle=True))
        except Exception as e:  # noqa
            errors[meth] = f"unreadable: {e!r}"
            continue
        meta = json.loads(str(d["meta"])) if "meta" in d else {}
        cls = d.get("classes")
        base = dict(names=uniq(d["names"]), pred=d["pred"].astype(str), score=d["score"].astype(float),
                    sets=d.get("sets"), classes=cls, meta=meta)
        out[meth] = base
        vs = sorted({k.split("__", 1)[1] for k in d if "__" in k})
        for v in vs:
            out[f"{meth}:{v}"] = dict(names=base["names"],
                                      pred=d[f"pred__{v}"].astype(str) if f"pred__{v}" in d else base["pred"],
                                      score=d[f"score__{v}"].astype(float) if f"score__{v}" in d else base["score"],
                                      sets=d.get(f"sets__{v}"), classes=cls, meta={"variant_of": meth})
    for f in sorted(glob.glob(f"res/comp/*_{tag}_seed{seed}*.error.json")):
        meth = os.path.basename(f).split(f"_{tag}_seed{seed}")[0]
        try:
            errors[meth] = json.load(open(f)).get("error")
        except Exception:  # noqa
            errors[meth] = "error file unreadable"
    # ARCANE outputs of evaluate2.py
    for name, cand in (("ARCANE", [f"res2/{tag}_seed{seed}_cells.npz", f"res/{tag}_seed{seed}_cells.npz"]),
                       ("ARCANE[scVI+scPoli]", [f"res2/{tag}_scpoli_seed{seed}_cells.npz",
                                                f"res/{tag}_scpoli_seed{seed}_cells.npz"])):
        p = next((c for c in cand if os.path.exists(c)), None)
        if p is None:
            continue
        d = dict(np.load(p, allow_pickle=True))
        lab = d["pred_ARCANE"].astype(str)
        unk = d["unknown"].astype(bool)
        sc = 1.0 - d["p_unknown"].astype(float) + 1e-9 * d["novelty"] if "p_unknown" in d else d["novelty"].astype(float)
        pr = np.where(unk, UNKNOWN, lab)
        out[name] = dict(names=uniq(d["names"]), pred=pr, score=sc, sets=d.get("sets"),
                         classes=d.get("classes"), meta={"source": p})
        out[name + "-noflag"] = dict(names=uniq(d["names"]), pred=lab, score=sc, sets=None, classes=None,
                                     meta={"source": p})
        for k in ("kNN-U", "kNN-S", "scANVI-clf", "scANVI-EM"):
            if f"pred_{k}" in d:
                out[f"{name}-file:{k}"] = dict(names=uniq(d["names"]), pred=d[f"pred_{k}"].astype(str),
                                               score=sc, sets=None, classes=None, meta={"source": p})
    return out, errors


def evaluate(m, truth, classes, min_n):
    names = m["names"]
    inter = truth.index.intersection(pd.Index(names))
    n_missing = int(len(truth) - len(inter))
    pos = pd.Series(np.arange(len(names)), index=names)
    pos = pos[~pos.index.duplicated()].reindex(inter).values
    t = truth.loc[inter]
    y, cond = t.y.values, t.cond.values
    pred, score = m["pred"][pos], m["score"][pos]
    known = np.isin(y, classes)
    novel = ~known
    unk = pred == UNKNOWN
    n_q = {c: int((y == c).sum()) for c in classes}
    ev = [c for c in classes if n_q[c] >= min_n]
    rec, prec, f1, rec_acc = {}, {}, {}, {}
    for c in ev:
        mc = y == c
        rec[c] = float((pred[mc] == c).mean())
        pc = pred == c
        prec[c] = float((y[pc] == c).mean()) if pc.any() else 0.0
        f1[c] = 0.0 if rec[c] + prec[c] == 0 else 2 * rec[c] * prec[c] / (rec[c] + prec[c])
        acc_m = mc & ~unk
        rec_acc[c] = float((pred[acc_m] == c).mean()) if acc_m.any() else None
    r = dict(_donor=pd.DataFrame({"batch": t.batch.values, "cond": cond, "unknown": unk, "score": score,
                                  "known": known}).groupby(["batch", "cond"]).agg(
                 n=("unknown", "size"), unknown_rate=("unknown", "mean"), mean_score=("score", "mean"),
                 frac_known_truth=("known", "mean")).reset_index(),
             n_eval=int(len(inter)), n_missing=n_missing, n_known=int(known.sum()), n_novel=int(novel.sum()),
             BA=float(np.mean(list(rec.values()))) if rec else None,
             macroF1=float(np.mean(list(f1.values()))) if f1 else None,
             acc=float((pred[known] == y[known]).mean()) if known.any() else None,
             BA_accepted=float(np.mean([v for v in rec_acc.values() if v is not None])) if rec_acc else None,
             unknown_known=float(unk[known].mean()) if known.any() else None,
             unknown_novel=float(unk[novel].mean()) if novel.any() else None,
             auroc_novel_score=auroc(score, novel) if novel.any() else None,
             auroc_novel_call=auroc(unk.astype(float), novel) if novel.any() else None,
             auroc_error_score=auroc(score[known], pred[known] != y[known]) if known.any() else None,
             per_type_recall=rec, per_type_precision=prec, per_type_f1=f1)
    if novel.any():
        vc = pd.Series(pred[novel]).value_counts(normalize=True)
        r["novel_pred_top"] = {k: round(float(v), 4) for k, v in vc.head(5).items()}
    conds = pd.unique(cond)
    if len(conds) > 1:
        r["by_condition"] = {}
        for cnd in conds:
            mc = cond == cnd
            recs = [float((pred[mc & (y == c)] == c).mean()) for c in ev if (mc & (y == c)).sum() >= min_n]
            r["by_condition"][str(cnd)] = dict(
                n=int(mc.sum()), BA=float(np.mean(recs)) if recs else None,
                acc=float((pred[mc & known] == y[mc & known]).mean()) if (mc & known).any() else None,
                unknown_all=float(unk[mc].mean()), unknown_known=float(unk[mc & known].mean()) if (mc & known).any() else None,
                mean_score=float(np.mean(score[mc])))
    S = m.get("sets")
    if S is not None and m.get("classes") is not None:
        S = np.asarray(S, bool)[pos]
        scl = list(np.asarray(m["classes"]).astype(str))
        col = {c: i for i, c in enumerate(scl)}
        cov_t = {}
        for c in ev:
            mc = y == c
            cov_t[c] = float(S[mc, col[c]].mean()) if c in col else 0.0
        sz = S.sum(1)
        r["sets"] = dict(mean_cov=float(np.mean(list(cov_t.values()))) if cov_t else None,
                         min_cov=float(np.min(list(cov_t.values()))) if cov_t else None,
                         marginal_cov=float(np.mean([S[i, col[y[i]]] if y[i] in col else False
                                                     for i in np.where(known)[0]])) if known.any() else None,
                         frac_types_cov085=float(np.mean(np.array(list(cov_t.values())) >= 0.85)) if cov_t else None,
                         per_type_cov=cov_t, size_known=float(sz[known].mean()) if known.any() else None,
                         singleton_known=float((sz[known] == 1).mean()) if known.any() else None,
                         empty_known=float((sz[known] == 0).mean()) if known.any() else None,
                         empty_novel=float((sz[novel] == 0).mean()) if novel.any() else None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("seed", type=int)
    ap.add_argument("--min_n", type=int, default=10)
    a = ap.parse_args()
    truth, classes, ho, src = load_truth(a.tag, a.seed)
    meths, errors = collect(a.tag, a.seed)
    if not meths:
        print(f"[eval] no result files for {a.tag} seed {a.seed}; errors: {errors}", flush=True)
    summ = dict(tag=a.tag, seed=a.seed, holdout=ho, truth_source=src, n_query=int(len(truth)),
                n_classes=int(len(classes)), classes=classes.tolist(),
                n_novel=int((~np.isin(truth.y.values, classes)).sum()), errors=errors, methods={})
    rows, donor_rows = [], []
    for name, m in sorted(meths.items()):
        try:
            r = evaluate(m, truth, classes, a.min_n)
        except Exception as e:  # noqa
            summ["errors"][name] = f"evaluation failed: {e!r}"
            continue
        don = r.pop("_donor")
        don.insert(0, "method", name)
        donor_rows.append(don)
        meta = m.get("meta", {})
        r["runtime_s"] = meta.get("runtime_s")
        r["ram_peak_gb"] = meta.get("ram_peak_gb")
        r["gpu_peak_gb"] = meta.get("gpu_peak_gb")
        summ["methods"][name] = r
        for c, v in r["per_type_recall"].items():
            rows.append(dict(tag=a.tag, seed=a.seed, method=name, type=c, n_query=int((truth.y == c).sum()),
                             recall=v, precision=r["per_type_precision"][c], f1=r["per_type_f1"][c],
                             set_cov=r.get("sets", {}).get("per_type_cov", {}).get(c)))
        print(f"{name:40s} BA={r['BA'] if r['BA'] is None else round(r['BA'], 3)} "
              f"F1={r['macroF1'] if r['macroF1'] is None else round(r['macroF1'], 3)} "
              f"unk_known={r['unknown_known'] if r['unknown_known'] is None else round(r['unknown_known'], 3)} "
              f"AUROC_novel={r['auroc_novel_score']}", flush=True)
    os.makedirs("res/comp", exist_ok=True)
    out = f"res/comp/summary_{a.tag}_seed{a.seed}.json"
    json.dump(summ, open(out, "w"), indent=1, default=float)
    pd.DataFrame(rows).to_csv(f"res/comp/summary_{a.tag}_seed{a.seed}_types.csv", index=False)
    if donor_rows:
        pd.concat(donor_rows).to_csv(f"res/comp/summary_{a.tag}_seed{a.seed}_donor.csv", index=False)
    print("[eval] wrote", out, "methods:", len(summ["methods"]), "errors:", list(errors), flush=True)


if __name__ == "__main__":
    main()
