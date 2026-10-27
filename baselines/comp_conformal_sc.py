"""(e) Reimplementation of the conformalized single-cell annotator of Lopez-De-Castro et al. 2025
(github.com/digital-medicine-research-group-UNAV/conformalized_single_cell_annotator;
"Conformal inference for reliable single cell RNA-seq annotation").

Usage: python comp_conformal_sc.py <ds> <seed> [holdout] [--features harmony|scvi] [--alpha 0.1]
       [--alpha_ood 0.1] [--min_cells 45]
Faithful core (from their code: conformalSC.py, annomaly_detector.py, conformalSC_annotator.py):
  * reference cell types with < min_cells (45) cells are dropped;
  * stratified reference split as with test=True: 10% test (in-distribution check / auto-alpha),
    then 10% of the rest validation (early stopping), then 45% of the rest calibration, rest
    training (approx. 44.6 / 9 / 36.5 / 10%);
  * classifier 'torch_net': feed-forward net (hidden [128, 64, 64], dropout 0.3, Adam) on
    features = 2000 HVGs (Seurat dispersion), log-normalized, PCA, Harmony across reference and
    query batches ("X_pca_harmony", as in their paper/README) -- or, with --features scvi, the scVI
    latent from train.py (no transductive query correction);
  * OOD detector: denoising sparse autoencoder (LayerNorm input, [128, 64, 32, 16] SiLU/LayerNorm/
    dropout blocks, Gaussian input noise 0.1, SmoothL1 reconstruction + 1e-3 L1 latent penalty, AdamW,
    early stopping patience 9) trained on the training split; nonconformity = mean squared
    reconstruction error; marginal conformal p-values from the calibration split; decision
    p <= alpha_eff with alpha_eff = alpha_ood / pi_hat, pi_hat = Storey estimate (lambda = 0.5) on the
    query p-values (as in their evaluate_marginal_pvalues with fixed alpha; default alpha_ood 0.1);
  * OOD cells get label 'unknown' (theirs: 'OOD') and an empty prediction set;
  * prediction sets: randomized APS score (README example), torchcp SplitPredictor ('standard',
    marginal) and ClassWisePredictor ('classwise'), alpha = 0.1, calibrated on the calibration split.
  Omitted: the random-projection contractive penalty of their AE, the 'cluster' predictor, RAPS.
Output: res/comp/ConformalSC-<features>_<tag>_seed<s>.npz
  pred = argmax label or 'unknown'; score = AE reconstruction error (monotone in 1 - p_OOD);
  sets = standard APS sets (empty for OOD cells), sets__classwise;
  variants: pred__noOOD; pred__autoalpha (their automatic alpha_OOD via KS test against the
  in-distribution test split); pred__knnOOD/score__knnOOD and pred__iforest/score__iforest
  (kNN-distance and isolation-forest nonconformity with the same conformal decision rule).
"""
import json
import numpy as np
import pandas as pd
from comp_common import (parse_args, run_main, load_std, load_emb, save_result, Timer, versions,
                         stratified_split, aps_scores, split_conformal, marginal_pvalues, nn_index,
                         query_part, UNKNOWN)


def extra(ap):
    ap.add_argument("--features", default="harmony", choices=["harmony", "scvi"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--alpha_ood", type=float, default=0.1)
    ap.add_argument("--min_cells", type=int, default=45)
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=100)


# ------------------------------------------------------------------------------------ features
def harmony_features(a, T):
    import scanpy as sc
    import harmonypy
    ref, qry = load_std(a.ds, a.holdout_label)
    for x in (ref, qry):
        sc.pp.normalize_total(x, target_sum=1e4)
        sc.pp.log1p(x)
    sc.pp.highly_variable_genes(ref, n_top_genes=a.n_hvg, flavor="seurat")
    hv = ref.var_names[ref.var.highly_variable]
    missing = [g for g in hv if g not in qry.var_names]
    if missing:
        raise ValueError(f"{len(missing)} HVGs absent in query")
    ref, qry = ref[:, hv].copy(), qry[:, hv].copy()
    from sklearn.decomposition import PCA

    def dense(X):
        return np.asarray(X.toarray() if hasattr(X, "toarray") else X, np.float32)
    Xr = dense(ref.X)
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-6
    Xr = np.clip((Xr - mu) / sd, -10, 10)
    pca = PCA(n_components=a.n_pcs, svd_solver="randomized", random_state=a.seed).fit(Xr)
    Zr = pca.transform(Xr).astype(np.float32)
    del Xr
    Zq = np.concatenate([pca.transform(np.clip((dense(qry.X[i:i + 50000]) - mu) / sd, -10, 10))
                         for i in range(0, qry.n_obs, 50000)]).astype(np.float32)
    T.mark("pca")
    Z = np.vstack([Zr, Zq])
    meta = pd.DataFrame({"batch": np.concatenate([ref.obs.batch.values, qry.obs.batch.values]).astype(str)})
    import inspect
    hkw = {"random_state": a.seed} if "random_state" in inspect.signature(harmonypy.run_harmony).parameters else {}
    ho = harmonypy.run_harmony(Z, meta, ["batch"], **hkw)
    Zc = np.asarray(ho.Z_corr)
    if hasattr(Zc, "cpu"):
        Zc = Zc.cpu().numpy()
    if Zc.shape[0] != Z.shape[0]:
        Zc = Zc.T
    if Zc.shape != Z.shape:
        raise ValueError(f"unexpected harmony output shape {Zc.shape}")
    T.mark("harmony")
    return (Zc[:len(Zr)].astype(np.float32), ref.obs.label.values.astype(str), Zc[len(Zr):].astype(np.float32),
            qry.obs_names.values.astype(str))


def scvi_features(a):
    F = load_emb(a.tag, a.seed, "full")
    q = query_part(F)
    return (F["ref_U"].astype(np.float32), np.asarray(F["y_ref"]).astype(str), F["tgt_U"][q].astype(np.float32),
            np.asarray(F["tgt_names"]).astype(str)[q])


# ------------------------------------------------------------------------------------ models
def make_mlp(d_in, n_out, hidden=(128, 64, 64), drop=0.3):
    import torch.nn as nn
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class AE:
    def __init__(self, d_in, hidden=(128, 64, 32, 16), drop=0.1):
        import torch.nn as nn
        enc, d = [nn.LayerNorm(d_in)], d_in
        for h in hidden:
            enc += [nn.utils.parametrizations.weight_norm(nn.Linear(d, h)) if hasattr(nn.utils, "parametrizations")
                    else nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(drop)]
            d = h
        dec = []
        for h in list(hidden[::-1][1:]):
            dec += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(drop)]
            d = h
        dec.append(nn.Linear(d, d_in))
        self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)


def fit_torch(model_params, loss_fn, Xtr, Ytr, Xva, Yva, epochs, patience, lr, bs, dev, seed, opt="adam"):
    import torch
    g = torch.Generator().manual_seed(seed)
    params = [p for m in model_params for p in m.parameters()]
    o = (torch.optim.AdamW if opt == "adamw" else torch.optim.Adam)(params, lr=lr, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    Xtr_t, Xva_t = torch.tensor(Xtr, device=dev), torch.tensor(Xva, device=dev)
    Ytr_t = torch.tensor(Ytr, device=dev) if Ytr is not None else None
    Yva_t = torch.tensor(Yva, device=dev) if Yva is not None else None
    for ep in range(epochs):
        for m in model_params:
            m.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(dev)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            o.zero_grad()
            l_ = loss_fn(Xtr_t[b], None if Ytr_t is None else Ytr_t[b], train=True)
            l_.backward()
            o.step()
        for m in model_params:
            m.eval()
        with torch.no_grad():
            vl = float(sum(loss_fn(Xva_t[i:i + 8192], None if Yva_t is None else Yva_t[i:i + 8192], train=False)
                           * len(Xva_t[i:i + 8192]) for i in range(0, len(Xva_t), 8192)) / len(Xva_t))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in model_params]
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("validation loss never finite: training diverged")
    for m, s in zip(model_params, best_state):
        m.load_state_dict(s)
    return ep + 1, best


def storey_alpha(p, alpha, lam=0.5):
    pi = (1.0 + np.sum(p > lam)) / (len(p) * (1.0 - lam))
    return alpha / pi, pi


def auto_alpha(p_test, p_id):
    """Their automatic alpha_OOD: first a in a 50-point grid on [0, 0.5) whose rescaled tails pass a KS test."""
    from scipy.stats import ks_2samp
    for a_ in np.linspace(0, 0.5, 50, endpoint=False):
        t, r = p_test[p_test >= a_], p_id[p_id >= a_]
        if len(t) < 10 or len(r) < 10:
            continue
        if ks_2samp((t - a_) / (1 - a_), (r - a_) / (1 - a_)).pvalue >= 0.025:
            return float(a_)
    return 0.1


# ------------------------------------------------------------------------------------ main
def main(a):
    import torch
    T = Timer()
    method = f"ConformalSC-{a.features}"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.features == "harmony":
        Zr, yr, Zq, names = harmony_features(a, T)
    else:
        Zr, yr, Zq, names = scvi_features(a)
    vc = pd.Series(yr).value_counts()
    keep_types = vc.index[vc >= a.min_cells]
    m = np.isin(yr, keep_types)
    Zr, yr = Zr[m], yr[m]
    classes = np.array(sorted(set(yr)))
    ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[c] for c in yr])
    te, va, ca, tr = stratified_split(yr, [0.10, 0.09, 0.3645, 0.4455], a.seed)
    mu, sd = Zr[tr].mean(0), Zr[tr].std(0) + 1e-6
    S = lambda Z: ((Z - mu) / sd).astype(np.float32)  # noqa
    Xr, Xq = S(Zr), S(Zq)
    T.mark("split")
    # ---------------- classifier
    import torch.nn.functional as Fn
    net = make_mlp(Xr.shape[1], len(classes)).to(dev)

    def ce(x, yy, train):
        return Fn.cross_entropy(net(x), yy)
    ep_c, _ = fit_torch([net], ce, Xr[tr], y[tr], Xr[va], y[va], a.epochs, 10, 1e-3, 256, dev, a.seed)

    def proba(X):
        net.eval()
        with torch.no_grad():
            return np.concatenate([torch.softmax(net(torch.tensor(X[i:i + 16384], device=dev)), 1).cpu().numpy()
                                   for i in range(0, len(X), 16384)]).astype(np.float64)
    Pcal, Pq, Pte = proba(Xr[ca]), proba(Xq), proba(Xr[te])
    T.mark("classifier")
    # ---------------- OOD autoencoder
    ae = AE(Xr.shape[1])
    ae.enc.to(dev), ae.dec.to(dev)

    def ae_loss(x, _, train):
        xin = x + 0.1 * torch.randn_like(x) if train else x
        z = ae.enc(xin)
        return Fn.smooth_l1_loss(ae.dec(z), x) + 1e-3 * z.abs().mean()
    ep_a, _ = fit_torch([ae.enc, ae.dec], ae_loss, Xr[tr], None, Xr[va], None, 150, 9, 1e-3, 64 if len(tr) < 20000 else 256,
                        dev, a.seed, opt="adamw")

    def recon(X):
        ae.enc.eval(), ae.dec.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 16384):
                x = torch.tensor(X[i:i + 16384], device=dev)
                out.append(((ae.dec(ae.enc(x)) - x) ** 2).mean(1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)
    s_cal = recon(Xr[ca]) + rng.uniform(-1e-6, 1e-6, len(ca))
    s_q, s_te = recon(Xq), recon(Xr[te])
    p_q, p_te = marginal_pvalues(s_cal, s_q), marginal_pvalues(s_cal, s_te)
    a_eff, pi = storey_alpha(p_q, a.alpha_ood)
    ood = p_q <= a_eff
    a_auto = auto_alpha(p_q, p_te)
    T.mark("ood_ae")
    # ---------------- alternative OOD scores (same decision rule)
    idx = nn_index(Xr[tr], 15)
    d_cal = idx.kneighbors(Xr[ca])[0].mean(1)
    d_q = idx.kneighbors(Xq)[0].mean(1)
    p_knn = marginal_pvalues(d_cal, d_q)
    ood_knn = p_knn <= storey_alpha(p_knn, a.alpha_ood)[0]
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=200, random_state=a.seed, n_jobs=-1).fit(Xr[tr])
    i_cal, i_q = -iso.score_samples(Xr[ca]), -iso.score_samples(Xq)
    p_iso = marginal_pvalues(i_cal, i_q)
    ood_iso = p_iso <= storey_alpha(p_iso, a.alpha_ood)[0]
    T.mark("ood_alt")
    # ---------------- conformal prediction sets (APS, randomized)
    ycal = y[ca]
    Scal, Sq, Ste = aps_scores(Pcal, rng), aps_scores(Pq, rng), aps_scores(Pte, rng)
    sets_std, _, q_std = split_conformal(Scal, ycal, Sq, a.alpha, len(classes), "standard")
    sets_cw, _, q_cw = split_conformal(Scal, ycal, Sq, a.alpha, len(classes), "classwise")
    te_std = split_conformal(Scal, ycal, Ste, a.alpha, len(classes), "standard")[0]
    te_cw = split_conformal(Scal, ycal, Ste, a.alpha, len(classes), "classwise")[0]
    yte = y[te]
    internal = dict(test_acc=float((Pte.argmax(1) == yte).mean()),
                    test_cov_standard=float(te_std[np.arange(len(te)), yte].mean()),
                    test_cov_classwise_min=float(min(te_cw[yte == t, t].mean() for t in range(len(classes))
                                                     if (yte == t).any())),
                    test_ood_rate=float((p_te <= a_eff).mean()))
    sets_std[ood] = False
    sets_cw[ood] = False
    base = classes[Pq.argmax(1)]

    def lab(mask):
        o = base.astype(object).copy()
        o[mask] = UNKNOWN
        return o.astype(str)
    variants = {"sets__classwise": sets_cw, "pred__noOOD": base, "score__noOOD": 1 - Pq.max(1),
                "pred__autoalpha": lab(p_q <= a_auto), "pred__knnOOD": lab(ood_knn), "score__knnOOD": d_q,
                "pred__iforest": lab(ood_iso), "score__iforest": i_q}
    meta = dict(features=a.features, alpha=a.alpha, alpha_ood=a.alpha_ood, alpha_eff=float(a_eff), storey_pi=float(pi),
                alpha_auto=a_auto, min_cells=a.min_cells, dropped_types=sorted(set(vc.index) - set(keep_types)),
                split_sizes=dict(test=len(te), val=len(va), cal=len(ca), train=len(tr)),
                classifier_epochs=ep_c, ae_epochs=ep_a, q_standard=float(q_std[0]),
                internal_test=internal, device=dev,
                versions=versions(["torch", "sklearn", "scanpy", "harmonypy", "numpy"]))
    print("[ConformalSC]", json.dumps(internal), flush=True)
    save_result(method, a.tag, a.seed, names, lab(ood), s_q, sets=sets_std, classes=classes, variants=variants,
                meta=meta, timer=T)


if __name__ == "__main__":
    args = parse_args(__doc__, extra)
    run_main(f"ConformalSC-{args.features}", args, main)
