"""Simulation check of the excess-flag estimator of the out-of-reference fraction (no data needed).

Control (in-reference) cell scores ~ N(0, 1); novel cell scores ~ N(delta, 1). Each of n donors has m cells, a
fraction pi of them novel (binomial). Conformal p-values against n_cal control calibration cells (fresh per
simulation), flag p <= alpha. With F_bar the mean per-donor flag rate,

    E[F_bar] = alpha_eff (1 - pi) + pi * power,   alpha_eff = floor(alpha (n_cal + 1)) / (n_cal + 1) (<= alpha)
    est = (F_bar - alpha) / (1 - alpha)  =>  E[est] = pi (power - alpha) / (1 - alpha)   (taking alpha_eff = alpha)
    bias = E[est] - pi = -pi (1 - power) / (1 - alpha)   (~ -pi (1 - power) for small alpha)

so est is unbiased when novel cells are always flagged (power = 1) and a lower bound on pi otherwise.

Usage: python analysis/excess_sim.py [--sims 500 --n_donors 20 --cells 500 --n_cal 2000]
Outputs: res3/excess_sim/{table.csv, summary.json}; KEY lines.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument("--deltas", nargs="+", type=float, default=[1, 2, 3, 5])
ap.add_argument("--pis", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.3])
ap.add_argument("--n_donors", nargs="+", type=int, default=[20])
ap.add_argument("--cells", type=int, default=500)
ap.add_argument("--n_cal", type=int, default=2000)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--sims", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="res3/excess_sim")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
al = A.alpha
al_eff = np.floor(al * (A.n_cal + 1)) / (A.n_cal + 1)
rows = []
for nd in A.n_donors:
    for delta in A.deltas:
        for pi in A.pis:
            rng = np.random.default_rng([A.seed, nd, int(delta * 100), int(pi * 1000)])
            Fbar, pw, pi_hat = np.empty(A.sims), np.empty(A.sims), np.empty(A.sims)
            for s in range(A.sims):
                cal = np.sort(rng.standard_normal(A.n_cal))  # fresh calibration set
                nov = rng.random((nd, A.cells)) < pi
                x = rng.standard_normal((nd, A.cells)) + delta * nov
                p = (1.0 + A.n_cal - np.searchsorted(cal, x, side="left")) / (A.n_cal + 1.0)
                fl = p <= al
                Fbar[s] = fl.mean(1).mean()
                pw[s] = fl[nov].mean() if nov.any() else np.nan
                pi_hat[s] = nov.mean(1).mean()
            est = (Fbar - al) / (1 - al)
            power = float(np.nanmean(pw))
            power_theory = float(1 - norm.cdf(norm.ppf(1 - al) - delta))
            rows.append(dict(n_donors=nd, cells=A.cells, n_cal=A.n_cal, delta=delta, pi=pi,
                             power=power, power_theory=power_theory,
                             F_mean=float(Fbar.mean()), F_identity=float(al_eff * (1 - pi) + pi * power),
                             est_mean=float(est.mean()), est_sd=float(est.std(ddof=1)),
                             est_identity=float((al_eff * (1 - pi) + pi * power - al) / (1 - al)),
                             est_identity_simple=float(pi * (power - al) / (1 - al)),
                             bias=float(est.mean() - pi),
                             bias_identity=float((al_eff * (1 - pi) + pi * power - al) / (1 - al) - pi),
                             bias_simple=float(-pi * (1 - power) / (1 - al)),
                             bias_approx=float(-pi * (1 - power)), ratio=float(est.mean() / pi),
                             mc_se_est=float(est.std(ddof=1) / np.sqrt(A.sims)),
                             coverage_lower_bound=float(np.mean(est <= pi_hat))))
df = pd.DataFrame(rows)
df.to_csv(f"{A.out}/table.csv", index=False)
summ = dict(settings=vars(A), alpha_eff=float(al_eff),
            identity="E[F] = alpha(1-pi) + pi*power; (F-alpha)/(1-alpha) has mean pi*(power-alpha)/(1-alpha); "
                     "bias = -pi(1-power)/(1-alpha)",
            max_abs_dev_est_vs_identity=float((df.est_mean - df.est_identity).abs().max()),
            max_abs_dev_F_vs_identity=float((df.F_mean - df.F_identity).abs().max()),
            max_abs_bias_when_power_ge_0999=float(df.loc[df.power >= 0.999, "bias"].abs().max())
            if (df.power >= 0.999).any() else None,
            all_mean_est_le_pi=bool((df.est_mean <= df.pi + 3 * df.mc_se_est).all()),
            rows=rows)
json.dump(summ, open(f"{A.out}/summary.json", "w"), indent=1)
for r in rows:
    print(f"KEY excess_sim D={r['n_donors']} delta={r['delta']:g} pi={r['pi']:.2f}: power {r['power']:.3f} | "
          f"est {r['est_mean']:.4f}±{r['est_sd']:.4f} (identity {r['est_identity']:.4f}) | bias {r['bias']:+.4f} "
          f"(theory {r['bias_identity']:+.4f}) | ratio {r['ratio']:.3f}")
print(f"KEY excess_sim max|est - identity| = {summ['max_abs_dev_est_vs_identity']:.2e}; "
      f"max|F - identity| = {summ['max_abs_dev_F_vs_identity']:.2e}; "
      f"max|bias| at power>=0.999 = {summ['max_abs_bias_when_power_ge_0999']}")
