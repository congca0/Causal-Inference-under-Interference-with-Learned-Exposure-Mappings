
run_mc.py — Monte Carlo driver.

For each transport-data replicate r (new winds, daily emissions, monitor
noise) and each scenario (A: AD-PDE class correct; B: no candidate correct):
  1. generate truth and monitor data;
  2. fit the four candidate transport models on 60 training days at 30
     training monitors; evaluate prediction on 20 held-out days and on 10
     held-out monitors;
  3. compute each model's exposure mapping for every policy, plus S
     approximate-posterior draws (within-model uncertainty propagation);
  4. draw R_H outcome data sets per policy (common across models) and
     estimate beta by OLS on [1, X, e_m] under each model and each draw;
  5. store design-stage diagnostics (computed without outcomes).

Usage:  python run_mc.py START STOP      (replicate indices, saves results/rep_XXX.pkl)
 
import sys, os, pickle, time
import numpy as np
from transport_sim import (make_geography, draw_days, truth, observe, PDEModel,
                           KernelModel, SpectralOperator, residualizer, ols_beta,
                           rho, rel_resid_err, BETA, BETA_X, SIGMA_H)

R_H = 400      # outcome replicates per transport replicate
S_DRAWS = 20   # within-model posterior draws
TRAIN_DAYS = np.arange(60)
TEST_DAYS = np.arange(60, 80)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def one_replicate(r, geo):
    out = {}
    rng = np.random.default_rng(10_000 + r)
    days = draw_days(rng, geo)
    Z = np.column_stack([np.ones(len(geo["rec"])), geo["Xcov"]])
    Mf = residualizer(Z)
    mt, ms = geo["mon_train"], geo["mon_test"]
    for sc in ["A", "B"]:
        base, e_star = truth(geo, days, sc)
        y = observe(rng, geo, base)
        res = {"e_star_mean": {p: float(e.mean()) for p, e in e_star.items()},
               "oracle_norm": {p: float(np.linalg.norm(Mf(e))) for p, e in e_star.items()},
               "pred": {}, "expo_mean": {}, "rho_true": {}, "fit": {}}
        models = [PDEModel(True), PDEModel(False), KernelModel(), SpectralOperator()]
        expo, draws = {}, {}
        for M in models:
            M.fit(geo, days, y, TRAIN_DAYS)
            e, pred = M.exposures(geo, days)
            expo[M.name] = e
            draws[M.name] = M.draws(geo, days, y, TRAIN_DAYS, S_DRAWS, rng)
            yt, yh_t = y[np.ix_(TEST_DAYS, mt)], pred[np.ix_(TEST_DAYS, mt)]
            ys, yh_s = y[:, ms], pred[:, ms]
            res["pred"][M.name] = dict(
                train=rmse(y[np.ix_(TRAIN_DAYS, mt)], pred[np.ix_(TRAIN_DAYS, mt)]),
                temporal=rmse(yt, yh_t), spatial=rmse(ys, yh_s),
                r2_spatial=1 - np.mean((ys - yh_s) ** 2) / np.var(ys),
                corr2_spatial=np.corrcoef(ys.ravel(), yh_s.ravel())[0, 1] ** 2)
            res["fit"][M.name] = {k: np.asarray(getattr(M, k)).tolist()
                                  for k in ["phi", "gamma", "lam", "edf"] if hasattr(M, k)}
            res["expo_mean"][M.name] = {p: float(v.mean()) for p, v in e.items()}
            res["rho_true"][M.name] = {p: float(rho(Mf, e[p], e_star[p])) for p in e}
        names = [M.name for M in models]
        # design-stage diagnostics among fitted candidates (no outcome data used)
        res["S"], res["Drel"], res["norm"] = {}, {}, {}
        for p in geo["policies"]:
            res["S"][p] = np.array([[rho(Mf, expo[a][p], expo[b][p]) for b in names] for a in names])
            res["Drel"][p] = np.array([[rel_resid_err(Mf, expo[a][p], expo[b][p]) for b in names] for a in names])
            res["norm"][p] = {a: float(np.linalg.norm(Mf(expo[a][p]))) for a in names}
        # outcome analyses
        res["est"] = {}
        for p in geo["policies"]:
            Hm = (BETA * e_star[p] + BETA_X * geo["Xcov"])[None] + SIGMA_H * rng.standard_normal((R_H, len(geo["rec"])))
            res["est"][p] = {}
            b_or, se_or = ols_beta(Mf, e_star[p], Hm)
            res["est"][p]["Oracle"] = dict(b=b_or.astype(np.float32), se=se_or.astype(np.float32))
            for a in names:
                b, se = ols_beta(Mf, expo[a][p], Hm)
                bd = np.empty((R_H, S_DRAWS)); sd = np.empty((R_H, S_DRAWS))
                for s, ed in enumerate(draws[a]):
                    bd[:, s], sd[:, s] = ols_beta(Mf, ed[p], Hm)
                bbar = bd.mean(1)
                Tvar = (sd ** 2).mean(1) + (1 + 1 / S_DRAWS) * bd.var(1, ddof=1)
                res["est"][p][a] = dict(b=b.astype(np.float32), se=se.astype(np.float32),
                                        bprop=bbar.astype(np.float32),
                                        seprop=np.sqrt(Tvar).astype(np.float32),
                                        tau=(b * expo[a][p].mean()).astype(np.float32))
        res["names"] = names
        out[sc] = res
    return out


if __name__ == "__main__":
    start, stop = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs("results", exist_ok=True)
    geo = make_geography()
    for r in range(start, stop):
        fn = f"results/rep_{r:03d}.pkl"
        if os.path.exists(fn):
            continue
        t0 = time.time()
        o = one_replicate(r, geo)
        with open(fn, "wb") as f:
            pickle.dump(o, f)
        print(f"rep {r} done in {time.time() - t0:.1f}s", flush=True)
