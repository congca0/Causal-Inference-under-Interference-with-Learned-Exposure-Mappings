"""
california_reanalysis.py — empirical illustration for the revised manuscript.

Run locally on the county-day panel used in the original analysis:
    python california_reanalysis.py merged_data.csv county_coordinates.csv
(epa_pm25_transport.csv or final_model_data.csv also work, but merged_data.csv is
preferred because it has not been filtered on lagged values.)

Expected columns (names are cleaned to lower_snake_case and common aliases accepted)
    panel file             : county_name, date_local, pm25, wind_speed, temperature
    county_coordinates.csv : county_name, latitude, longitude

What it does (all distances in km via the haversine formula, which replaces
Euclidean distance on raw latitude/longitude degrees):
  1. Builds a county-day panel with two within-county PM2.5 lags.
  2. Defines candidate transport specifications for the source-receptor
     weights W (zero diagonal): isotropic Gaussian, isotropic exponential, and
     a directional Gaussian that up-weights sources lying upwind of the
     receptor along a fitted prevailing direction (no wind-direction data are
     available, so the direction is a fitted parameter).
  3. Selects each specification's range (and direction) by blocked
     cross-validation of a PM2.5 prediction model with the transport term
     sum_j W_ij PM_{j,t-1} (lagged to avoid simultaneity), county fixed
     effects, two PM lags, wind speed and temperature.
  4. Reports predictive performance (day-blocked and county-blocked CV RMSE).
  5. For hypothetical county-level emission-control policies, computes the
     design-stage diagnostics of Section 3.4 with Z = [1, A] (own treatment):
     the transport sensitivity matrix S_mk, relative residual distances,
     Spearman correlation of exposure across counties, and top-10 overlap.
No health outcome is used; the diagnostics require only geography, the
policy and the fitted transport specifications.

Outputs: ca_out/prediction.csv, ca_out/diagnostics_<policy>.csv,
         ca_out/S_<policy>.csv, ca_out/fig_ca_S.png, ca_out/summary.txt
"""
import sys, os
import numpy as np
import pandas as pd
from itertools import product

EARTH_R = 6371.0
POLICIES = {
    "South Coast": ["Los Angeles", "Orange", "Riverside", "San Bernardino"],
    "San Joaquin Valley": ["Fresno", "Kern", "Kings", "Madera", "Merced", "San Joaquin", "Stanislaus", "Tulare"],
    "Bay Area": ["Alameda", "Contra Costa", "San Francisco", "San Mateo", "Santa Clara"],
    "Single county (Kern)": ["Kern"],
}


ALIASES = {
    "county_name": ["county_name", "county"],
    "date_local": ["date_local", "date"],
    "pm25": ["pm25", "pm2_5", "pm_25", "arithmetic_mean", "pm25_mean"],
    "wind_speed": ["wind_speed", "wind", "ws", "wind_speed_mean"],
    "temperature": ["temperature", "temp", "temp_mean"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "long", "lng"],
}


def clean_names(df):
    import re
    df = df.copy()
    df.columns = [re.sub(r"[^0-9a-z]+", "_", c.strip().lower()).strip("_") for c in df.columns]
    out = {}
    for std, al in ALIASES.items():
        for a in al:
            if a in df.columns:
                out[a] = std
                break
    return df.rename(columns=out)


def require(df, cols, fname):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f"{fname} is missing columns {miss}. Columns found: {list(df.columns)}")


def haversine_matrix(lat, lon):
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    dlat, dlon = la - la.T, lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_matrix(lat, lon):
    """Bearing (radians, math convention: 0 = east, pi/2 = north) from j to i."""
    la, lo = np.radians(lat), np.radians(lon)
    dx = (lo[:, None] - lo[None, :]) * np.cos((la[:, None] + la[None, :]) / 2)
    dy = la[:, None] - la[None, :]
    return np.arctan2(dy, dx)


def weights(kind, D, B, par):
    if kind == "Gaussian":
        W = np.exp(-D**2 / (2 * par["range"] ** 2))
    elif kind == "Exponential":
        W = np.exp(-D / par["range"])
    elif kind == "Directional":
        # receptor i receives more from sources j located upwind, i.e. when the
        # bearing j -> i aligns with the prevailing transport direction theta
        W = np.exp(-D**2 / (2 * par["range"] ** 2)) * np.exp(par["kappa"] * np.cos(B - par["theta"]))
    else:
        raise ValueError(kind)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(1, keepdims=True)
    return W / np.where(rs > 0, rs, 1)


GRIDS = {
    "Gaussian": [dict(range=r) for r in [25, 50, 75, 100, 150, 200, 300]],
    "Exponential": [dict(range=r) for r in [25, 50, 75, 100, 150, 200, 300]],
    "Directional": [dict(range=r, theta=t, kappa=1.0)
                    for r, t in product([50, 100, 150, 200], np.linspace(-np.pi, np.pi, 8, endpoint=False))],
}


def design(panel, W, counties):
    """Transport term sum_j W_ij PM_{j,t-1} for each county-day row."""
    wide = panel.pivot_table(index="date", columns="county_name", values="pm25", aggfunc="mean")
    wide = wide.reindex(index=pd.date_range(wide.index.min(), wide.index.max(), freq="D"), columns=counties)
    lag = wide.shift(1)                                    # previous calendar day
    lag_f = lag.T.fillna(lag.mean(axis=1)).T             # fill missing neighbours by the day's mean
    T = pd.DataFrame(lag_f.values @ W.T, index=pd.Index(wide.index, name="date"),
                     columns=pd.Index(counties, name="county_name"))
    t = T.stack().rename("transport").reset_index()
    return panel.merge(t, on=["date", "county_name"], how="left")


def fe_ols(df, y, xs, fe="county_name"):
    """OLS with county fixed effects via within transformation."""
    d = df[[y, fe] + xs].dropna()
    g = d.groupby(fe)
    yc = d[y] - g[y].transform("mean")
    Xc = d[xs] - g[xs].transform("mean")
    beta, *_ = np.linalg.lstsq(Xc.values, yc.values, rcond=None)
    alpha = (d[y] - d[xs].values @ beta).groupby(d[fe]).mean()
    return beta, alpha


def predict(df, beta, alpha, xs, fe="county_name"):
    a = df[fe].map(alpha).fillna(alpha.mean())
    return a.values + df[xs].values @ beta


def cv_rmse(df, xs, block):
    keys = df[block].unique()
    rng = np.random.default_rng(1)
    folds = np.array_split(rng.permutation(keys), 5)
    err = []
    for f in folds:
        te = df[block].isin(f)
        b, a = fe_ols(df[~te], "pm25", xs)
        err.append((df.loc[te, "pm25"].values - predict(df[te], b, a, xs)) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(err))))


def main(pm_file, coord_file, out="ca_out"):
    os.makedirs(out, exist_ok=True)
    pm = clean_names(pd.read_csv(pm_file))
    co = clean_names(pd.read_csv(coord_file))
    require(pm, ["county_name", "date_local", "pm25", "wind_speed", "temperature"], pm_file)
    require(co, ["county_name", "latitude", "longitude"], coord_file)
    for c in ["pm25", "wind_speed", "temperature"]:
        pm[c] = pd.to_numeric(pm[c], errors="coerce")
    pm["date"] = pd.to_datetime(pm["date_local"])
    pm = (pm.groupby(["county_name", "date"], as_index=False)
            .agg(pm25=("pm25", "mean"), wind_speed=("wind_speed", "mean"), temperature=("temperature", "mean")))
    co = co.drop_duplicates("county_name")
    co = co[co.county_name.isin(pm.county_name.unique())].sort_values("county_name").reset_index(drop=True)
    pm = pm[pm.county_name.isin(co.county_name)]
    counties = co.county_name.tolist()
    D = haversine_matrix(co.latitude.values, co.longitude.values)
    B = bearing_matrix(co.latitude.values, co.longitude.values)
    # calendar-day lags within county (a missing day gives a missing lag)
    wide = pm.pivot_table(index="date", columns="county_name", values="pm25", aggfunc="mean")
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    for k in (1, 2):
        lk = wide.shift(k).stack().rename(f"pm25_lag{k}").reset_index().rename(columns={"level_0": "date"})
        lk.columns = ["date", "county_name", f"pm25_lag{k}"]
        pm = pm.merge(lk, on=["date", "county_name"], how="left")
    base_x = ["pm25_lag1", "pm25_lag2", "wind_speed", "temperature"]

    # ---- fit each specification by day-blocked CV
    rows, fitted = [], {}
    panel0 = pm.dropna(subset=base_x + ["pm25"])
    rows.append(dict(spec="No transport term", range_km=np.nan, theta_deg=np.nan,
                     cv_rmse_day=cv_rmse(panel0, base_x, "date"),
                     cv_rmse_county=cv_rmse(panel0, base_x, "county_name"), n=len(panel0),
                     n_counties=panel0.county_name.nunique()))
    for kind, grid in GRIDS.items():
        best = None
        for par in grid:
            W = weights(kind, D, B, par)
            d = design(pm, W, counties).dropna(subset=base_x + ["pm25", "transport"])
            r = cv_rmse(d, base_x + ["transport"], "date")
            if best is None or r < best[0]:
                best = (r, par, W, d)
        r, par, W, d = best
        fitted[kind] = W
        bta, _ = fe_ols(d, "pm25", base_x + ["transport"])
        rows.append(dict(spec=kind, range_km=par["range"],
                         theta_deg=np.degrees(par.get("theta", np.nan)),
                         cv_rmse_day=r, cv_rmse_county=cv_rmse(d, base_x + ["transport"], "county_name"),
                         transport_coef=bta[-1], n=len(d), n_counties=d.county_name.nunique()))
    pred = pd.DataFrame(rows)
    pred.to_csv(f"{out}/prediction.csv", index=False)

    # ---- design-stage diagnostics for hypothetical policies
    names = list(fitted)
    sel = pred[pred.spec.isin(names)].sort_values("cv_rmse_day").spec.iloc[0]
    lines = [pred.round(3).to_string(), "", f"Selected (lowest day-blocked CV RMSE): {sel}", ""]
    Smats, tsi, ntreat = {}, {}, {}
    for pol, members in POLICIES.items():
        A = np.isin(counties, members).astype(float)
        if A.sum() == 0:
            lines.append(f"{pol}: none of {members} in data; skipped"); continue
        Z = np.column_stack([np.ones(len(A)), A])
        Q, _ = np.linalg.qr(Z)
        Mf = lambda v: v - Q @ (Q.T @ v)
        E = {m: fitted[m] @ A for m in names}
        S = np.array([[(Mf(E[a]) @ Mf(E[b])) / (Mf(E[a]) @ Mf(E[a])) for b in names] for a in names])
        Dr = np.array([[np.linalg.norm(Mf(E[b]) - Mf(E[a])) / np.linalg.norm(Mf(E[a])) for b in names] for a in names])
        Smats[pol] = S
        col = S[:, names.index(sel)]
        tsi[pol] = float(np.log(col.max() / col.min())) if col.min() > 0 else np.nan
        ntreat[pol] = int(A.sum())
        pd.DataFrame(S, index=names, columns=names).to_csv(f"{out}/S_{pol.replace(' ', '_')}.csv")
        rk = pd.DataFrame(E).rank()
        sp = rk.corr()
        top = {m: set(np.argsort(-E[m])[:10]) for m in names}
        diag = []
        for a, b in product(names, names):
            if a < b:
                diag.append(dict(policy=pol, TSI=tsi[pol], model_a=a, model_b=b, S_ab=S[names.index(a), names.index(b)],
                                 S_ba=S[names.index(b), names.index(a)],
                                 rel_resid_dist=Dr[names.index(a), names.index(b)],
                                 spearman=sp.loc[a, b], top10_overlap=len(top[a] & top[b]) / 10))
        dd = pd.DataFrame(diag)
        dd.to_csv(f"{out}/diagnostics_{pol.replace(' ', '_')}.csv", index=False)
        lines += [f"== {pol} ({int(A.sum())} treated counties)", dd.round(3).to_string(), ""]
    open(f"{out}/summary.txt", "w").write("\n".join(lines))
    write_fill(out, pm, panel0, pred, sel, tsi, ntreat, names)
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = len(Smats)
        fig, axs = plt.subplots(1, k, figsize=(2.3 * k, 2.4))
        axs = np.atleast_1d(axs)
        for ax, (pol, S) in zip(axs, Smats.items()):
            ax.imshow(np.log(S), cmap="RdBu_r", vmin=-0.7, vmax=0.7)
            for a in range(len(names)):
                for b in range(len(names)):
                    ax.text(b, a, f"{S[a, b]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
            ax.set_title(pol, fontsize=8)
        fig.tight_layout(); fig.savefig(f"{out}/fig_ca_S.png", dpi=300)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


def write_fill(out, pm, panel0, pred, sel, tsi, ntreat, names):
    """Values to paste into Section 6 and Tables 4-5 of the manuscript."""
    f = lambda x, k=3: "—" if pd.isna(x) else f"{x:.{k}f}"
    L = ["# Values for Section 6", "",
         f"- Study period (raw panel): {pm.date.min():%B} {pm.date.min().day}, {pm.date.min().year} to {pm.date.max():%B} {pm.date.max().day}, {pm.date.max().year}",
         f"- County-days in the analysis panel (no-transport model): {len(panel0)}",
         f"- Counties in the analysis panel: {panel0.county_name.nunique()}",
         f"- Selected specification (lowest day-blocked CV RMSE): {sel}", "",
         "## Table 4", "",
         "| Specification | Selected parameters | Day-blocked CV RMSE | County-blocked CV RMSE | Transport coefficient |",
         "|:--|:--|--:|--:|--:|"]
    for _, r in pred.iterrows():
        if r.spec == "No transport term":
            par = "—"
        elif r.spec == "Directional":
            par = f"σ = {r.range_km:.0f} km, θ = {r.theta_deg:.0f}°"
        else:
            par = f"{'σ' if r.spec == 'Gaussian' else 'ℓ'} = {r.range_km:.0f} km"
        L.append(f"| {r.spec} | {par} | {f(r.cv_rmse_day)} | {f(r.cv_rmse_county)} | {f(r.get('transport_coef', np.nan))} |")
    L += ["", "## Table 5", "",
          "| Policy | Treated counties | TSI | Pair (m, k) | S_mk | S_km | D_mk | Spearman | Top-10 overlap |",
          "|:--|--:|--:|:--|--:|--:|--:|--:|--:|"]
    for pol in tsi:
        dd = pd.read_csv(f"{out}/diagnostics_{pol.replace(' ', '_')}.csv")
        for k, (_, r) in enumerate(dd.iterrows()):
            L.append(f"| {pol if k == 0 else ''} | {ntreat[pol] if k == 0 else ''} | {f(tsi[pol], 2) if k == 0 else ''} | "
                     f"{r.model_a}, {r.model_b} | {f(r.S_ab, 2)} | {f(r.S_ba, 2)} | {f(r.rel_resid_dist, 2)} | "
                     f"{f(r.spearman, 2)} | {f(r.top10_overlap, 1)} |")
    open(f"{out}/section6_fill.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
