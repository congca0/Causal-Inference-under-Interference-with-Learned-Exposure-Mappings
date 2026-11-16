"""
analyze.py — summarize Monte Carlo results into tables (CSV) and figures (PNG).
Usage: python analyze.py   (reads results/rep_*.pkl, writes out/)
"""
import glob, os, pickle, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from transport_sim import BETA

Z975 = 1.959964
SCREEN_TOL = 0.15      # screened set: temporal-holdout RMSE within 15% of the best
DELTAS = [0.1, 0.2]
OUT = "out"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
COL = {"Oracle": "#000000", "AD-PDE": "#0072B2", "D-PDE": "#D55E00", "Kernel": "#E69F00", "LSO": "#009E73"}  # Okabe-Ito (colour-blind safe)
MARK = {"Regional": "o", "Point source": "s", "Upwind": "^", "Downwind": "v"}

reps = [pickle.load(open(f, "rb")) for f in sorted(glob.glob("results/rep_*.pkl"))]
R = len(reps)
names = reps[0]["A"]["names"]
policies = list(reps[0]["A"]["est"].keys())
print("replicates:", R)


def cover(lo, hi):
    return ((lo <= BETA) & (BETA <= hi)).mean()


# ---------------------------------------------------------------- Table 1
rows = []
for sc in "AB":
    for m in names:
        P = pd.DataFrame([rep[sc]["pred"][m] for rep in reps])
        rows.append(dict(scenario=sc, model=m, **{k: P[k].mean() for k in P}))
t1 = pd.DataFrame(rows)
t1.to_csv(f"{OUT}/table1_prediction.csv", index=False)

# ---------------------------------------------------------------- Table 2
rows = []
for sc in "AB":
    for p in policies:
        for m in ["Oracle"] + names:
            b = np.concatenate([rep[sc]["est"][p][m]["b"] for rep in reps]).astype(float)
            se = np.concatenate([rep[sc]["est"][p][m]["se"] for rep in reps]).astype(float)
            d = dict(scenario=sc, policy=p, model=m, mean=b.mean(), bias=b.mean() - BETA,
                     sd=b.std(), rmse=np.sqrt(np.mean((b - BETA) ** 2)), mean_se=se.mean(),
                     cov_plugin=cover(b - Z975 * se, b + Z975 * se))
            if m != "Oracle":
                bp = np.concatenate([rep[sc]["est"][p][m]["bprop"] for rep in reps]).astype(float)
                sp_ = np.concatenate([rep[sc]["est"][p][m]["seprop"] for rep in reps]).astype(float)
                d["cov_prop"] = cover(bp - Z975 * sp_, bp + Z975 * sp_)
                d["mean_se_prop"] = sp_.mean()
                d["prop1_pred"] = BETA * np.mean([rep[sc]["rho_true"][m][p] for rep in reps])
                # policy effect: tau_hat / tau per replicate
                rt = [rep[sc]["est"][p][m]["tau"].astype(float).mean() / (BETA * rep[sc]["e_star_mean"][p]) for rep in reps]
                d["tau_relbias"] = np.mean(rt) - 1
                d["prop2_pred"] = np.mean([rep[sc]["rho_true"][m][p] * rep[sc]["expo_mean"][m][p] / rep[sc]["e_star_mean"][p] for rep in reps]) - 1
                d["beta_relbias"] = b.mean() / BETA - 1
            rows.append(d)
t2 = pd.DataFrame(rows)
t2.to_csv(f"{OUT}/table2_estimation.csv", index=False)

# ---------------------------------------------------------------- Table 3: interval methods
rows = []
sel_freq = {}
screen_freq = {}
for sc in "AB":
    for p in policies:
        acc = {k: [] for k in ["Selected: plug-in", "Selected: within-model", "Equal-weight ensemble",
                               "Union (all candidates)", "Union (screened)"] + [f"Selected: delta={d}" for d in DELTAS]}
        sel_list, scr_count = [], np.zeros(len(names))
        for rep in reps:
            r = rep[sc]
            spat = np.array([r["pred"][m]["spatial"] for m in names])
            temp = np.array([r["pred"][m]["temporal"] for m in names])
            s = int(np.argmin(spat)); sel_list.append(names[s])
            scr = temp <= (1 + SCREEN_TOL) * temp.min(); scr_count += scr
            B = np.column_stack([r["est"][p][m]["b"] for m in names]).astype(float)
            SE = np.column_stack([r["est"][p][m]["se"] for m in names]).astype(float)
            BP = r["est"][p][names[s]]["bprop"].astype(float); SP = r["est"][p][names[s]]["seprop"].astype(float)
            lo, hi = B - Z975 * SE, B + Z975 * SE
            acc["Selected: plug-in"].append((lo[:, s], hi[:, s]))
            acc["Selected: within-model"].append((BP - Z975 * SP, BP + Z975 * SP))
            M = len(names)
            bbar = B.mean(1); T = (SE ** 2).mean(1) + (1 + 1 / M) * B.var(1, ddof=1)
            acc["Equal-weight ensemble"].append((bbar - Z975 * np.sqrt(T), bbar + Z975 * np.sqrt(T)))
            acc["Union (all candidates)"].append((lo.min(1), hi.max(1)))
            acc["Union (screened)"].append((lo[:, scr].min(1), hi[:, scr].max(1)))
            D = r["Drel"][p][s]                     # relative residual distance from selected model
            dhat = D.max()
            for dlt in DELTAS:
                L, U = lo[:, s], hi[:, s]
                if dlt < 1:
                    lo_d = np.minimum(L / (1 + dlt), L / (1 - dlt)); hi_d = np.maximum(U / (1 + dlt), U / (1 - dlt))
                acc[f"Selected: delta={dlt}"].append((lo_d, hi_d))
        sel_freq[(sc, p)] = pd.Series(sel_list).value_counts(normalize=True).to_dict()
        screen_freq[(sc, p)] = dict(zip(names, scr_count / R))
        for k, v in acc.items():
            lo = np.concatenate([a[0] for a in v]); hi = np.concatenate([a[1] for a in v])
            rows.append(dict(scenario=sc, policy=p, method=k, coverage=cover(lo, hi),
                             median_width=np.median(hi - lo)))
t3 = pd.DataFrame(rows)
t3.to_csv(f"{OUT}/table3_intervals.csv", index=False)
json.dump({f"{k[0]}|{k[1]}": v for k, v in sel_freq.items()}, open(f"{OUT}/selection_freq.json", "w"), indent=1)
json.dump({f"{k[0]}|{k[1]}": v for k, v in screen_freq.items()}, open(f"{OUT}/screen_freq.json", "w"), indent=1)

# ---------------------------------------------------------------- design-stage diagnostic
rows = []
for sc in "AB":
    for p in policies:
        for rep in reps:
            r = rep[sc]
            S = r["S"][p]                              # S[a, b] = rho(e_a, e_b): analysis a, assumed truth b
            spat = np.array([r["pred"][m]["spatial"] for m in names]); sidx = int(np.argmin(spat))
            col = S[:, sidx]
            tsi = np.log(col.max() / col.min())        # predicted cross-model spread if selected model were true
            bm = np.array([r["est"][p][m]["b"].astype(float).mean() for m in names])
            realized = np.log(bm.max() / bm.min())
            b1 = np.array([float(r["est"][p][m]["b"][0]) for m in names])
            realized1 = np.log(b1.max() / b1.min()) if b1.min() > 0 else np.nan
            rows.append(dict(scenario=sc, policy=p, TSI=tsi, realized=realized, realized_single=realized1,
                             S_AD=S[0].tolist()))
t4 = pd.DataFrame(rows)
t4.to_csv(f"{OUT}/table4_diagnostic.csv", index=False)
Smean = {(sc, p): np.mean([rep[sc]["S"][p] for rep in reps], 0) for sc in "AB" for p in policies}
np.save(f"{OUT}/S_mean.npy", {f"{k[0]}|{k[1]}": v for k, v in Smean.items()}, allow_pickle=True)

# ================================================================= FIGURES
# Fig 3: Proposition 1 validation
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
d = t2[t2.model != "Oracle"]
ax[0].plot([1, 4.5], [1, 4.5], color="grey", lw=0.8, ls="--")
for _, rr in d.iterrows():
    ax[0].scatter(rr.prop1_pred, rr["mean"], color=COL[rr.model], marker=MARK[rr.policy],
                  facecolor="none" if rr.scenario == "B" else COL[rr.model], s=28, lw=1)
ax[0].set_xlabel(r"Proposition 1 prediction  $\beta\,\bar\rho_m$")
ax[0].set_ylabel(r"Monte Carlo mean of $\hat\beta_m$")
ax[0].set_title("(a) Spillover coefficient", loc="left")
lim = [-0.75, 0.35]
ax[1].plot(lim, lim, color="grey", lw=0.8, ls="--")
for _, rr in d.iterrows():
    ax[1].scatter(rr.prop2_pred, rr.tau_relbias, color=COL[rr.model], marker=MARK[rr.policy],
                  facecolor="none" if rr.scenario == "B" else COL[rr.model], s=28, lw=1)
ax[1].set_xlabel(r"Proposition 2 prediction  $\rho_m\kappa_m-1$")
ax[1].set_ylabel(r"Monte Carlo relative bias of $\hat\tau_m$")
ax[1].set_title("(b) Policy effect", loc="left")
from matplotlib.lines import Line2D
h = [Line2D([], [], color=COL[m], marker="o", ls="", label=m) for m in names]
h += [Line2D([], [], color="grey", marker=MARK[p], ls="", label=p) for p in policies]
h += [Line2D([], [], color="grey", marker="o", ls="", label="Scenario A"),
      Line2D([], [], color="grey", marker="o", mfc="none", ls="", label="Scenario B")]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_prop_validation.png", bbox_inches="tight", dpi=600); fig.savefig(f"{OUT}/fig3_prop_validation.pdf", bbox_inches="tight")
plt.close(fig)

# Fig 2: prediction vs inference
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for k, sc in enumerate("AB"):
    ax = axs[k]
    for m in names:
        x = t1[(t1.scenario == sc) & (t1.model == m)]["spatial"].values[0]
        for p in policies:
            y = t2[(t2.scenario == sc) & (t2.model == m) & (t2.policy == p)]["mean"].values[0]
            ax.scatter(x, y, color=COL[m], marker=MARK[p], s=34)
    ax.axhline(BETA, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Held-out-monitor RMSE of PM prediction")
    ax.set_title(f"Scenario {sc}" + (": AD-PDE class correct" if sc == "A" else ": no candidate correct"), loc="left")
axs[0].set_ylabel(r"Monte Carlo mean of $\hat\beta_m$ (truth = 2)")
fig.legend(handles=h[:len(names) + len(policies)], loc="lower center", ncol=8, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_prediction_vs_inference.png", bbox_inches="tight", dpi=600); fig.savefig(f"{OUT}/fig2_prediction_vs_inference.pdf", bbox_inches="tight")
plt.close(fig)

# Fig 4: interval coverage / width
meths = ["Selected: plug-in", "Selected: within-model", "Selected: delta=0.1", "Selected: delta=0.2",
         "Equal-weight ensemble", "Union (screened)", "Union (all candidates)"]
fig, axs = plt.subplots(1, 4, figsize=(8.0, 3.4), sharey=True, gridspec_kw=dict(wspace=0.12))
for k, sc in enumerate("AB"):
    for j, p in enumerate(policies):
        dd = t3[(t3.scenario == sc) & (t3.policy == p)].set_index("method").loc[meths]
        yy = np.arange(len(meths)) + (j - 1.5) * 0.15
        axs[2 * k].scatter(dd.coverage, yy, marker=MARK[p], color="#333333", s=18, label=p if k == 0 else None)
        axs[2 * k + 1].scatter(dd.median_width, yy, marker=MARK[p], color="#0072B2", s=18)
    axs[2 * k].axvline(0.95, color="grey", lw=0.8, ls="--")
    axs[2 * k].set_xlim(-0.03, 1.03); axs[2 * k].set_xlabel("Coverage")
    axs[2 * k + 1].set_xscale("log"); axs[2 * k + 1].set_xlabel("Median width")
    axs[2 * k + 1].set_xticks([0.2, 0.5, 1, 2, 5]); axs[2 * k + 1].set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    axs[2 * k + 1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); axs[2 * k + 1].set_xlim(0.1, 7)
    axs[2 * k].set_title(f"Scenario {sc}: coverage", loc="left", fontsize=8.5)
    axs[2 * k + 1].set_title("width", loc="left", fontsize=8.5)
axs[0].set_yticks(np.arange(len(meths))); axs[0].set_yticklabels(meths)
axs[0].invert_yaxis()
fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, -0.12))
fig.savefig(f"{OUT}/fig4_coverage.png", bbox_inches="tight", dpi=600); fig.savefig(f"{OUT}/fig4_coverage.pdf", bbox_inches="tight")
plt.close(fig)

# Fig 5: diagnostic
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axs[0]
for sc, fc in [("A", True), ("B", False)]:
    for p in policies:
        dd = t4[(t4.scenario == sc) & (t4.policy == p)]
        ax.scatter(dd.TSI, dd.realized, marker=MARK[p], s=10, alpha=0.6,
                   facecolor="#0072B2" if fc else "none", edgecolor="#0072B2", lw=0.6)
mx = max(t4.TSI.max(), t4.realized.max()) * 1.05
ax.plot([0, mx], [0, mx], color="grey", ls="--", lw=0.8)
ax.set_xlabel("Design-stage TSI (no outcome data)")
ax.text(0.03,0.93,f"r = {np.corrcoef(t4.TSI,t4.realized)[0,1]:.3f}",transform=ax.transAxes,fontsize=8)
ax.set_ylabel(r"Realized spread of mean $\hat\beta_m$ (log ratio)")
ax.set_title("(a)", loc="left")
for j, p in enumerate(["Upwind", "Regional"]):
    ax = axs[j + 1]
    S = Smean[("A", p)]
    im = ax.imshow(np.log(S), cmap="RdBu_r", vmin=-1.6, vmax=1.6)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names if j == 0 else [""] * len(names), fontsize=7)
    for a in range(len(names)):
        for b in range(len(names)):
            ax.text(b, a, f"{S[a, b]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(np.log(S[a, b])) > 0.9 else "black")
    ax.set_title(f"({'bc'[j]}) $S_{{mk}}$, {p} (A)", loc="left", fontsize=8.5)
    ax.set_xlabel("assumed true $T_k$", fontsize=7.5)
    if j == 0:
        ax.set_ylabel("analysis model $T_m$", fontsize=7.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_diagnostic.png", bbox_inches="tight", dpi=600); fig.savefig(f"{OUT}/fig5_diagnostic.pdf", bbox_inches="tight")
plt.close(fig)

print(t1.round(3).to_string())
print(t2[["scenario", "policy", "model", "mean", "sd", "cov_plugin", "cov_prop", "prop1_pred", "tau_relbias", "prop2_pred"]].round(3).to_string())
print(t3.round(3).to_string())
print("TSI-realized corr:", np.corrcoef(t4.TSI, t4.realized)[0, 1], " single-draw:", np.corrcoef(t4.TSI, t4.realized_single)[0, 1])
print(sel_freq); print(screen_freq)
