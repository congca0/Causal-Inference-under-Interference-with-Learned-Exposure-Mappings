 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transport_sim import (make_geography, draw_days, truth, observe, PDEModel, KernelModel,
                           SpectralOperator, GRID_XY, N, LDOM)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
geo = make_geography()
rng = np.random.default_rng(10_000)
days = draw_days(rng, geo)
base, _ = truth(geo, days, "A")
y = observe(rng, geo, base)
geo2 = dict(geo, rec=np.arange(N * N), rxy=GRID_XY)       # exposures on the full grid
_, e_true = truth(geo2, days, "A")
maps = {"Truth": e_true["Upwind"]}
for M in [PDEModel(True), PDEModel(False), KernelModel(), SpectralOperator()]:
    M.fit(geo, days, y, np.arange(60))
    e, _ = M.exposures(geo2, days)
    maps[M.name] = e["Upwind"]

fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
ext = [0, LDOM, 0, LDOM]
ax = axs[0, 0]
im = ax.imshow(base.mean(0).reshape(N, N).T, origin="lower", extent=ext, cmap="Greys", vmax=np.quantile(base.mean(0), 0.99))
grp = np.full(geo["J"], "#999999", dtype=object)
grp[geo["policies"]["Upwind"]] = "#0072B2"; grp[geo["policies"]["Downwind"]] = "#D55E00"
grp[geo["policies"]["Regional"]] = "#009E73"; grp[geo["policies"]["Point source"]] = "#E69F00"
ax.scatter(*geo["fxy"].T, s=12 + 14 * geo["s_base"], c=list(grp), edgecolor="k", lw=0.4, zorder=3)
ax.scatter(*geo["rxy"].T, s=0.6, c="#555555", alpha=0.35, zorder=2)
ax.scatter(*geo["mxy"][geo["mon_train"]].T, marker="^", s=16, c="white", edgecolor="k", lw=0.6, zorder=4)
ax.scatter(*geo["mxy"][geo["mon_test"]].T, marker="^", s=16, c="#f4a6a6", edgecolor="k", lw=0.6, zorder=4)
ax.annotate("", xy=(2.4, 9.3), xytext=(0.6, 8.85), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(0.5, 9.45, "prevailing wind", fontsize=7)
ax.rectangle = ax.add_patch(plt.Rectangle((3, 3), 4, 4, fill=False, ls="--", lw=0.7, color="#009E73"))
ax.axvline(3.5, color="#0072B2", lw=0.6, ls=":"); ax.axvline(6.5, color="#D55E00", lw=0.6, ls=":")
ax.set_title("(a) Domain, sources, monitors", loc="left")
ax.set_xticks([]); ax.set_yticks([])
vmax = np.quantile(maps["Truth"], 0.995)
for k, (name, v) in enumerate(maps.items()):
    ax = axs[(k + 1) // 3, (k + 1) % 3]
    im = ax.imshow(np.clip(v, 0, None).reshape(N, N).T, origin="lower", extent=ext, cmap="magma_r", vmin=0, vmax=vmax)
    ax.scatter(*geo["fxy"][geo["policies"]["Upwind"]].T, s=6, c="#0072B2", edgecolor="w", lw=0.3)
    ax.set_title(f"({'bcdef'[k]}) {name}", loc="left")
    ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, pad=0.02)
cb.set_label("Upwind-policy exposure change $e_i(A)$ (period mean)")
fig.savefig("out/fig1_domain.png", dpi=600, bbox_inches="tight"); fig.savefig("out/fig1_domain.pdf", bbox_inches="tight")
print({k: (float(v.mean()), float(v.max())) for k, v in maps.items()})
