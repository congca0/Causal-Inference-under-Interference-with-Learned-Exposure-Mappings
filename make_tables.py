"""make_tables.py — markdown tables for the manuscript from out/*.csv."""
import pandas as pd, json

t1 = pd.read_csv("out/table1_prediction.csv")
t2 = pd.read_csv("out/table2_estimation.csv")
t3 = pd.read_csv("out/table3_intervals.csv")
names = ["AD-PDE", "D-PDE", "Kernel", "LSO"]
pols = ["Regional", "Point source", "Upwind", "Downwind"]
out = []

# Table 1
out.append("| Model | A: train | A: held-out days | A: held-out monitors | A: corr² | B: train | B: held-out days | B: held-out monitors | B: corr² |")
out.append("|:--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|")
for m in names:
    a = t1[(t1.scenario == "A") & (t1.model == m)].iloc[0]; b = t1[(t1.scenario == "B") & (t1.model == m)].iloc[0]
    out.append(f"| {m.replace('-', chr(0x2011))} | {a.train:.3f} | {a.temporal:.3f} | {a.spatial:.3f} | {a.corr2_spatial:.2f} | "
               f"{b.train:.3f} | {b.temporal:.3f} | {b.spatial:.3f} | {b.corr2_spatial:.2f} |")
open("out/T1.md", "w").write("\n".join(out)); out = []

# Table 2
out.append("| Policy | Model | A: mean (SD) | A: βρ | A: cov. | A: τ bias | B: mean (SD) | B: βρ | B: cov. | B: τ bias |")
out.append("|:------------|:-----------|--------------:|------:|------:|-------:|--------------:|------:|------:|-------:|")
for p in pols:
    for m in ["Oracle"] + names:
        cells = []
        for sc in "AB":
            r = t2[(t2.scenario == sc) & (t2.policy == p) & (t2.model == m)].iloc[0]
            pr = "—" if m == "Oracle" else f"{r.prop1_pred:.3f}"
            tb = "—" if m == "Oracle" else f"{100 * r.tau_relbias:+.1f}%".replace("+0.0%", "0.0%").replace("-0.0%", "0.0%").replace("-", "−")
            cells += [f"{r['mean']:.3f} ({r.sd:.3f})", pr, f"{r.cov_plugin:.2f}", tb]
        out.append(f"| {p if m == 'Oracle' else ''} | {m.replace('-', chr(0x2011))} | " + " | ".join(cells) + " |")
open("out/T2.md", "w").write("\n".join(out)); out = []

# Table 3
meths = ["Selected: plug-in", "Selected: within-model", "Selected: delta=0.1", "Selected: delta=0.2",
         "Equal-weight ensemble", "Union (screened)", "Union (all candidates)"]
lab = {"Selected: plug-in": "Selected model, plug-in", "Selected: within-model": "Selected model, within-model propagation",
       "Selected: delta=0.1": "Selected model, sensitivity δ = 0.1", "Selected: delta=0.2": "Selected model, sensitivity δ = 0.2",
       "Equal-weight ensemble": "Equal-weight ensemble", "Union (screened)": "Union, screened candidates",
       "Union (all candidates)": "Union, all candidates"}
out.append("| Scenario | Method | Regional | Point source | Upwind | Downwind |")
out.append("|:-----------|:------------------------------------|-----------:|-----------:|-----------:|-----------:|")
for sc in "AB":
    for k, mth in enumerate(meths):
        cells = []
        for p in pols:
            r = t3[(t3.scenario == sc) & (t3.policy == p) & (t3.method == mth)].iloc[0]
            cells.append(f"{r.coverage:.2f} ({r.median_width:.2f})")
        out.append(f"| {sc if k == 0 else ''} | {lab[mth]} | " + " | ".join(cells) + " |")
open("out/T3.md", "w").write("\n".join(out))
print(open("out/T1.md").read()); print(open("out/T2.md").read()); print(open("out/T3.md").read())
print(json.load(open("out/screen_freq.json")))
