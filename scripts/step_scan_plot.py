"""Readable step-scan figure: one row per lens parameter, one column per lens model, one colour per fiducial point,
one shared legend.  y axis = F_kk / F_kk(base step); dotted = +-1 %; red line = step used in the Fisher runs.
    python scripts/step_scan_plot.py --scan DIR --steps steps.json --out DIR
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FID = {0: "Caliskan  M_BH=1e7  y=0.1  k=0.1  M200=1e11", 1: "z1_m1e6_q0.3  M_BH=1e7  y=1  k=0.1  M200=1e9",
       2: "Caliskan  M_BH=1e6  y=0.1  k=0.1  M200=1e11", 3: "z1_m1e6_q0.3  M_BH=1e6  y=1  k=0.1  M200=1e9",
       4: "Caliskan  M_BH=1e7  y=1  k=0  M200=1e11", 5: "z1_m1e6_q0.3  M_BH=1e7  y=1  k=0.1  M200=1e11"}
ap = argparse.ArgumentParser(); ap.add_argument("--scan", required=True); ap.add_argument("--steps", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--ylim", type=float, default=0.05); ap.add_argument("--xmin-note", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); used = json.load(open(a.steps))
D = {}
for f in glob.glob(os.path.join(a.scan, "stepscan_*.json")):
    m = re.match(r"stepscan_(.+)_f(\d)_(pml|nfw|pmlnfw)\.json", os.path.basename(f))
    if m: D[(m.group(1), m.group(3), int(m.group(2)))] = json.load(open(f))
params = ["log10_mlz", "log10_psi0", "log10_xs", "y", "kappa", "gamma1", "gamma2"]; models = ["pml", "nfw", "pmlnfw"]
cols = plt.cm.tab10(np.arange(6)); plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15})
fig, axs = plt.subplots(len(params), 3, figsize=(17, 3.3 * len(params)), sharex=True)
for i, p in enumerate(params):
    for j, mo in enumerate(models):
        ax = axs[i, j]; ks = [k for k in range(6) if (p, mo, k) in D]
        if not ks: ax.axis("off"); continue
        for k in ks:
            d = D[(p, mo, k)]; st = np.array(d["mults"]) * d["base_step"]; fk = np.array(d["fkk"], float)
            ax.plot(st, fk / fk[list(d["mults"]).index(1)], "o-", ms=4, lw=1.4, color=cols[k])
        for s in (-0.01, 0.01): ax.axhline(1 + s, color="0.5", ls=":", lw=1)
        if p in used: ax.axvline(used[p], color="red", lw=1.6)
        ax.set_xscale("log"); ax.set_ylim(1 - a.ylim, 1 + a.ylim); ax.grid(alpha=0.25)
        ax.set_title(f"{p}  |  {mo}", fontsize=14)
        if j == 0 or not any((p, m2, k) in D for m2 in models[:j] for k in range(6)): ax.set_ylabel(r"$F_{kk}/F_{kk}^{\rm base}$")
for ax in axs[-1]: ax.set_xlabel("finite-difference step")
h = [plt.Line2D([], [], color=cols[k], marker="o", lw=1.4, label=FID[k]) for k in range(6)]
h += [plt.Line2D([], [], color="red", lw=1.6, label="step used in the Fisher runs"), plt.Line2D([], [], color="0.5", ls=":", label="±1 %")]
fig.legend(handles=h, loc="lower center", ncol=2, fontsize=14, frameon=True)
fig.suptitle(f"Step scans: F_kk relative to the base step (axis ±{a.ylim:.0%}; points outside the axis are off-scale)", fontsize=18, y=0.995)
fig.tight_layout(rect=(0, 0.085, 1, 0.985)); out = os.path.join(a.out, "step_scan_plot.png"); fig.savefig(out, dpi=110); print("wrote", out)
