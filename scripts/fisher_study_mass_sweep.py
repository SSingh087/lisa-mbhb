"""Fig. 3-style sweep: 90% CI half-width of every parameter vs the lens (BH) mass M_Lz, median over random sky/inclination draws
with a 68% band (16-84 percentiles), PML and PML+NFW; unlensed band for the binary parameters.  CLUSTER JOB (light).
    python scripts/fisher_study_mass_sweep.py --rows DIR --fisher DIR --out DIR
Rows: rows/fisher_study_sweep_*.json (cluster/make_analysis_dag.py --sweep).  Masses whose Fisher jobs failed simply have no point."""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from lisa_mbhb import fisher_analysis as FA  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--fisher", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--scale", type=float, default=1.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); FA.style(a.scale)
rows = FA.load(a.rows, "sweep_")
for kname, known in (("kappa free", ()), ("kappa known", ("kappa",))):
    data = {}                                            # (model, param) -> {mass: [hw per draw]}
    draws = set()
    for r, d in rows:
        m, M = d["model"], d["m_ref"]; draws.add(d["info"]["draw"]); res = FA.analyse(r, m, d["info"], M, known)
        for n in res:
            if not n.startswith("_"): data.setdefault((m, n), {}).setdefault(M, []).append(FA.hw90(res, n))
    unl = {}
    for k in sorted(draws):
        u = FA.load_unlensed(a.fisher, f"z1_m1e6_q1_d{k}")
        if u:
            for n, v in FA.analyse(u[0], "unlensed").items():
                if not n.startswith("_"): unl.setdefault(n, []).append(FA.Z90 * v[1])
    params = [n for n in FA.LABEL if any(np.isfinite(v).any() for (m, nn), dd in data.items() if nn == n for v in map(np.array, dd.values()))]
    nc = 5; nr = -(-len(params) // nc); fig, axs = plt.subplots(nr, nc, figsize=(6.4 * nc, 5.4 * nr), squeeze=False)
    for ax in axs.ravel()[len(params):]: ax.axis("off")
    for ax, n in zip(axs.ravel(), params):
        for m in ("pml", "pmlnfw"):
            dd = data.get((m, n));
            if not dd: continue
            Ms = np.array(sorted(dd)); q = np.array([np.nanpercentile(dd[M], [16, 50, 84]) if np.isfinite(dd[M]).any() else [np.nan] * 3 for M in Ms])
            ax.fill_between(Ms, q[:, 0], q[:, 2], color=FA.MODEL_COL[m], alpha=0.25, lw=0)
            ax.plot(Ms, q[:, 1], "o-", color=FA.MODEL_COL[m], lw=2.5, ms=6, label=FA.MODEL_LAB[m])
        if n in unl:
            uq = np.nanpercentile(unl[n], [16, 50, 84]); ax.axhspan(uq[0], uq[2], color=FA.MODEL_COL["unlensed"], alpha=0.2)
            ax.axhline(uq[1], color=FA.MODEL_COL["unlensed"], ls="--", lw=2.5, label="unlensed")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(FA.LABEL.get(n, n)); ax.set_xlabel(r"$M_{Lz}\ [M_\odot]$"); ax.set_ylabel("90% CI")
    axs.ravel()[0].legend()
    fig.suptitle(f"90% CI vs lens mass, median and 68% band over {len(draws)} random sky/inclination draws ({kname}; z=1, M=1e6, q=1; "
                 r"y=0.1, $\kappa$=0.1, $|\gamma|$=0.1, M200=1e11)", y=1.0)
    fig.tight_layout(); fn = os.path.join(a.out, f"mass_sweep_{kname.replace(' ', '_')}.png"); fig.savefig(fn, bbox_inches="tight"); plt.close(fig); print("wrote", fn)
