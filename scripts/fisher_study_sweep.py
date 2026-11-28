"""Sweep plot: 90% CI half-width (1.645 sigma) of key parameters vs the impact parameter y, one line per lens model, unlensed as a reference line.
    python scripts/fisher_study_sweep.py --fisher DIR --out DIR --source z1_m1e6_q1_d0 --m-ref 1e6 --kappa 0.1 --m200 1e11 [--fontscale 1]
Parameters whose Fisher is degenerate along the mass-sheet direction (they move by more than their own sigma for delta kappa = 0.1) are left out of a line."""
import argparse, csv, glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from lisa_mbhb.fisher_lenses import FisherLens, derived_jacobian  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--fisher", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--source", required=True); ap.add_argument("--m-ref", type=float, required=True)
ap.add_argument("--kappa", type=float, default=0.1); ap.add_argument("--m200", type=float, default=1e11)
ap.add_argument("--cond-ms", type=float, default=1e11); ap.add_argument("--fontscale", type=float, default=1.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
Z90 = 1.6448536269514722
COL = dict(pml="#d62728", nfw="#1f77b4", pmlnfw="#2ca02c", unlensed="#7f7f7f")
PANELS = [("log10_mlz", r"$\log_{10}M_{Lz}$"), ("y_ms", r"$y/\sqrt{1-\kappa}$ (derived)"), ("gamma1_ms", r"$\gamma_1/(1-\kappa)$ (derived)"),
          ("gamma2_ms", r"$\gamma_2/(1-\kappa)$ (derived)"), ("log10_mchirp", r"$\log_{10}\mathcal{M}_z$"), ("cos_inc", r"$\cos\iota$"),
          ("sin_ecl_lat", r"$\sin\beta$"), ("tc", r"$t_c$ [s]")]


def ci(r, model, zs, mref):
    """{name: 90% half-width or nan if unconstrained} for the free parameters and the derived quantities of one Fisher row."""
    names, F = r["names"], np.array(r["F"]); d = np.sqrt(np.diag(F)); w, U = np.linalg.eigh(0.5 * (F + F.T) / np.outer(d, d)); keep = w > w.max() / a.cond_ms
    Cp = (U[:, keep] / w[keep]) @ U[:, keep].T / np.outer(d, d); sig = np.sqrt(np.clip(np.diag(Cp), 0, None)); unc, vs = set(), []
    for k in np.where(~keep)[0]:
        v = U[:, k] / d
        v = v * 0.1 / abs(v[names.index("kappa")]) if "kappa" in names and abs(v[names.index("kappa")]) > 0 else v / np.max(np.abs(v / np.where(sig > 0, sig, 1)))
        vs.append(v); unc |= {names[i] for i in range(len(names)) if abs(v[i]) > sig[i] * 1.0001}
    out = {n: (np.nan if n in unc else Z90 * sig[i]) for i, n in enumerate(names)}
    if model != "unlensed":
        dn, dv, J = derived_jacobian(FisherLens(model, mref, zs), r["vals"], {k: r["binary"][k] for k in ("log10_dl", "tc")}, names)
        sd = np.sqrt(np.clip(np.diag(J @ Cp @ J.T), 0, None))
        for q, nm in enumerate(dn): out[nm] = np.nan if (nm in unc or any(abs(J[q] @ v) > sd[q] * 1.0001 for v in vs)) else Z90 * sd[q]
    return out


data, info = {}, None
for f in sorted(glob.glob(os.path.join(a.fisher, "fisher_study*.json"))):
    d = json.load(open(f))
    if d["info"]["source"] != a.source: continue
    if d["model"] == "unlensed": data["unlensed"] = (d["rows"][0], d); continue
    if abs(d["m_ref"] / a.m_ref - 1) > 1e-9: continue
    info = d["info"]; rows = [r for r in d["rows"] if r.get("ok") and abs(r["kappa"] - a.kappa) < 1e-9 and (r["m200"] is None or abs(r["m200"] / a.m200 - 1) < 1e-6)]
    data[d["model"]] = sorted(rows, key=lambda r: r["y"])
zs = info["z"]; fs = a.fontscale
plt.rcParams.update({"font.size": 20 * fs, "mathtext.fontset": "cm"})
fig, axs = plt.subplots(2, 4, figsize=(34 * fs ** 0.5, 15 * fs ** 0.5)); table = []
for ax, (p, lab) in zip(axs.ravel(), PANELS):
    for model in ("pml", "nfw", "pmlnfw"):
        if model not in data: continue
        ys = np.array([r["y"] for r in data[model]]); hw = np.array([ci(r, model, zs, a.m_ref).get(p, np.nan) for r in data[model]])
        table += [(model, p, y, h) for y, h in zip(ys, hw)]
        if np.isfinite(hw).any(): ax.plot(ys, hw, "o-", lw=3, ms=10, color=COL[model], label=model)
    if "unlensed" in data and p in data["unlensed"][0]["names"]:
        u = ci(data["unlensed"][0], "unlensed", zs, a.m_ref).get(p, np.nan); ax.axhline(u, color=COL["unlensed"], ls="--", lw=3, label="unlensed")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(lab, fontsize=26 * fs); ax.set_xlabel("$y$", fontsize=24 * fs); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=20 * fs); ax.set_ylabel("90% CI half-width", fontsize=20 * fs)
axs[0, 0].legend(fontsize=22 * fs)
fig.suptitle(f"90% CI half-width vs impact parameter y   (z={zs:g}, M={info['m_tot']:g} Msun, q={info['q']:g}, extrinsic draw {info.get('draw', '-')}; "
             f"M_Lz={a.m_ref:g}, kappa={a.kappa:g}, M200={a.m200:g}).  Lines are omitted where the parameter is unconstrained (mass sheet).", fontsize=22 * fs)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fn = os.path.join(a.out, f"sweep_y_{a.source}_ref{a.m_ref:.0e}_k{a.kappa:g}.png"); fig.savefig(fn, dpi=100)
with open(fn.replace(".png", ".csv"), "w", newline="") as fh:
    w_ = csv.writer(fh); w_.writerow(["model", "parameter", "y", "ci90_half_width"]); w_.writerows(table)
print("wrote", fn)
