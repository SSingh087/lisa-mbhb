"""Step 1-2: choose a waveform approximant, build the LISA TDI signal, compute the SNR.

    python scripts/lisa_mbhb_01_waveform_snr.py --approximants IMRPhenomD IMRPhenomXHM
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from pycbc.waveform import fd_approximants                      # noqa: E402
from lisa_mbhb import demo, noise, workflow   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--approximants", nargs="+", default=["IMRPhenomD", "IMRPhenomXHM", "IMRPhenomHM"])
ap.add_argument("--m1", type=float, default=5e5)
ap.add_argument("--m2", type=float, default=2.5e5)
ap.add_argument("--z", type=float, default=1.0)
ap.add_argument("--tdi", type=int, default=1)
ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
a = ap.parse_args()

print("PyCBC Phenom FD approximants:", [x for x in fd_approximants() if "Phenom" in x and "Tidal" not in x])
p = demo.default_params(a.m1, a.m2, a.z)
print(f"detector-frame m1={p['mass1']:.3e} m2={p['mass2']:.3e} dL={p['distance']:.0f} Mpc")

comp = workflow.compare_approximants(p, a.approximants, tdi=a.tdi)
results, sigs = {}, {}
for name, r in comp.items():
    inj = r["inj"]
    sigs[name] = inj
    results[name] = dict(snr=r["snr"], snr_channels=r["snr_channels"], n_freq=len(inj.model.f))
    print(f"{name:14s} SNR={r['snr']:9.1f}  A={r['snr_channels']['A']:.0f} E={r['snr_channels']['E']:.0f} "
          f"T={r['snr_channels']['T']:.1f}  ({len(inj.model.f)} freq points)")
    if r["mismatch_vs_ref"] is not None:
        results[name]["mismatch_vs_" + a.approximants[0]] = r["mismatch_vs_ref"]
        print(f"   mismatch vs {a.approximants[0]}: {r['mismatch_vs_ref']:.2e}")

fg = np.logspace(-5, 0, 800)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.loglog(fg, np.sqrt(fg * noise.sensitivity_curve(fg)), "k", label="SciRDv1")
ax.loglog(fg, np.sqrt(fg * noise.sensitivity_curve(fg, confusion=True)), "k--", lw=0.8, label="+ confusion")
for name, inj in sigs.items():
    f = inj.model.f
    ax.loglog(f, 2 * f * np.abs(inj.signal["A"]), label=f"{name} (A), SNR={inj.snr:.0f}")
ax.set_ylim(1e-21, 1e-15)
ax.set_xlabel("f [Hz]"); ax.set_ylabel(r"$\sqrt{fS_n}$, $2f|\tilde A|$"); ax.legend(fontsize=8)
os.makedirs(a.out, exist_ok=True)
fig.savefig(os.path.join(a.out, "lisa_mbhb_01_waveform_snr.png"), dpi=140, bbox_inches="tight")
json.dump(results, open(os.path.join(a.out, "lisa_mbhb_01_results.json"), "w"), indent=1)
print("saved figure + json in", a.out)
