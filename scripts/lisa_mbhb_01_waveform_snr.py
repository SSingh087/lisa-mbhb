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
from lisa_mbhb import demo, inference as I, noise, overlap, waveform as W   # noqa: E402

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

results, sigs = {}, {}
for name in a.approximants:
    cfg = W.WaveformConfig(approximant=name, tc_ref=p["tc"])
    inj = I.make_injection(p, cfg, tdi=a.tdi)
    sigs[name] = inj
    results[name] = dict(snr=inj.snr, snr_channels=inj.snr_channels, n_freq=len(inj.model.f))
    print(f"{name:14s} SNR={inj.snr:9.1f}  A={inj.snr_channels['A']:.0f} E={inj.snr_channels['E']:.0f} "
          f"T={inj.snr_channels['T']:.1f}  ({len(inj.model.f)} freq points)")

ref = a.approximants[0]
for name in a.approximants[1:]:
    # compare on the reference grid
    m = sigs[ref].model
    other = I.TDIModel(m.f, W.WaveformConfig(approximant=name, tc_ref=p["tc"]), tdi=a.tdi)(p)
    # approximants use different time-origin conventions -> maximise over a +-300 s shift
    mm = overlap.mismatch(m, sigs[ref].signal, other, max_dt=300.0, n_dt=601)
    results[name]["mismatch_vs_" + ref] = mm
    print(f"mismatch {name} vs {ref}: {mm:.2e}")

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
