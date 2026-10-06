"""Step 6: mismatch between lensed and unlensed LISA signals vs y and lens mass.

Mismatch is maximised over a global phase and a time shift (so a pure delay is not
counted as a difference).  Use --quick for a small grid.

    python scripts/lisa_mbhb_03_mismatch.py --mass-lz 3e6 1e7 3e7 --y 0.3 0.5 1 2 3
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lisa_mbhb import demo, lensing as L, waveform as W   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--approximant", default="IMRPhenomD")
ap.add_argument("--mass-lz", type=float, nargs="+", default=[3e6, 1e7, 3e7])
ap.add_argument("--y", type=float, nargs="+", default=[0.3, 0.5, 1.0, 2.0, 3.0])
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
a = ap.parse_args()
if a.quick:
    a.mass_lz, a.y = [1e7], [0.5, 1.0, 2.0]

p = demo.default_params()
cfg = W.WaveformConfig(approximant=a.approximant, tc_ref=p["tc"])
mm = {}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for M in a.mass_lz:
    max_dt, n_dt = demo.shift_range(M, cap=1001 if a.quick else 2501)
    vals = []
    for y in a.y:
        s = L.LensedMBHB(p, L.PointMassLens(M, y), cfg)
        vals.append(s.mismatch(max_dt, n_dt))
        print(f"M_Lz={M:.0e} y={y}: mismatch={vals[-1]:.3e}  (SNR_lensed/unlensed={s.snr() / s.snr(False):.3f})", flush=True)
    mm[M] = vals
    ax.loglog(a.y, vals, "o-", label=f"$M_{{Lz}}$={M:.0e}")
ax.axhline(1 / (2 * 3376**2) * 11, c="gray", ls=":", label="~distinguishability (11 params, SNR 3400)")
ax.set_xlabel("impact parameter y"); ax.set_ylabel("mismatch"); ax.legend(fontsize=8)
os.makedirs(a.out, exist_ok=True)
fig.savefig(os.path.join(a.out, "lisa_mbhb_03_mismatch.png"), dpi=140, bbox_inches="tight")
json.dump({str(k): v for k, v in mm.items()} | {"y": a.y}, open(os.path.join(a.out, "lisa_mbhb_03_results.json"), "w"), indent=1)
