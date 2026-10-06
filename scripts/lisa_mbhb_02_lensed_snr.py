"""Step 4-5: lens the same MBHB with a GWLENS point-mass lens and compare SNRs.

    python scripts/lisa_mbhb_02_lensed_snr.py --mass-lz 1e7 --y 0.3 0.5 1.0 2.0
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lisa_mbhb import demo, workflow   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--approximant", default="IMRPhenomD")
ap.add_argument("--mass-lz", type=float, default=1e7, help="redshifted lens mass [Msun]")
ap.add_argument("--y", type=float, nargs="+", default=[0.3, 0.5, 1.0, 2.0])
ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
a = ap.parse_args()

p = demo.default_params()
scan = workflow.lensed_snr_scan(p, a.mass_lz, a.y, a.approximant)
res = {}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
base = None
for y, r in scan.items():
    s = r["sig"]
    if base is None:
        base = r["snr_unlensed"]
        hp0, _, _ = s.polarizations(lensed=False)
        ax[1].loglog(s.f, 2 * s.f * np.abs(hp0), "k", lw=2, label="unlensed")
    res[y] = r["snr"]
    ax[0].semilogx(s.f, np.abs(r["F"]), label=f"y={y}")
    ax[1].loglog(s.f, 2 * s.f * np.abs(r["hp"]), lw=0.8, label=f"y={y}")
    print(f"M_Lz={a.mass_lz:.1e} y={y}: unlensed SNR={base:.0f}  lensed SNR={r['snr']:.0f}  ratio={r['snr'] / base:.3f}")
ax[0].set_xlabel("f [Hz]"); ax[0].set_ylabel("|F(f)|"); ax[0].legend()
ax[1].set_xlabel("f [Hz]"); ax[1].set_ylabel(r"$2f|\tilde h_+|$"); ax[1].legend(fontsize=8)
ax[1].set_ylim(1e-19, 3e-17)
os.makedirs(a.out, exist_ok=True)
fig.savefig(os.path.join(a.out, "lisa_mbhb_02_lensed_snr.png"), dpi=140, bbox_inches="tight")
json.dump(dict(unlensed=base, lensed={str(k): v for k, v in res.items()}),
          open(os.path.join(a.out, "lisa_mbhb_02_results.json"), "w"), indent=1)
