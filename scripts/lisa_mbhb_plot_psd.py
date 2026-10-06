"""Project an MBHB onto the LISA SciRDv1 sensitivity: characteristic strain vs noise."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lisa_mbhb import inference as I, noise, waveform as W   # noqa: E402
from lisa_mbhb.constants import YEAR                          # noqa: E402

m1, m2, dl = W.source_to_detector(5e5, 2.5e5, 1.0)
p = dict(mass1=m1, mass2=m2, spin1z=0.3, spin2z=0.1, distance=dl, inclination=0.9, coa_phase=0.3,
         psi=0.7, ecl_lon=1.2, ecl_lat=0.4, tc=0.9 * YEAR)
cfg = W.WaveformConfig(tc_ref=p["tc"])
inj = I.make_injection(p, cfg)
f = inj.model.f
hp, hc, _ = W.generate(p, f, cfg)

fg = np.logspace(-5, 0, 1000)
sn = noise.sensitivity_curve(fg)
sn_c = noise.sensitivity_curve(fg, confusion=True)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(fg, np.sqrt(fg * sn), label="SciRDv1 (sky-averaged)")
ax.loglog(fg, np.sqrt(fg * sn_c), "--", label="+ Galactic confusion (4 yr)")
ax.loglog(f, 2 * f * np.sqrt(np.abs(hp) ** 2 + np.abs(hc) ** 2), label=f"MBHB, SNR={inj.snr:.0f}")
ax.set_xlabel("f [Hz]")
ax.set_ylabel(r"$\sqrt{f S_n}$,  $2f|\tilde h|$")
ax.set_ylim(1e-21, 1e-15)
ax.legend()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/lisa_mbhb_psd_projection.png", dpi=150, bbox_inches="tight")
print("saved figures/lisa_mbhb_psd_projection.png")
