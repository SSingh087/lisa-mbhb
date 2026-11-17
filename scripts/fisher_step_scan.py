"""Finite-difference step scan: F_kk = <d_k h|d_k h> vs step multiplier for ONE parameter at ONE fiducial
(source + lens).  A stable step sits on the plateau.  CLUSTER JOB.

    python scripts/fisher_step_scan.py --param log10_mchirp --z 1 --m-tot 1e6 --q 0.5 --inc 0.9 \
        --lens pm --mass-lz 1e7 --y 1 --out DIR --tag _fid0
Writes DIR/stepscan_<param><tag>.json.  Collect + pick steps with scripts/fisher_plot.py steps.
"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lisa_mbhb import demo, fisher as FI, waveform as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--param", required=True, help="one of SAMPLED or log10_mlz / y / log10_mhalo; 'all' = every parameter (slow)")
ap.add_argument("--approximant", default="IMRPhenomD")
ap.add_argument("--z", type=float, default=1.0); ap.add_argument("--m-tot", type=float, default=1e6)
ap.add_argument("--q", type=float, default=0.5); ap.add_argument("--inc", type=float, default=0.9)
ap.add_argument("--lens", default="pm"); ap.add_argument("--mass-lz", type=float, default=1e7)
ap.add_argument("--y", type=float, default=1.0); ap.add_argument("--zl-frac", type=float, default=0.5)
ap.add_argument("--mults", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
a = ap.parse_args()

m1 = a.m_tot / (1 + a.q); m2 = a.q * m1
p = demo.default_params(m1=m1, m2=m2, z=a.z, inclination=a.inc)
S = FI.SourceDerivs(p, W.WaveformConfig(approximant=a.approximant, tc_ref=p["tc"]))
lens = FI.LensSpec(a.lens)
params = (list(FI.SAMPLED) + lens.names) if a.param == "all" else [a.param]
base = {**FI.STEP_MBH, **FI.STEP_LENS}
os.makedirs(a.out, exist_ok=True)
for name in params:
    t0 = time.time(); res = []
    for m in a.mults:
        h = base[name] * m
        S._D.clear()                                      # don't accumulate derivative arrays
        try:
            v = S.fkk(name, h, lens, a.mass_lz, a.y, a.z, a.zl_frac)
        except Exception as e:
            v = float("nan"); print("fail", name, m, repr(e))
        res.append(v); print(f"{name} mult={m:g} step={h:.3g} F_kk={v:.6e}", flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        v = np.array(res); fig, ax = plt.subplots(figsize=(4, 3))
        ax.loglog(np.array(a.mults) * base[name], np.where(v > 0, v, np.nan), "o-"); ax.set_xlabel("step"); ax.set_ylabel("F_kk")
        ax.set_title(f"{name} {a.tag}", fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(a.out, f"stepscan_{name}{a.tag}.png"), dpi=110); plt.close(fig)
    except Exception as e: print("plot failed", repr(e))
    json.dump(dict(param=name, base_step=base[name], mults=a.mults, fkk=res, fiducial=vars(a), seconds=time.time() - t0),
              open(os.path.join(a.out, f"stepscan_{name}{a.tag}.json"), "w"))
