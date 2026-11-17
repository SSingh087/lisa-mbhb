"""How much do the point-mass Fisher errors of the main grid change if the lens amplification is computed with the refined method used for the
Fig. 3 reproduction (>=200 grid points over the band, spline derivative in log M_Lz, adaptive y step) instead of the main-grid method
(fixed spacing 0.05 in w from w=1e-3, finite differences with the plateau steps)?   CLUSTER JOB, one source per job.
    python scripts/fisher_gridcheck.py --z 1 --m-tot 3e5 --q 0.3 --out DIR"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scipy.interpolate import CubicSpline
from lisa_mbhb import demo, fisher as FI, lensing as L, waveform as W

ap = argparse.ArgumentParser(); ap.add_argument("--z", type=float, required=True); ap.add_argument("--m-tot", type=float, required=True)
ap.add_argument("--q", type=float, default=0.3); ap.add_argument("--inc", type=float, default=0.9); ap.add_argument("--steps", default=None); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); LN10 = np.log(10.0)
steps = {**FI.STEP_MBH, **FI.STEP_LENS}
if a.steps and os.path.exists(a.steps): steps.update(json.load(open(a.steps)))


def refined(f, M, y):
    from gwlens.lens.point import F_analytic
    w = L.W_CONST * M * f; dw = float(min(np.clip(0.01 * w.min(), 0.05, 0.5), (w.max() - w.min()) / 200.0)); g0 = max(w.min() * 0.5, 1e-5)
    grid = np.arange(g0, w.max() + 2 * dw, dw); wc = np.clip(w, grid[0], None)
    def spl(y_):
        v = np.array([F_analytic(wi, y_) for wi in grid]); return CubicSpline(grid, v.real), CubicSpline(grid, v.imag)
    r0, i0 = spl(y); F0 = r0(wc) + 1j * i0(wc); dFdw = r0.derivative()(wc) + 1j * i0.derivative()(wc); h = min(4e-4, 0.02 / max(w.max(), 1.0))
    rp, ip = spl(y + h); rm, im = spl(y - h); dFdy = ((rp(wc) + 1j * ip(wc)) - (rm(wc) + 1j * im(wc))) / (2 * h)
    return F0, LN10 * w * dFdw, dFdy, len(grid)


m1 = a.m_tot / (1 + a.q); p = demo.default_params(m1=m1, m2=a.q * m1, z=a.z, inclination=a.inc)
S = FI.SourceDerivs(p, W.WaveformConfig(tc_ref=p["tc"])); out = []
for M in (1e6, 1e7, 1e8):
    for y in (0.3, 1.0, 2.0):
        r0 = S.fisher("pm", M, y, a.z, 0.5, steps)                                 # main-grid method
        F0, dM, dy, ng = refined(S.f, M, y); k = len(S.mbh_names)
        D = np.concatenate([np.stack([F0 * S.deriv_mbh(n, steps.get(n)) for n in S.mbh_names]), np.stack([dM * S.base, dy * S.base])])
        C, _, _ = FI.covariance(S.gram(D)); w = L.W_CONST * M * S.f
        out.append(dict(z=a.z, m_tot=a.m_tot, q=a.q, mass_lz=M, y=y, w_min=float(w.min()), w_max=float(w.max()), ngrid_refined=ng,
                        ngrid_main=int(np.ceil((w.max() + 0.1 - max(w.min() - 0.1, 1e-3)) / 0.05)), sig_y_main=r0["sigma"]["y"], sig_y_refined=float(np.sqrt(C[k + 1, k + 1])),
                        sig_m_main=r0["sigma"]["log10_mlz"], sig_m_refined=float(np.sqrt(C[k, k]))))
        o = out[-1]; print(f"z={a.z:g} M={a.m_tot:.0e} ML={M:.0e} y={y:g} wmax={o['w_max']:.2g} ngrid main/refined {o['ngrid_main']}/{ng}  sig_y ratio {o['sig_y_refined'] / o['sig_y_main']:.3f}  sig_M ratio {o['sig_m_refined'] / o['sig_m_main']:.3f}", flush=True)
json.dump(out, open(os.path.join(a.out, f"gridcheck_z{a.z:g}_m{a.m_tot:.0e}_q{a.q:g}.json"), "w"), indent=1)
