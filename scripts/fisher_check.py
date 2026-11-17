"""Independent validation of the Fisher matrices: compare the EXACT chi^2 change  <h(theta+d)-h(theta)|h(theta+d)-h(theta)>
with the Fisher prediction d^T F d, for displacements d of fixed Fisher-size eps along (a) every single parameter,
(b) random directions of the full (MBH+lens) space, (c) the weakest (smallest-eigenvalue) direction of the normalised F.
Ratio exact/predicted = 1 means the matrix (derivatives, normalisation, step sizes) is right at that order.
CLUSTER JOB.   python scripts/fisher_check.py --shards DIR --out DIR [--eps 0.3]
"""
import argparse, glob, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lisa_mbhb import demo, fisher as FI, inference as I, waveform as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--eps", type=float, default=0.3); ap.add_argument("--nrand", type=int, default=4)
ap.add_argument("--select", nargs="+", default=["pm_stable", "nfw:1e11", "halo:1e11", "q0.8"])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

pts = []
for f in sorted(glob.glob(os.path.join(a.shards, "fisher*.json"))):
    d = json.load(open(f))
    for r, m in zip(d["rows"], d["fisher"]):
        if r.get("ok") and m: pts.append((r, m["names"], np.array(m["F"])))
chosen = []
for sel in a.select:
    if sel == "pm_stable": c = [p for p in pts if p[0]["lens"] == "pm" and p[0]["cond"] < 1e8 and p[0]["q"] < .5]
    elif sel == "q0.8": c = [p for p in pts if p[0]["lens"] == "pm" and p[0]["q"] == 0.8]
    else: c = [p for p in pts if p[0]["lens"] == sel and p[0]["q"] < .5]
    if c: chosen.append(sorted(c, key=lambda p: p[0]["snr_lensed"])[len(c) // 2])      # median-SNR representative
rng = np.random.default_rng(1); out = []
for r, names, F in chosen:
    m1 = r["m_tot"] / (1 + r["q"]); p = demo.default_params(m1=m1, m2=r["q"] * m1, z=r["z"], inclination=r["inc"])
    S = FI.SourceDerivs(p, W.WaveformConfig(tc_ref=p["tc"]))
    lens = FI.LensSpec(r["lens"]); lv0 = lens.values(r["mass_lz"], r["y"]); th0 = dict(zip(I.SAMPLED, I.params_to_theta(p)))
    F0 = S.amp(lens, lv0, r["z"], 0.5); h0 = F0 * S.base
    mbh = [n for n in names if n in I.SAMPLED]; ln = [n for n in names if n not in I.SAMPLED]

    def chi2(delta):
        th, lv = dict(th0), dict(lv0)
        for n, dd in zip(names, delta):
            if n in th: th[n] += dd
            else: lv[n] += dd
        base = FI._stack(S.model(I.theta_to_params([th[k] for k in I.SAMPLED])), S.model) if any(n in th0 and dd for n, dd in zip(names, delta)) else S.base
        Fa = S.amp(lens, lv, r["z"], 0.5) if any(n not in th0 and dd for n, dd in zip(names, delta)) else F0
        h = Fa * base; return S.ip(h - h0, h - h0)

    def test(label, v):
        pred = float(v @ F @ v); res = chi2(v)
        out.append(dict(point=f"{r['lens']} z={r['z']:g} M={r['m_tot']:.0e} q={r['q']:g} ML={r['mass_lz']:.0e} y={r['y']:g}", test=label, predicted=pred, exact=res, ratio=res / pred))
        print(f"{out[-1]['point']:48s} {label:22s} exact/predicted = {res / pred:8.4f}", flush=True)

    for k, n in enumerate(names):                                    # (a) single parameters, conditional
        v = np.zeros(len(names)); v[k] = a.eps / np.sqrt(F[k, k]); test(f"single {n}", v)
    for i in range(a.nrand):                                          # (b) random joint directions
        v = rng.normal(size=len(names)); v *= a.eps / np.sqrt(v @ F @ v); test(f"random {i}", v)
    d = np.sqrt(np.diag(F)); Fn = F / np.outer(d, d); w, U = np.linalg.eigh(Fn)   # (c) weakest direction
    for k in (0, 1):
        v = U[:, k] / d; v *= a.eps / np.sqrt(v @ F @ v); test(f"weakest eig {k} (lam={w[k]:.1e})", v)
json.dump(out, open(os.path.join(a.out, "fisher_check.json"), "w"), indent=1)
r_ = np.array([o["ratio"] for o in out]); print(f"\nratio exact/predicted: median {np.median(r_):.3f}, min {r_.min():.3f}, max {r_.max():.3f}  (1 = Fisher is right at eps={a.eps})")
