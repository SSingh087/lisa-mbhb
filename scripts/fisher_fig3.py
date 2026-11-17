"""Reproduce the Fisher analysis behind Fig. 3 of Caliskan et al. 2023 (arXiv:2206.02803, PRD 107, 043029):
relative uncertainty on the redshifted point-mass lens mass, Delta M_Lz / M_Lz, vs M_Lz, for y = 0.1 and an MBHB with
M_Tz = 2e6 Msun (detector frame), q = 1, z_S = 1; O(100) random extrinsic draws; 13-parameter Fisher matrix (11 source + M_Lz, y),
LISA SciRDv1, TDI A/E/T, at most 1 yr before merger.  Waveform variants: ISCO-truncated IMRPhenomD, IMRPhenomD (MR),
IMRPhenomHM (+- extremal-ish aligned/anti-aligned spins, chi = +-0.99 here).  NO SNR normalisation.

Efficient: the frequency grid (and so F, dF/dlogM, dF/dy) does NOT depend on the extrinsic draw -> computed ONCE per (grid, M_Lz)
  lens  --i K                      cache F, dF/dlog10M_Lz, dF/dy for mass index K (both grids)     [cheap, parallel over K]
  draws --config C --start a --stop b   Fisher for draws a..b-1 of config C over all masses        [parallel over chunks]
  plot  --shards DIR --out DIR     median/1-sigma curves = the paper's Fig. 3, + critical-mass table vs the paper's numbers
Derivative accuracy at large M_Lz: dF/dlogM from the spline derivative (exact chain rule), dF/dy by a central difference with
h_y = min(4e-4, 0.02 / w_max) so that the phase change across the step stays small even when w reaches 1e4.
"""
import argparse, glob, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

LN10 = np.log(10.0)
M_TZ, Z_S, Q, Y = 2e6, 1.0, 1.0, 0.1
LOGM = np.arange(3.0, 9.0001, 0.25)                                  # 25 lens masses, 1e3 .. 1e9
CONFIGS = {"isco": ("IMRPhenomD", 0.0, True), "mr": ("IMRPhenomD", 0.0, False), "hm": ("IMRPhenomHM", 0.0, False),
           "hmp": ("IMRPhenomHM", 0.99, False), "hmm": ("IMRPhenomHM", -0.99, False)}
GRID_OF = {"isco": "isco", "mr": "full", "hm": "full", "hmp": "full", "hmm": "full"}
# paper (Sec. IV.A): median and 68% interval of the critical lens mass (100% relative error), Msun
PAPER_MCRIT = {"isco": (1.08e5, 0.40e5, 0.64e5), "mr": (4.73e3, 1.75e3, 2.45e3), "hm": (3.19e3, 1.17e3, 1.39e3),
               "hmp": (1.49e3, 0.49e3, 0.62e3), "hmm": (6.63e3, 2.24e3, 3.46e3)}
LABEL = {"isco": "ISCO ($\\chi_1,\\chi_2=0$)", "mr": "MR ($\\chi_1,\\chi_2=0$)", "hm": "MR + HM ($\\chi_1,\\chi_2=0$)",
         "hmp": "MR + HM ($\\chi_1,\\chi_2=+1$)", "hmm": "MR + HM ($\\chi_1,\\chi_2=-1$)"}
COLOR = {"isco": "#f28e2b", "mr": "#e15759", "hm": "#59a14f", "hmp": "#8c564b", "hmm": "#4e79a7"}

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["lens", "draws", "plot"])
ap.add_argument("--i", type=int, default=0); ap.add_argument("--config", default="mr")
ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=1)
ap.add_argument("--steps", default=None); ap.add_argument("--cache", default=None)
ap.add_argument("--shards", default=None); ap.add_argument("--out", required=True)
ap.add_argument("--masses", type=int, nargs="+", default=None, help="subset of mass indices (tests)")
ap.add_argument("--selfcheck", action="store_true"); ap.add_argument("--grid", choices=["both", "full", "isco"], default="both")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
cache = a.cache or os.path.join(a.out, "lenscache")


def source(draw, name):
    from lisa_mbhb import demo
    rng = np.random.default_rng(1000 + draw); approx, chi, _ = CONFIGS[name]
    m = M_TZ / (1 + Z_S) / (1 + Q)                                    # source-frame m1; m2 = Q m1
    return demo.default_params(m1=m, m2=Q * m, z=Z_S, inclination=float(np.arccos(rng.uniform(-1, 1))), coa_phase=float(rng.uniform(0, 2 * np.pi)),
                               psi=float(rng.uniform(0, np.pi)), ecl_lon=float(rng.uniform(0, 2 * np.pi)), ecl_lat=float(np.arcsin(rng.uniform(-1, 1))),
                               spin1z=chi, spin2z=chi)


def wave_cfg(name, p):
    from lisa_mbhb import waveform as W
    from lisa_mbhb.constants import MTSUN
    approx, _, isco = CONFIGS[name]
    kw = dict(approximant=approx, tc_ref=p["tc"])
    if isco: kw["f_max"] = 1.0 / (6 ** 1.5 * np.pi * (p["mass1"] + p["mass2"]) * MTSUN)     # GW frequency at the ISCO
    return W.WaveformConfig(**kw)


def cov_pinv(F, cut=1e-13):
    """Covariance via the diagonally-normalised pseudo-inverse: exact null directions (q=1: chi1-chi2 has no effect) are dropped instead of
    amplified by round-off.  Returns (C, number of dropped directions, normalised condition number of the kept part)."""
    Fs = 0.5 * (F + F.T); d = 1 / np.sqrt(np.clip(np.diag(Fs), 1e-300, None)); Fn = Fs * d[:, None] * d[None, :]
    w, U = np.linalg.eigh(Fn); keep = w > cut * w.max()
    Cn = (U[:, keep] / w[keep]) @ U[:, keep].T
    return Cn * d[:, None] * d[None, :], int((~keep).sum()), float(w[keep].max() / w[keep].min())


def lens_arrays(f, M, y):
    from scipy.interpolate import CubicSpline
    from lisa_mbhb import lensing as L
    from gwlens.lens.point import F_analytic
    w = L.W_CONST * M * f
    dw = float(min(np.clip(0.01 * w.min(), 0.05, 0.5), (w.max() - w.min()) / 200.0))     # >= 200 grid points across the band, also at low M_Lz
    g0 = max(w.min() * 0.5, 1e-5)                                                         # cover the whole band (F -> 1 below 1e-5)
    grid = np.arange(g0, w.max() + 2 * dw, dw); wc = np.clip(w, grid[0], None)

    def spl(y_):
        v = np.array([F_analytic(wi, y_) for wi in grid]); return CubicSpline(grid, v.real), CubicSpline(grid, v.imag)
    r0, i0 = spl(y); F0 = r0(wc) + 1j * i0(wc); dFdw = r0.derivative()(wc) + 1j * i0.derivative()(wc)
    h = min(4e-4, 0.02 / max(w.max(), 1.0))
    rp, ip = spl(y + h); rm, im = spl(y - h)
    dFdy = ((rp(wc) + 1j * ip(wc)) - (rm(wc) + 1j * im(wc))) / (2 * h)
    return F0, LN10 * w * dFdw, dFdy, dict(w_min=float(w.min()), w_max=float(w.max()), dw=dw, h_y=h, ngrid=len(grid))


if a.mode == "lens":
    for gname in (("full", "isco") if a.grid == "both" else (a.grid,)):
        name = "mr" if gname == "full" else "isco"; p = source(0, name)
        from lisa_mbhb import waveform as W
        f = W.design_grid(p, wave_cfg(name, p)); fn = os.path.join(cache, f"{gname}_{a.i:02d}.npz")
        if os.path.exists(fn): print("skip", fn); continue
        os.makedirs(cache, exist_ok=True); t0 = time.time(); M = 10 ** LOGM[a.i]
        F0, dM, dy, info = lens_arrays(f, M, Y)
        # self-check of the spline chain-rule derivative against a tiny finite difference of M
        if a.selfcheck:                                              # chain-rule dF/dlogM vs a direct finite difference of F(M)
            e = 1e-5; Fp = lens_arrays(f, 10 ** (LOGM[a.i] + e), Y)[0]; Fm = lens_arrays(f, 10 ** (LOGM[a.i] - e), Y)[0]
            print(f"  selfcheck dF/dlogM: max|spline - FD| / max|spline| = {np.abs((Fp - Fm) / (2 * e) - dM).max() / np.abs(dM).max():.2e}", flush=True)
        np.savez(fn + ".tmp.npz", F0=F0, dM=dM, dy=dy, nf=len(f), f0=f[0], f1=f[-1], **info); os.replace(fn + ".tmp.npz", fn)
        print(f"{gname} logM={LOGM[a.i]:.2f} nf={len(f)} {info} {time.time() - t0:.0f}s |F0-1| max {np.abs(F0 - 1).max():.3f}", flush=True)

elif a.mode == "draws":
    from lisa_mbhb import fisher as FI
    steps = {**FI.STEP_MBH}
    if a.steps and os.path.exists(a.steps): steps.update({k: v for k, v in json.load(open(a.steps)).items() if k in FI.STEP_MBH})
    idx = a.masses if a.masses else list(range(len(LOGM))); gname = GRID_OF[a.config]
    arr = {}
    for i in idx:
        z = np.load(os.path.join(cache, f"{gname}_{i:02d}.npz")); arr[i] = (z["F0"], z["dM"], z["dy"], int(z["nf"]))
    for d in range(a.start, a.stop):
        fn = os.path.join(a.out, f"fig3_{a.config}_d{d:03d}.json")
        if os.path.exists(fn): print("skip", fn); continue
        t0 = time.time(); p = source(d, a.config); S = FI.SourceDerivs(p, wave_cfg(a.config, p))
        assert all(v[3] == S.f.size for v in arr.values()), "frequency grid differs from the lens cache"
        rel, rel_inv, ndrop, sy, cond, snr_u, snr_l, ok = [], [], [], [], [], float(np.sqrt(S.ip(S.base, S.base))), [], []
        for i in idx:
            F0, dM, dy, _ = arr[i]
            try:
                Dm = np.stack([F0 * S.deriv_mbh(n, steps.get(n)) for n in S.mbh_names]); Dl = np.stack([dM * S.base, dy * S.base])
                F = S.gram(np.concatenate([Dm, Dl])); C0, _, _ = FI.covariance(F); C, nd, c = cov_pinv(F); k = len(S.mbh_names)
                rel.append(float(LN10 * np.sqrt(C[k, k]))); rel_inv.append(float(LN10 * np.sqrt(C0[k, k]))); ndrop.append(nd)
                sy.append(float(np.sqrt(C[k + 1, k + 1]))); cond.append(float(c))
                h = F0 * S.base; snr_l.append(float(np.sqrt(S.ip(h, h)))); ok.append(1)
            except Exception as e:
                print("FAILED mass", i, repr(e), flush=True); rel.append(np.nan); rel_inv.append(np.nan); ndrop.append(-1); sy.append(np.nan); cond.append(np.nan); snr_l.append(np.nan); ok.append(0)
        json.dump(dict(config=a.config, draw=d, logM=[float(LOGM[i]) for i in idx], rel_dM=rel, rel_dM_plain_inverse=rel_inv, n_null_dropped=ndrop, sig_y=sy, cond=cond, snr_unlensed=snr_u, snr_lensed=snr_l, ok=ok,
                       cosi=float(np.cos(p["inclination"])), params={k: float(v) for k, v in p.items()}), open(fn, "w"))
        print(f"{a.config} draw {d}: {time.time() - t0:.0f}s  SNR_unl={snr_u:.0f}  min rel err {np.nanmin(rel):.2e}", flush=True)

else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {c: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.shards, f"fig3_{c}_d*.json")))] for c in CONFIGS}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True); lines = ["config  ndraws  Mcrit(median, -16%, +84%)  ours  |  paper"]
    summary = {}
    for c, runs in data.items():
        if not runs: continue
        lm = np.array(runs[0]["logM"]); R = np.array([r["rel_dM"] for r in runs], float)
        q16, q50, q84 = np.nanpercentile(R, [16, 50, 84], axis=0); M = 10 ** lm
        crit = []
        for row in R:                                                   # lowest M at which the relative error drops below 100%
            ok = np.where(row <= 1.0)[0]
            if len(ok) == 0 or ok[0] == 0: continue
            j = ok[0]; crit.append(10 ** np.interp(0.0, [np.log10(row[j]), np.log10(row[j - 1])], [lm[j], lm[j - 1]]))
        pm = PAPER_MCRIT[c]
        if crit:
            m50, m16, m84 = np.percentile(crit, [50, 16, 84]); summary[c] = (m50, m16, m84, len(crit))
            lines.append(f"{c:5s} {len(runs):5d}   {m50:9.3g} ({m16:.3g} .. {m84:.3g})   |  {pm[0]:.3g} (-{pm[1]:.2g} +{pm[2]:.2g})   ratio ours/paper = {m50 / pm[0]:.2f}")
        for ax_i, group in ((0, ("isco", "mr", "hm")), (1, ("hmm", "hm", "hmp"))):
            if c in group:
                ax = axs[ax_i]; ax.fill_between(M, q16, q84, color=COLOR[c], alpha=0.25, lw=0); ax.plot(M, q50, color=COLOR[c], lw=2, label=LABEL[c] + f"  [{len(runs)} draws]")
                ax.errorbar([pm[0]], [1.0], xerr=[[pm[1]], [pm[2]]], fmt="D", color=COLOR[c], mec="k", ms=6, capsize=3, zorder=6)
    for ax in axs:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e3, 1e9); ax.set_ylim(1e-4, 1); ax.grid(alpha=0.3, which="both"); ax.set_xlabel(r"$M_{Lz}$ [$M_\odot$]"); ax.legend(loc="lower left", fontsize=9)
    axs[0].set_ylabel(r"$\Delta M_{Lz}/M_{Lz}$")
    fig.suptitle("Our Fisher reproduction of Fig. 3, Caliskan et al. (arXiv:2206.02803): PM lens, y=0.1, M$_{Tz}$=2e6, q=1, z$_S$=1; lines = median, bands = 16-84%;"
                 "  diamonds = the paper's quoted critical lens masses (100% error), with 68% errors", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3_reproduction.png"), dpi=140); plt.close(fig)
    open(os.path.join(a.out, "fig3_mcrit_comparison.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    with open(os.path.join(a.out, "fig3_curves.csv"), "w") as fh:
        fh.write("config,logM,median,p16,p84,n\n")
        for c, runs in data.items():
            if not runs: continue
            R = np.array([r["rel_dM"] for r in runs], float); lo, md, hi = np.nanpercentile(R, [16, 50, 84], axis=0)
            for k, l in enumerate(runs[0]["logM"]): fh.write(f"{c},{l},{md[k]},{lo[k]},{hi[k]},{len(runs)}\n")
