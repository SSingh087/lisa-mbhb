"""Fisher study of the three lens models of lisa_mbhb.fisher_lenses (pml, nfw, pmlnfw; each with kappa and shear).  CLUSTER JOB.

    stepscan : F_kk vs finite-difference step for ONE lens parameter at one fiducial -> stepscan_<param><tag>.json
        python scripts/fisher_study.py stepscan --model pmlnfw --param log10_xs --source caliskan --m-ref 1e7 --y 0.1 --out DIR --tag _f0_pmlnfw
    point    : joint Fisher (11 binary + lens) for one source x model x m_ref over a list of (y, M200, kappa) -> fisher_study<tag>.json
        python scripts/fisher_study.py point --model pml --source z1_m1e6_q0.3 --m-ref 1e7 --y 0.3 1 --kappa 0 0.1 0.3 --steps S.json --ncpu 4 --out DIR --tag _id

Sources: 'caliskan' = Caliskan et al. 2023 Fig. 3 source (m1 = m2 = 5e5 Msun source frame, z = 1, M_Tz = 2e6, q = 1, non-spinning);
         'z<z>_m<Mtot>_q<q>' = the grid sources of the report (spins 0.3 / 0.1, demo angles).  For q = 1 the two spins are exactly
         degenerate, so spin2z is held fixed (removes the null direction, as in the Fig. 3 reproduction).
No priors.  Lens derivatives: steps from --steps (step sweep), capped at 0.02 / w_max, with an h vs h/2 check (deriv_conv).
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lisa_mbhb import demo, fisher as FI, inference as I, waveform as W  # noqa: E402
from lisa_mbhb.fisher_lenses import FisherLens  # noqa: E402


def make_source(spec, inc=0.9):
    """-> (detector-frame params, mbh_free, info dict)"""
    mm = re.fullmatch(r"z([\d.]+)_m([\d.e+-]+)_q([\d.]+)_d(\d+)", spec)
    if mm:    # z<z>_m<M_total>_q<q>_d<k>: source-frame total mass, random extrinsic draw k (seed 1000 + k), non-spinning
        z, m, q, k = float(mm.group(1)), float(mm.group(2)), float(mm.group(3)), int(mm.group(4)); rng = np.random.default_rng(1000 + k); m1 = m / (1 + q)
        p = demo.default_params(m1=m1, m2=q * m1, z=z, inclination=float(np.arccos(rng.uniform(-1, 1))), coa_phase=float(rng.uniform(0, 2 * np.pi)),
                                psi=float(rng.uniform(0, np.pi)), ecl_lon=float(rng.uniform(0, 2 * np.pi)), ecl_lat=float(np.arcsin(rng.uniform(-1, 1))),
                                spin1z=0.0, spin2z=0.0)
        return p, ([n for n in I.SAMPLED if n != "spin2z"] if q == 1 else list(I.SAMPLED)), dict(z=z, m_tot=m, q=q, inc=p["inclination"], source=spec, draw=k, seed=1000 + k)
    if spec.startswith("fig3draw"):    # Caliskan et al. Fig. 3 binary (M_Tz = 2e6, q = 1, z = 1, chi = 0) with random extrinsic draw k, seed 1000 + k
        k = int(spec[len("fig3draw"):]); rng = np.random.default_rng(1000 + k)
        p = demo.default_params(m1=5e5, m2=5e5, z=1.0, inclination=float(np.arccos(rng.uniform(-1, 1))), coa_phase=float(rng.uniform(0, 2 * np.pi)),
                                psi=float(rng.uniform(0, np.pi)), ecl_lon=float(rng.uniform(0, 2 * np.pi)), ecl_lat=float(np.arcsin(rng.uniform(-1, 1))),
                                spin1z=0.0, spin2z=0.0)
        return p, [n for n in I.SAMPLED if n != "spin2z"], dict(z=1.0, m_tot=1e6, q=1.0, inc=p["inclination"], source=spec, draw=k, seed=1000 + k)
    if spec == "caliskan":
        p = demo.default_params(m1=5e5, m2=5e5, z=1.0, inclination=inc, spin1z=0.0, spin2z=0.0)
        return p, [n for n in I.SAMPLED if n != "spin2z"], dict(z=1.0, m_tot=1e6, q=1.0, inc=inc, source="caliskan")
    z, m, q = spec.split("_"); z, m, q = float(z[1:]), float(m[1:]), float(q[1:])
    m1 = m / (1 + q)
    p = demo.default_params(m1=m1, m2=q * m1, z=z, inclination=inc)
    return p, ([n for n in I.SAMPLED if n != "spin2z"] if q == 1 else list(I.SAMPLED)), dict(z=z, m_tot=m, q=q, inc=inc, source=spec)


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["stepscan", "point", "cumul", "fullscan"])
ap.add_argument("--model", required=True, choices=["pml", "nfw", "pmlnfw", "unlensed"])
ap.add_argument("--source", required=True); ap.add_argument("--inc", type=float, default=0.9)
ap.add_argument("--m-ref", type=float, required=True, help="reference (BH, redshifted) mass: sets xi0, w; = truth BH mass")
ap.add_argument("--y", type=float, nargs="+", default=[1.0]); ap.add_argument("--m200", type=float, nargs="+", default=[1e11])
ap.add_argument("--kappa", type=float, nargs="+", default=[0.0])
ap.add_argument("--zl-frac", type=float, default=0.5)
ap.add_argument("--param", default=None); ap.add_argument("--mults", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
ap.add_argument("--steps", default=None, help="JSON {param: step} (binary + lens)")
ap.add_argument("--ncpu", type=int, default=1); ap.add_argument("--conv-tol", type=float, default=0.02)
ap.add_argument("--accuracy", default="full", choices=["pe", "full"], help="GWLENS solver accuracy (full ~1e-12, pe ~1e-7)")
ap.add_argument("--gamma", type=float, default=None, help="shear strength |gamma| (angle 22.5 deg); default lisa_mbhb.fisher_lenses.GAMMA_TRUE")
ap.add_argument("--nf-cumul", type=int, default=60, help="cumul mode: number of upper frequencies")
ap.add_argument("--mult", type=float, default=1.0, help="fullscan: multiplier of every lens step")
ap.add_argument("--stencil", type=int, default=2, choices=[2, 4], help="fullscan: 2-point (O(h^2)) or 5-point (O(h^4)) central difference")
ap.add_argument("--interp-tol", type=float, default=None, help="GWLENS interpolation tolerance (default: accuracy setting)")
ap.add_argument("--no-interp", action="store_true", help="GWLENS exact spectrum (no interpolation)")
ap.add_argument("--kfix", action="store_true", help="also compute the kappa-fixed Fisher with the h/2 convergence check (extra solves)")
ap.add_argument("--lens-mult", type=float, default=None, help="multiply every lens step (after the adaptive cap); default 30 for nfw/pmlnfw "
                "(full-Fisher step scan: inside the flat valley of every resolved eigenvalue), 1 for pml")
ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
# full-Fisher step scans (plots/fullscan_fine_absolute): the flat valley of the resolved eigenvalues spans x0.25..x630 at M_Lz=1e7 but
# shrinks with mass (truncation sets in earlier as w_max grows): ..x25 at 1e8, ..x6 at 3e8, ..x2.5 at 1e9 -> 30 below 1e8, 10*(1e8/M) above
LMULT = a.lens_mult if a.lens_mult is not None else ((30.0 if a.m_ref < 1e8 else 10.0 * 1e8 / a.m_ref) if a.model in ("nfw", "pmlnfw") else 1.0)
HALF = a.model in ("nfw", "pmlnfw")     # store F at h/2 too: analyse() flags eigen-directions that are not converged

p, mbh_free, info = make_source(a.source, a.inc)
S = FI.SourceDerivs(p, W.WaveformConfig(tc_ref=p["tc"]), mbh_free=mbh_free)
if a.model == "unlensed":   # unlensed binary Fisher (11 parameters), independent of the lens: one matrix per source
    st = dict(FI.STEP_MBH)
    if a.steps and os.path.exists(a.steps): st.update(json.load(open(a.steps)))
    t0 = time.time(); D = np.stack([S.deriv_mbh(n, st.get(n)) for n in S.mbh_names]); F = S.gram(D)
    th0 = dict(zip(I.SAMPLED, I.params_to_theta(p))); dd = np.sqrt(np.diag(F)); ev = np.linalg.eigvalsh(0.5 * (F + F.T) / np.outer(dd, dd))
    snr = float(np.sqrt(S.ip(S.base, S.base)))
    row = dict(ok=1, y=0.0, m200=None, kappa=0.0, vals={}, binary={n: float(th0[n]) for n in I.SAMPLED}, names=list(S.mbh_names),
               F=F.tolist(), cond=float(ev.max() / ev.min()), eig_min=float(ev.min()), lens_steps={}, snr_unlensed=snr, snr_lensed=snr, seconds=time.time() - t0)
    out = os.path.join(a.out, f"fisher_study{a.tag}.json")
    json.dump(dict(args=vars(a), info=info, model="unlensed", m_ref=a.m_ref, zl=None, mbh_free=mbh_free, steps=st, rows=[row], seconds=row["seconds"]), open(out + ".tmp", "w"))
    os.replace(out + ".tmp", out); print("wrote", out, f"SNR {snr:.0f} cond {row['cond']:.2e}"); sys.exit(0)
SOPT = dict(threads=1, accuracy=a.accuracy)
if a.interp_tol: SOPT["interp_tol"] = a.interp_tol
if a.no_interp: SOPT["interpolate"] = False
Lm = FisherLens(a.model, a.m_ref, info["z"], a.zl_frac, solver_opts=SOPT)
steps = {**FI.STEP_MBH, **Lm.steps}
if a.steps and os.path.exists(a.steps):
    steps.update({k: v for k, v in json.load(open(a.steps)).items()})
print(f"{a.mode} {a.model} {a.source} m_ref={a.m_ref:g}: grid {S.f.size} pts, SNR_unl {np.sqrt(S.ip(S.base, S.base)):.1f}, "
      f"w_max {float(Lm.w(S.f).max()):.3g}", flush=True)

GKW = {} if a.gamma is None else dict(gamma=(a.gamma * np.cos(np.pi / 4), a.gamma * np.sin(np.pi / 4)))

if a.mode == "cumul":   # Fisher matrix accumulated up to increasing upper frequencies: where in the band the information comes from
    v = Lm.default_values(y=a.y[0], m200=a.m200[0], kappa=a.kappa[0], **GKW); F0 = FI.lens_amp(Lm, v, S.f)
    hs = {n: h * LMULT for n, h in S.lens_steps(Lm, v, Lm.names, steps, True).items()}
    Du = np.stack([S.deriv_mbh(n, steps.get(n)) for n in S.mbh_names])
    D = np.concatenate([F0 * Du, np.stack([S.dF_model(Lm, v, n, hs[n]) * S.base for n in Lm.names])])
    Dh = np.concatenate([F0 * Du, np.stack([S.dF_model(Lm, v, n, 0.5 * hs[n]) * S.base for n in Lm.names])]) if HALF else None
    order = np.argsort(S.f); fs_ = S.f[order]
    cut = np.unique(np.searchsorted(fs_, np.geomspace(fs_[0] * 1.01, fs_[-1], a.nf_cumul), side="right"))
    def acc(DD):
        out, Fa, prev = [], np.zeros((len(DD), len(DD))), 0
        for i in cut:
            sl = order[prev:i]
            for ic in range(DD.shape[1]):
                Dc = DD[:, ic, sl]; Fa = Fa + np.real((Dc * S.W[ic][sl]) @ Dc.conj().T)
            out.append(Fa.tolist()); prev = i
        return out
    th0 = dict(zip(I.SAMPLED, I.params_to_theta(p)))
    res = dict(model=a.model, info=info, m_ref=a.m_ref, vals=v, binary={n: float(th0[n]) for n in I.SAMPLED}, names=S.mbh_names + Lm.names,
               names_unlensed=list(S.mbh_names), f_max=[float(fs_[i - 1]) for i in cut], F=acc(D), F_unlensed=acc(Du),
               lens_mult=LMULT, lens_steps=hs, **({"F_half": acc(Dh)} if HALF else {}))
    out = os.path.join(a.out, f"cumul{a.tag}.json"); json.dump(res, open(out + ".tmp", "w")); os.replace(out + ".tmp", out); print("wrote", out); sys.exit(0)

if a.mode == "fullscan":   # full Fisher (11 binary + lens) with every lens step scaled by --mult, 2- or 5-point stencil
    t0 = time.time(); v = Lm.default_values(y=a.y[0], m200=a.m200[0], kappa=a.kappa[0], **GKW); F0 = FI.lens_amp(Lm, v, S.f)
    hs = S.lens_steps(Lm, v, Lm.names, steps, True)
    def dlens(n, h):
        f = lambda x: FI.lens_amp(Lm, {**v, n: x}, S.f); x0 = v[n]
        if a.stencil == 2: return (f(x0 + h) - f(x0 - h)) / (2 * h)
        return (-f(x0 + 2 * h) + 8 * f(x0 + h) - 8 * f(x0 - h) + f(x0 - 2 * h)) / (12 * h)
    D = np.concatenate([np.stack([F0 * S.deriv_mbh(n, steps.get(n)) for n in S.mbh_names]),
                        np.stack([dlens(n, hs[n] * a.mult) * S.base for n in Lm.names])])
    F = S.gram(D); names = S.mbh_names + Lm.names; th0 = dict(zip(I.SAMPLED, I.params_to_theta(p)))
    res = dict(model=a.model, info=info, m_ref=a.m_ref, y=a.y[0], m200=a.m200[0], kappa=a.kappa[0], mult=a.mult, stencil=a.stencil, solver=SOPT, lens_steps={n: hs[n] * a.mult for n in Lm.names},
               names=names, F=F.tolist(), vals=v, binary={n: float(th0[n]) for n in I.SAMPLED}, seconds=time.time() - t0)
    out = os.path.join(a.out, f"fullscan{a.tag}.json"); json.dump(res, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    print("wrote", out, f"({res['seconds']:.0f}s)"); sys.exit(0)

if a.mode == "stepscan":
    out = os.path.join(a.out, f"stepscan_{a.param}{a.tag}.json")
    v = Lm.default_values(y=a.y[0], m200=a.m200[0], kappa=a.kappa[0]); base = Lm.steps[a.param]; fkk = []
    for m in a.mults:
        t0 = time.time()
        try:
            d = S.dF_model(Lm, v, a.param, base * m) * S.base; fkk.append(S.ip(d, d))
        except Exception as e:
            fkk.append(float("nan")); print("fail", m, repr(e))
        print(f"{a.param} step {base * m:.3g}: F_kk {fkk[-1]:.8e} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(param=a.param, base_step=base, mults=a.mults, fkk=fkk, model=a.model, source=a.source, m_ref=a.m_ref,
                   vals=v), open(out + ".tmp", "w")); os.replace(out + ".tmp", out); sys.exit(0)

# ---------------- point mode ----------------
for n in S.mbh_names: S.deriv_mbh(n, steps.get(n))            # cache the binary derivatives before forking
truths = [(y, m2, k) for y in a.y for m2 in (a.m200 if a.model != "pml" else [None]) for k in a.kappa]
theta0 = dict(zip(I.SAMPLED, I.params_to_theta(p)))


def one(t):
    y, m200, k = t; t0 = time.time()
    v = Lm.default_values(y=y, m200=m200 or 1e11, kappa=k, **GKW)
    try:   # kappa free (shows the exact mass-sheet degeneracy) and kappa fixed (standard; its h vs h/2 check is the meaningful one)
        r = S.fisher_model(Lm, v, steps=steps, priors=None, adaptive=True, richardson=HALF, lens_mult=LMULT)
        rk = (S.fisher_model(Lm, v, steps=steps, free=[n for n in Lm.names if n != "kappa"], priors=None, adaptive=True, richardson=True, lens_mult=LMULT)
              if a.kfix else None)
    except Exception as e:
        return dict(ok=0, y=y, m200=m200, kappa=k, error=repr(e))
    out = dict(ok=1, y=y, m200=m200, kappa=k, vals=v, binary={n: float(theta0[n]) for n in I.SAMPLED},
               names=r["names"], F=r["F"].tolist(), cond=r["cond"], eig_min=r["eig_min"],
               lens_steps=r["lens_steps"], lens_mult=LMULT, snr_unlensed=r["snr_unlensed"], snr_lensed=r["snr_lensed"], seconds=time.time() - t0)
    if HALF: out["F_half"] = np.asarray(r["F_like_half"]).tolist()
    if rk is not None:
        out.update(names_kfix=rk["names"], F_kfix=rk["F"].tolist(), sigma_kfix={n: float(x) for n, x in rk["sigma"].items()},
                   cond_kfix=rk["cond"], deriv_conv=rk["deriv_conv"], deriv_flag=int(not rk["deriv_conv"] <= a.conv_tol))
    return out

t0 = time.time()
if a.ncpu > 1:
    with ProcessPoolExecutor(a.ncpu, mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(one, truths))
else:
    rows = [one(t) for t in truths]
for r in rows:
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k in ("ok", "y", "m200", "kappa", "cond", "cond_kfix", "deriv_conv", "seconds", "error")}, flush=True)
out = os.path.join(a.out, f"fisher_study{a.tag}.json")
json.dump(dict(args=vars(a), info=info, model=a.model, m_ref=a.m_ref, zl=Lm.zl, mbh_free=mbh_free, steps=steps, rows=rows,
               seconds=time.time() - t0), open(out + ".tmp", "w"))
os.replace(out + ".tmp", out)
print("wrote", out, f"({time.time() - t0:.0f}s)")
