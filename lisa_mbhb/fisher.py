"""Fisher matrix for a LENSED LISA MBHB: MBH parameters + lens parameters.

The lensed TDI signal is  h(f; theta, lam) = F(f; lam) * h_unlensed(f; theta)   (F is a pure
multiplicative amplification factor), so every derivative is cheap:

    d h / d theta_i = F0 * d h_unl / d theta_i        (unlensed derivatives computed ONCE per source)
    d h / d lam_k   = (dF / d lam_k) * h_unl0          (only the lens amplification is re-evaluated)

Parameters
    MBH  : inference.SAMPLED  (log10_mchirp, q, spin1z, spin2z, log10_dl, cos_inc, coa_phase, psi,
                               ecl_lon, sin_ecl_lat, tc)
    lens : pm   -> log10_mlz, y           (lens redshift is NOT a parameter: zl = zl_frac * zs, fixed)
           nfw  -> log10_mlz, y, log10_mhalo  (on-axis PM inside an NFW halo, lensing.PMInHaloLens1D)
           halo -> log10_mhalo, y             (the NFW halo alone; mass_lz only fixes the length unit)

    S = SourceDerivs(params)                       # one-off: 2*11 waveform evaluations
    r = S.fisher(LensSpec("pm"), mass_lz=1e7, y=1.0, zs=1.0)
    r["sigma"]["y"], r["sigma"]["log10_mlz"], r["cond"], r["snr_lensed"]

Arbitrary parameterised lens models (fisher_lenses.FisherLens: pm, nfw, pmlnfw with the free lens parameters of that model):
    r = S.fisher_model(FisherLens("pmlnfw", 1e7, zs=1), vals, free=..., priors=None|'default'|{name: (mu, sig)})
    free, pri = conc_case(plens, "c_fixed"|"c_prior"|"c_free");  schur(F, names, keep) -> marginalised Fisher.

Covariance = normalised inverse (same recipe as EMRI_TDE_*/fisher/fisher_cov.py): invert the diagonally
normalised matrix, because raw entries span >20 orders of magnitude.
Finite-difference steps: STEP_* below are starting values; find the stable ones with
scripts/fisher_step_scan.py (plateau of F_kk vs step) and pass them as `steps=`.
"""
import numpy as np

from . import inference as I, lensing as L, waveform
from .inference import SAMPLED

STEP_MBH = dict(log10_mchirp=1e-5, q=1e-4, spin1z=1e-4, spin2z=1e-4, log10_dl=1e-3, cos_inc=1e-3,
                coa_phase=1e-3, psi=1e-3, ecl_lon=1e-4, sin_ecl_lat=1e-4, tc=1.0)
STEP_LENS = dict(log10_mlz=1e-3, y=1e-3, log10_mhalo=1e-3)
# hard parameter bounds -> one-sided difference if theta +/- h would leave them
BOUNDS = dict(q=(0.02, 1.0), spin1z=(-0.99, 0.99), spin2z=(-0.99, 0.99), cos_inc=(-1.0, 1.0),
              sin_ecl_lat=(-1.0, 1.0), y=(0.0, np.inf))


def _conv(s1, s2, names):
    """max |sigma_h / sigma_{h/2} - 1| over `names` (Richardson convergence of the lens derivatives)."""
    return float(max((abs(s1[n] / s2[n] - 1) if s2[n] > 0 else np.inf) for n in names)) if names else np.nan


def covariance(F):
    """(C, cond_normalised, cond_raw) from a Fisher matrix, via diagonal normalisation."""
    Fs = 0.5 * (np.asarray(F, float) + np.asarray(F, float).T)
    d = 1.0 / np.sqrt(np.clip(np.diag(Fs), 1e-300, None))
    Fn = Fs * d[:, None] * d[None, :]
    try:
        Cn = np.linalg.inv(Fn)
    except np.linalg.LinAlgError:
        Cn = np.linalg.pinv(Fn)
    return Cn * d[:, None] * d[None, :], float(np.linalg.cond(Fn)), float(np.linalg.cond(Fs))


def _diff(fn, x0, h, bounds=(-np.inf, np.inf)):
    """Central difference of fn at x0 (one-sided if a bound is closer than h)."""
    lo, hi = bounds
    if x0 - h < lo:
        return (fn(x0 + h) - fn(x0)) / h
    if x0 + h > hi:
        return (fn(x0) - fn(x0 - h)) / h
    return (fn(x0 + h) - fn(x0 - h)) / (2 * h)


class LensSpec:
    """'pm' or 'nfw:<M200 Msun>'.  Lens parameters differentiated: see `names`."""

    def __init__(self, spec):
        self.spec = spec
        self.kind, _, mh = spec.partition(":")
        if self.kind not in ("pm", "nfw", "halo") or (self.kind != "pm" and not mh):
            raise ValueError(f"bad lens spec {spec!r}")
        self.halo_mass = float(mh) if mh else None
        # halo-only: mass_lz is just the unit choice (fixed), so only M_halo and y are lens parameters
        self.names = {"pm": ["log10_mlz", "y"], "nfw": ["log10_mlz", "y", "log10_mhalo"], "halo": ["log10_mhalo", "y"]}[self.kind]

    def values(self, mass_lz, y):
        v = dict(log10_mlz=np.log10(mass_lz), y=float(y))
        if self.kind != "pm":
            v["log10_mhalo"] = np.log10(self.halo_mass)
        return v

    def amplification(self, vals, f, zs, zl_frac=0.5):
        spec = "pm" if self.kind == "pm" else f"{self.kind}:{float(10 ** vals['log10_mhalo'])!r}"
        return L.make_lens(spec, 10 ** vals["log10_mlz"], vals["y"], zs, zl_frac).amplification(f)


CONC_CASES = ("c_fixed", "c_prior", "c_free")


def _stack(ch, model):
    return np.stack([ch[c] for c in model.channels])


class SourceDerivs:
    """Unlensed TDI signal + its derivatives w.r.t. the MBH parameters for ONE source.

    params : detector-frame dict (demo.default_params).  mbh_free: subset of SAMPLED to include.
    """

    def __init__(self, params, cfg=None, channels=("A", "E", "T"), tdi=1, mbh_free=None):
        self.params = params
        self.cfg = cfg or waveform.WaveformConfig(tc_ref=params["tc"])
        self.f = waveform.design_grid(params, self.cfg)
        self.model = I.TDIModel(self.f, self.cfg, channels, tdi)
        self.theta0 = dict(zip(SAMPLED, I.params_to_theta(params)))
        self.mbh_names = list(mbh_free or SAMPLED)
        self.base = _stack(self.model(params), self.model)           # unlensed signal (nch, nf)
        self.W = np.stack([4 * self.model.w / self.model.psd[c] for c in self.model.channels])
        self._D = {}                                                  # (name, step) -> deriv

    # ---------- inner products ----------
    def ip(self, a, b):
        return float(np.sum(self.W * np.real(a * np.conj(b))))

    def gram(self, D):
        """Fisher block from derivative stack D (npar, nch, nf)."""
        F = np.zeros((len(D), len(D)))
        for ic in range(D.shape[1]):
            Dc = D[:, ic, :]
            F += np.real((Dc * self.W[ic]) @ Dc.conj().T)
        return F

    # ---------- unlensed derivative wrt one MBH parameter ----------
    def deriv_mbh(self, name, step=None):
        step = STEP_MBH[name] if step is None else step
        key = (name, step)
        if key not in self._D:
            def fn(x):
                th = dict(self.theta0); th[name] = x
                return _stack(self.model(I.theta_to_params([th[k] for k in SAMPLED])), self.model)
            self._D[key] = _diff(fn, self.theta0[name], step, BOUNDS.get(name, (-np.inf, np.inf)))
        return self._D[key]

    def unlensed_at(self, theta):
        """Unlensed TDI stack (nch, nf) at a theta dict (missing SAMPLED names -> theta0)."""
        th = {**self.theta0, **theta}
        return _stack(self.model(I.theta_to_params([th[k] for k in SAMPLED])), self.model)

    # ---------- lens amplification and its derivative ----------
    def amp(self, lens, vals, zs, zl_frac):
        return lens.amplification(vals, self.f, zs, zl_frac)

    def dF(self, lens, vals, name, zs, zl_frac, step=None):
        step = STEP_LENS[name] if step is None else step
        def fn(x):
            v = dict(vals); v[name] = x
            return self.amp(lens, v, zs, zl_frac)
        return _diff(fn, vals[name], step, BOUNDS.get(name, (-np.inf, np.inf)))

    # ---------- full Fisher ----------
    def fisher(self, lens, mass_lz, y, zs, zl_frac=0.5, steps=None, lens_free=None, adaptive=False):
        """Joint Fisher over [MBH params, lens params].  steps: dict name -> step (defaults STEP_*).

        Returns dict with names, F, cov, sigma{name}, cond, snr_unlensed, snr_lensed, snr_diff,
        sigma_lensknown{mbh name} (lens fixed to truth), sigma_unlensed{mbh name} (no lens in model).
        """
        steps = steps or {}
        lens = LensSpec(lens) if isinstance(lens, str) else lens
        vals = lens.values(mass_lz, y)
        F0 = self.amp(lens, vals, zs, zl_frac)
        lnames = list(lens_free or lens.names)
        Dm = np.stack([F0 * self.deriv_mbh(n, steps.get(n)) for n in self.mbh_names])
        if adaptive:                                                  # opt-in; default reproduces old shards
            wmax = L.W_CONST * mass_lz * float(np.max(self.f))
            steps = {**steps, **{n: adaptive_step(n, steps.get(n, STEP_LENS.get(n, 1e-3)), wmax) for n in lnames}}
        Dl = np.stack([self.dF(lens, vals, n, zs, zl_frac, steps.get(n)) * self.base for n in lnames])
        D = np.concatenate([Dm, Dl])
        names = self.mbh_names + lnames
        F = self.gram(D)
        C, cond, cond_raw = covariance(F)
        sig = dict(zip(names, np.sqrt(np.clip(np.diag(C), 0, None))))
        nm = len(self.mbh_names)
        Cm, _, _ = covariance(F[:nm, :nm])                           # lens known
        Cu, _, _ = covariance(self.gram(np.stack([self.deriv_mbh(n, steps.get(n)) for n in self.mbh_names])))
        h = F0 * self.base
        return dict(names=names, F=F, cov=C, sigma=sig, cond=cond, cond_raw=cond_raw,
                    eig_min=float(np.linalg.eigvalsh(0.5 * (F + F.T) / np.sqrt(np.outer(np.diag(F), np.diag(F)))).min()),
                    snr_unlensed=float(np.sqrt(self.ip(self.base, self.base))),
                    snr_lensed=float(np.sqrt(self.ip(h, h))),
                    snr_diff=float(np.sqrt(self.ip(h - self.base, h - self.base))),
                    sigma_lensknown=dict(zip(self.mbh_names, np.sqrt(np.clip(np.diag(Cm), 0, None)))),
                    sigma_unlensed=dict(zip(self.mbh_names, np.sqrt(np.clip(np.diag(Cu), 0, None)))))

    # ---------- single diagonal element, for the step scan ----------
    def fkk(self, name, step, lens, mass_lz, y, zs, zl_frac=0.5):
        lens = LensSpec(lens) if isinstance(lens, str) else lens
        vals = lens.values(mass_lz, y)
        if name in SAMPLED:
            d = self.amp(lens, vals, zs, zl_frac) * self.deriv_mbh(name, step)
        else:
            d = self.dF(lens, vals, name, zs, zl_frac, step) * self.base
        return self.ip(d, d)

    # ---------- arbitrary parameterised lens model (fisher_lenses.FisherLens) ----------
    def dF_model(self, plens, vals, name, step=None):
        """dF/d(name) for a FisherLens at vals, central difference (one-sided at plens.bounds)."""
        step = plens.steps.get(name, 1e-3) if step is None else step
        def fn(x):
            v = dict(vals); v[name] = x
            return lens_amp(plens, v, self.f)
        return _diff(fn, vals[name], step, plens.bounds.get(name, (-np.inf, np.inf)))

    def lens_steps(self, plens, vals, names, steps=None, adaptive=True):
        """Lens FD steps: steps[name] or plens.steps, capped (adaptive) at c / w_max_eff (see adaptive_step)."""
        steps = steps or {}
        h = {n: steps.get(n) or plens.steps.get(n, 1e-3) for n in names}
        if adaptive:
            a = 10 ** (vals["log10_mbh"] - np.log10(plens.m_ref_z)) if "log10_mbh" in vals else 1.0
            wmax = float(np.max(plens.w(self.f))) * max(1.0, a)
            h = {n: adaptive_step(n, v, wmax) for n, v in h.items()}
        return h

    def fisher_model(self, plens, vals, steps=None, free=None, priors=None, F0=None, adaptive=True, richardson=True, lens_mult=1.0):
        """Joint Fisher over [MBH params, free lens params of `plens`] at lens values `vals` (all plens.names).

        free   : lens names to vary (default plens.names); the others stay fixed at vals (e.g. 'c fixed').
        priors : None | 'default' (plens.priors) | dict name -> (mean, sigma) or sigma; adds 1/sigma^2 to the
                 diagonal of every PRESENT parameter (MBH or lens).  F0: precomputed amplification at vals.
        adaptive: lens steps capped at c / w_max (adaptive_step).  richardson: also differentiate with h/2 and
                 store deriv_conv = max |sigma_h / sigma_{h/2} - 1| over the lens parameters (> 0.02: not converged).
        lens_mult: every lens step (after the adaptive cap) is multiplied by this (full-Fisher step scan: 30 for NFW models).
        Same output as fisher(); also F_like (no priors), the prior dict used, lens_steps (as used), deriv_conv.
        """
        steps = steps or {}
        vals = complete_vals(plens, vals)
        F0 = lens_amp(plens, vals, self.f) if F0 is None else F0
        lnames = [n for n in plens.names if free is None or n in free]
        hs = self.lens_steps(plens, vals, lnames, steps, adaptive)
        Dm = np.stack([F0 * self.deriv_mbh(n, steps.get(n)) for n in self.mbh_names])
        names = self.mbh_names + lnames
        pr = plens.priors if isinstance(priors, str) and priors == "default" else priors

        def run(mult):
            D = np.concatenate([Dm, np.stack([self.dF_model(plens, vals, n, hs[n] * mult) * self.base for n in lnames])]) if lnames else Dm
            return self.fisher_summary(self.gram(D), names, pr, F0, steps)
        r = run(lens_mult)
        r.update(lens_steps={n: h * lens_mult for n, h in hs.items()}, deriv_conv=np.nan)
        if richardson and lnames:
            r2 = run(0.5 * lens_mult)
            r["deriv_conv"] = _conv(r["sigma"], r2["sigma"], lnames)
            r["F_like_half"] = r2["F_like"]                           # lets fisher_model_cases test each case separately
        return r

    def fisher_model_cases(self, plens, vals, cases=CONC_CASES, steps=None, free=None, F0=None, adaptive=True, richardson=True):
        """{case: fisher_model result} for the concentration cases, sharing ONE set of lens derivatives
        (deriv_conv is evaluated PER CASE from the h and h/2 matrices: a near-singular c-free matrix must not flag the
        well-conditioned c-fixed / c-prior cases)."""
        full = self.fisher_model(plens, vals, steps, free, None, F0, adaptive, richardson)
        out = {}
        for c in cases:
            fr, pr = conc_case(plens, c, [n for n in full["names"] if n not in self.mbh_names])
            if fr == full["names"][len(self.mbh_names):] and not pr:
                out[c] = full; continue
            keep = self.mbh_names + fr
            r = dict(self.fisher_summary(full["F_like"], full["names"], pr, full["F0"], steps, keep=keep), lens_steps=full["lens_steps"], deriv_conv=np.nan)
            if "F_like_half" in full:
                r2 = self.fisher_summary(full["F_like_half"], full["names"], pr, full["F0"], steps, keep=keep)
                r["deriv_conv"] = _conv(r["sigma"], r2["sigma"], fr)
            out[c] = r
        return out

    def fisher_summary(self, F_like, names, priors=None, F0=None, steps=None, keep=None):
        """cov / sigma / cond / SNRs from a likelihood Fisher (optionally restricted to `keep`) plus diagonal priors."""
        steps = steps or {}
        if keep is not None:
            idx = [names.index(n) for n in keep]; F_like, names = F_like[np.ix_(idx, idx)], list(keep)
        pr = {k: (v[1] if np.ndim(v) else v) for k, v in (priors or {}).items() if k in names and v is not None}
        F = F_like + np.diag([1.0 / pr[n] ** 2 if n in pr else 0.0 for n in names])
        C, cond, cond_raw = covariance(F)
        nm = len(self.mbh_names)
        Cm, _, _ = covariance(F[:nm, :nm])
        Cu, _, _ = covariance(self.gram(np.stack([self.deriv_mbh(n, steps.get(n)) for n in self.mbh_names])))
        F0 = np.ones_like(self.base[0]) if F0 is None else F0
        h = F0 * self.base
        d = np.sqrt(np.clip(np.diag(F), 1e-300, None))
        return dict(names=names, F=F, F_like=F_like, priors=pr, cov=C, cond=cond, cond_raw=cond_raw, F0=F0,
                    sigma=dict(zip(names, np.sqrt(np.clip(np.diag(C), 0, None)))),
                    eig_min=float(np.linalg.eigvalsh(0.5 * (F + F.T) / np.outer(d, d)).min()),
                    snr_unlensed=float(np.sqrt(self.ip(self.base, self.base))),
                    snr_lensed=float(np.sqrt(self.ip(h, h))),
                    snr_diff=float(np.sqrt(self.ip(h - self.base, h - self.base))),
                    sigma_lensknown=dict(zip(self.mbh_names, np.sqrt(np.clip(np.diag(Cm), 0, None)))),
                    sigma_unlensed=dict(zip(self.mbh_names, np.sqrt(np.clip(np.diag(Cu), 0, None)))))


ADAPT_C = dict(y=0.02, phi_y=0.02, gamma=0.02, log10_mlz=0.02 / np.log(10), log10_mhalo=0.02 / np.log(10),
               log10_mbh=0.02 / np.log(10), log10_m200=0.02 / np.log(10), log10_fbh=0.02 / np.log(10),
               log10_c200=0.02 / np.log(10),
               gamma1=0.02, gamma2=0.02, kappa=0.02, log10_psi0=0.02 / np.log(10), log10_xs=0.02 / np.log(10))   # fisher_lenses models


def adaptive_step(name, h, wmax):
    """min(h, c / w_max) for parameters that rescale w or move y (F oscillates on scale ~1/w); others unchanged."""
    return min(h, ADAPT_C[name] / max(wmax, 1e-30)) if name in ADAPT_C else h


def complete_vals(plens, vals):
    """{name: value} for exactly plens.names; missing ones filled from plens.default_values(**known)."""
    known = {k: float(v) for k, v in vals.items() if k in plens.names}
    if len(known) < len(plens.names):
        known = {**plens.default_values(**known), **known}
    return {n: float(known[n]) for n in plens.names}


def lens_amp(plens, vals, f):
    """plens.amplification with only its free names passed (fixed / M-c-tied parameters stay with plens)."""
    return plens.amplification({n: vals[n] for n in plens.names}, f)


def conc_case(plens, case, free=None):
    """(free, priors) for SourceDerivs.fisher_model under a concentration treatment:
    'c_fixed' (log10_c200 not a parameter), 'c_prior' (free + plens.priors['log10_c200']), 'c_free' (free, no prior).
    Models without log10_c200 ignore the case."""
    free = list(free or plens.names)
    if "log10_c200" not in plens.names or case == "c_free":
        return free, None
    if case == "c_fixed":
        return [n for n in free if n != "log10_c200"], None
    if case == "c_prior":
        return free, {"log10_c200": plens.priors["log10_c200"]}
    raise ValueError(f"bad concentration case {case!r}; use one of {CONC_CASES}")


def schur(F, names, keep):
    """Effective (marginalised) Fisher on `keep` = Schur complement of F over the other parameters."""
    k = [names.index(n) for n in keep]; o = [i for i in range(len(names)) if i not in k]
    if not o:
        return F[np.ix_(k, k)]
    Coo, _, _ = covariance(F[np.ix_(o, o)])
    return F[np.ix_(k, k)] - F[np.ix_(k, o)] @ Coo @ F[np.ix_(o, k)]
