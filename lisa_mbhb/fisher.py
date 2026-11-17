"""Fisher matrix for a LENSED LISA MBHB: MBH parameters + lens parameters.

The lensed TDI signal is  h(f; theta, lam) = F(f; lam) * h_unlensed(f; theta)   (F is a pure
multiplicative amplification factor), so every derivative is cheap:

    d h / d theta_i = F0 * d h_unl / d theta_i        (unlensed derivatives computed ONCE per source)
    d h / d lam_k   = (dF / d lam_k) * h_unl0          (only the lens amplification is re-evaluated)

Parameters
    MBH  : inference.SAMPLED  (log10_mchirp, q, spin1z, spin2z, log10_dl, cos_inc, coa_phase, psi,
                               ecl_lon, sin_ecl_lat, tc)
    lens : pm   -> log10_mlz, y           (lens redshift is NOT a parameter: zl = zl_frac * zs, fixed)
           nfw  -> log10_mlz, y, log10_mhalo  (on-axis PM inside an NFW halo, lensing.CompositeLens)
           halo -> log10_mhalo, y             (the NFW halo alone; mass_lz only fixes the length unit)

    S = SourceDerivs(params)                       # one-off: 2*11 waveform evaluations
    r = S.fisher(LensSpec("pm"), mass_lz=1e7, y=1.0, zs=1.0)
    r["sigma"]["y"], r["sigma"]["log10_mlz"], r["cond"], r["snr_lensed"]

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
    def fisher(self, lens, mass_lz, y, zs, zl_frac=0.5, steps=None, lens_free=None):
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
