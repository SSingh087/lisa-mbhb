"""Lensed MBHB waveforms: wave-optics amplification factors from GWLENS applied to
the LISA MBHB signal.

    lens = get_lens("pm", mass_lz=1e6, y=0.5)         # by name; unset parameters use defaults
    lens = get_lens({"name": "nfw", "halo_mass": 1e11, "zs": 2.0})   # or from a config dict
    sig  = LensedMBHB(params, lens)                    # lens may also be a name or a config dict
    hp, hc, F = sig.polarizations()                    # lensed h+, hx and F(f) on sig.f
    tdi = sig.tdi()                                    # lensed A, E, T
    sig.snr(lensed=False), sig.snr()                   # unlensed vs lensed SNR

``available_lenses()`` lists every lens name with its parameters and defaults.
Lens classes only need an ``amplification(f)`` method returning the complex F(f)
for GW frequencies f [Hz] (detector frame, lens mass redshifted).
GWLENS (``gwlens`` on PyPI, >=0.2.2) is a regular dependency. To use a local GWLENS
checkout instead, set the GWLENS_PATH environment variable to its root directory.
"""
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, RectBivariateSpline

from . import inference, overlap, waveform
from .constants import C, G, MSUN, MPC

_GWLENS = os.environ.get("GWLENS_PATH")
if _GWLENS and os.path.isdir(_GWLENS) and _GWLENS not in sys.path:
    sys.path.insert(0, _GWLENS)     # explicit local checkout takes precedence over the installed gwlens

W_CONST = 8 * np.pi * G * MSUN / C**3     # w = W_CONST * M_Lz[Msun] * f


def solver_kw(S=None, **extra):
    """Keyword arguments for every GWLENS solver call: PE_SETTINGS of the solver module `S`
    (default: gwlens.solver.solver1d; pass solver2d for Solver2D) updated with `extra`.

    time_origin=None keeps the solvers in the same time convention as the analytic point mass
    (gwlens.lens.point.F_analytic).  gwlens >= 0.2.2 defaults to time_origin='first_image', which
    would give solver lenses (nfw, halo, sis, cis) a different, lens-dependent time origin from 'pm'.
    """
    import inspect
    if S is None:
        from gwlens.solver import solver1d as S
    kw = dict(getattr(S, "PE_SETTINGS", {}), **extra)
    cls = getattr(S, "Solver1D", None) or getattr(S, "Solver2D")
    if "time_origin" in inspect.signature(cls.__init__).parameters:   # older gwlens: no option
        kw.setdefault("time_origin", None)
    return kw


class PointMassLens:
    """Point-mass lens, exact F(w, y) (Takahashi & Nakamura 2003) from gwlens.lens.point.

    F is evaluated on log-spaced knots below w = 1 and a uniform grid (spacing `dw`) above,
    then spline-interpolated.
    """

    def __init__(self, mass_lz, y, dw=0.05):
        self.mass_lz, self.y, self.dw = mass_lz, y, dw

    def w(self, f):
        return W_CONST * self.mass_lz * np.asarray(f)

    def amplification(self, f):
        from gwlens.lens.point import F_analytic
        w = self.w(f)
        lo, hi = 0.9 * w.min(), w.max() + 2 * self.dw
        # log-spaced knots below w = 1 (F varies on scale ~w there), uniform spacing dw above
        grid = np.arange(1.0, max(hi, 1.0 + 2 * self.dw), self.dw)
        if lo < 1.0:
            n_log = max(16, int(np.ceil(60 * np.log10(1.0 / lo))))
            grid = np.concatenate([np.geomspace(lo, 1.0, n_log, endpoint=False), grid])
        vals = np.array([F_analytic(wi, self.y) for wi in grid])
        return CubicSpline(grid, vals.real)(w) + 1j * CubicSpline(grid, vals.imag)(w)


class LookupTableLens:
    """Any lens with a GWLENS HDF5 table (keys w_values, y_values, results[w, y, (re, im)]),
    e.g. NFW or composite lenses made with `gwlens_interpolate`.

    w_of_f : callable f[Hz] -> dimensionless frequency w for this lens
             (e.g. gwlens.tools.halo.NFW(...).Dimensionless_Frequency).
    """

    def __init__(self, table_path, w_of_f, y):
        import h5py
        with h5py.File(table_path, "r") as h:
            w, yv, res = h["w_values"][:], h["y_values"][:], h["results"][:]
        self.re = RectBivariateSpline(w, yv, res[:, :, 0], kx=3, ky=3)
        self.im = RectBivariateSpline(w, yv, res[:, :, 1], kx=3, ky=3)
        self.wlim, self.ylim = (w[0], w[-1]), (yv[0], yv[-1])
        self.w_of_f, self.y = w_of_f, y

    def amplification(self, f):
        w = np.clip(self.w_of_f(np.asarray(f)), *self.wlim)
        y = float(np.clip(self.y, *self.ylim))
        return self.re.ev(w, y) + 1j * self.im.ev(w, y)


class PMInHaloLens1D:
    """On-axis point mass embedded in an NFW halo (GWLENS 1D solver, axisymmetric).
    Used by lisa_mbhb.fisher, which needs fast on-axis lenses for finite-difference derivatives.

    Dimensionless variables are the *point-mass* ones, so a vanishing halo recovers PointMassLens:
        w = W_CONST * mass_lz * f,   x = xi / xi0(mass_lz),   y in units of xi0(mass_lz).
    Potential psi(x) = ln x + psi0 * NFW(k x), with gwlens' NFW(v) = [ln^2(v/2) + arccos^2(1/v)] / 2,
    whose deflection is psi0 * g(v) / x.  Matching the projected NFW mass 4 pi rho_s r_s^3 g(v) with
    4 pi rho_s r_s^3 = M_200 / m(c), m(c) = ln(1 + c) - c / (1 + c), gives
        psi0 = (1 + zl) * halo_mass / (m(c_200) * mass_lz),   k = xi0 / r_s,
        xi0 = sqrt(4 G d_eff mass_lz) / c,       d_eff = D_L D_LS / ((1+zl) D_S).
    r_s = R_200 / c_200 with 200*rho_crit(zl) and c_200 = 8/(1+zl) (M h/1e14)^-0.13 (as in Compositecontour.py).
    ASSUMPTIONS: point mass exactly at the halo centre; mass_lz is the redshifted PM mass; halo_mass is the
    physical M_200 in Msun; flat LCDM H0=70, Om=0.3.
    """
    H0, OM, OL, DELTA = 70e3 / MPC, 0.3, 0.7, 200

    def __init__(self, mass_lz, y, halo_mass, zl, zs, conc=None):
        self.mass_lz, self.y, self.halo_mass, self.zl, self.zs = mass_lz, y, halo_mass, zl, zs
        self.conc = conc
        self.d_eff = self._d_eff(zl, zs)
        self.xi0 = np.sqrt(4 * G * self.d_eff * mass_lz * MSUN) / C
        self.c200 = self.conc or 8 / (1 + zl) * (halo_mass * 0.7 / 1e14) ** (-0.13)
        self.xs = self._r_s() / self.xi0
        self.m_c = np.log1p(self.c200) - self.c200 / (1 + self.c200)
        self.psi0 = (1 + zl) * halo_mass / (self.m_c * mass_lz)

    @classmethod
    def _d_eff(cls, zl, zs):
        E = lambda z: np.sqrt(cls.OM * (1 + z) ** 3 + cls.OL)
        dh = C / cls.H0
        dc = lambda a, b: dh * quad(lambda z: 1 / E(z), a, b)[0]
        d_l, d_s, d_ls = dc(0, zl) / (1 + zl), dc(0, zs) / (1 + zs), dc(zl, zs) / (1 + zs)
        return d_l * d_ls / ((1 + zl) * d_s)

    def _r_s(self):
        c200 = self.c200
        h_z = self.H0 * np.sqrt(self.OM * (1 + self.zl) ** 3 + self.OL)
        rho_c = 3 * h_z ** 2 / (8 * np.pi * G)
        r200 = (3 * self.halo_mass * MSUN / (4 * np.pi * self.DELTA * rho_c)) ** (1 / 3)
        return r200 / c200

    def amplification(self, f):
        from gwlens.solver import solver1d as S1
        w = W_CONST * self.mass_lz * np.asarray(f)
        pot = S1.point_mass(1.0) + S1.nfw(k=1.0 / self.xs, psi0=self.psi0)
        with S1.Solver1D(pot, y=self.y, wmin=w.min() * 0.99, wmax=w.max() * 1.01, **solver_kw()) as s:
            return np.asarray(s.spectrum(w), dtype=complex)


class HaloOnlyLens1D(PMInHaloLens1D):
    """The NFW halo ALONE (no central point mass), in exactly the same normalisation as
    PMInHaloLens1D (w, x, y and psi0 defined through the reference mass `mass_lz`), so it compares
    point by point with the composite at the same (mass_lz, y).  `mass_lz` is then only a unit
    choice (xi0 = Einstein radius of a point mass mass_lz)."""

    def amplification(self, f):
        from gwlens.solver import solver1d as S1
        w = W_CONST * self.mass_lz * np.asarray(f)
        pot = S1.nfw(k=1.0 / self.xs, psi0=self.psi0)
        with S1.Solver1D(pot, y=self.y, wmin=w.min() * 0.99, wmax=w.max() * 1.01, **solver_kw()) as s:
            return np.asarray(s.spectrum(w), dtype=complex)


class IsothermalLens1D:
    """Singular (rc = 0) or cored (rc > 0) isothermal sphere, GWLENS 1D solver, on axis.
    Units: xi0 is the SIS Einstein radius and mass_lz the redshifted mass inside it, so
    w = W_CONST * mass_lz * f as for the point mass; rc is the core radius in units of xi0."""

    def __init__(self, mass_lz, y, rc=0.0):
        self.mass_lz, self.y, self.rc = mass_lz, y, rc

    def amplification(self, f):
        from gwlens.solver import solver1d as S1
        w = W_CONST * self.mass_lz * np.asarray(f)
        pot = S1.sis(1.0) if self.rc == 0 else S1.cis(1.0, self.rc)
        with S1.Solver1D(pot, y=self.y, wmin=w.min() * 0.99, wmax=w.max() * 1.01, **solver_kw()) as s:
            return np.asarray(s.spectrum(w), dtype=complex)


# ------------------------------------------------------------------ lens registry
_REQUIRED = object()      # marks a parameter with no default


class LensType:
    """A named lens: `factory(**params)` builds the lens object; `defaults` maps every parameter
    to its default value (or to _REQUIRED if it must be given)."""

    def __init__(self, name, factory, defaults, doc=""):
        self.name, self.factory, self.defaults, self.doc = name, factory, dict(defaults), doc

    def resolve(self, params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"lens '{self.name}' has no parameter(s) {sorted(unknown)}; "
                             f"parameters: {list(self.defaults)}")
        out = {**self.defaults, **params}
        missing = [k for k, v in out.items() if v is _REQUIRED]
        if missing:
            raise ValueError(f"lens '{self.name}' needs {missing}")
        return out

    def __call__(self, **params):
        return self.factory(**self.resolve(params))

    def __repr__(self):
        d = {k: ("<required>" if v is _REQUIRED else v) for k, v in self.defaults.items()}
        return f"LensType({self.name!r}, {d})"


LENSES = {}


def register_lens(name, defaults, doc="", aliases=()):
    """Register a lens under `name` (and `aliases`).  Use as a decorator on a function or class
    taking the parameters in `defaults` as keyword arguments and returning an object with
    ``amplification(f)``."""
    def deco(factory):
        t = LensType(name, factory, defaults, doc or (factory.__doc__ or "").strip().split("\n")[0])
        for n in (name, *aliases):
            LENSES[n] = t
        return factory
    return deco


_HALO_DEFAULTS = dict(mass_lz=1e7, y=1.0, halo_mass=1e11, zs=1.0, zl_frac=0.5, conc=None)


def _halo(cls):
    def build(mass_lz, y, halo_mass, zs, zl_frac, conc):
        return cls(mass_lz, y, halo_mass, zl_frac * zs, zs, conc)
    return build


register_lens("pm", dict(mass_lz=1e7, y=1.0, dw=0.05), "point mass, exact F(w, y)",
              aliases=("point", "point_mass"))(PointMassLens)
register_lens("nfw", _HALO_DEFAULTS, "point mass at the centre of an NFW halo (M200 = halo_mass, zl = zl_frac * zs)",
              aliases=("pm_nfw",))(_halo(PMInHaloLens1D))
register_lens("halo", _HALO_DEFAULTS, "NFW halo alone, in the point-mass units of mass_lz",
              aliases=("nfw_only",))(_halo(HaloOnlyLens1D))
register_lens("sis", dict(mass_lz=1e7, y=1.0), "singular isothermal sphere")(
    lambda mass_lz, y: IsothermalLens1D(mass_lz, y, 0.0))
register_lens("cis", dict(mass_lz=1e7, y=1.0, rc=0.05), "cored isothermal sphere (rc in units of xi0)")(IsothermalLens1D)
register_lens("lookup", dict(table_path=_REQUIRED, mass_lz=1e7, y=1.0),
              "interpolated GWLENS HDF5 table, w = W_CONST * mass_lz * f")(
    lambda table_path, mass_lz, y: LookupTableLens(table_path, lambda f: W_CONST * mass_lz * np.asarray(f), y))


def available_lenses():
    """{name: {parameter: default}} for every registered lens (aliases included)."""
    return {n: {k: ("<required>" if v is _REQUIRED else v) for k, v in t.defaults.items()}
            for n, t in sorted(LENSES.items())}


def get_lens(lens="pm", **params):
    """Build a lens from a name, a config dict or a ready lens object.

        get_lens("pm")                                    # all defaults
        get_lens("nfw", halo_mass=1e9, zs=2.0)            # override some parameters
        get_lens({"name": "halo", "halo_mass": 1e9})      # config dict (e.g. from JSON / YAML)
        get_lens("nfw:1e9", mass_lz=1e7, y=0.5)           # legacy 'kind:<M200>' spec

    Keyword arguments override values in a config dict.  Unknown parameter names raise
    ValueError, so typos do not silently fall back to defaults.
    """
    if hasattr(lens, "amplification"):
        if params:
            raise ValueError("parameters cannot be applied to an already built lens object")
        return lens
    if isinstance(lens, dict):
        cfg = dict(lens)
        name = cfg.pop("name", None) or cfg.pop("model", None)
        if name is None:
            raise ValueError("lens config dict needs a 'name' key, e.g. {'name': 'pm', 'y': 0.5}")
        params = {**cfg, **params}
    elif isinstance(lens, str):
        name = lens
        kind, sep, mh = lens.partition(":")
        if sep:                                        # legacy 'nfw:1e9' / 'halo:1e9'
            if kind not in ("nfw", "halo") or not mh:
                raise ValueError(f"bad lens spec {lens!r}; use 'pm', 'nfw:<M200>' or 'halo:<M200>'")
            name, params = kind, {"halo_mass": float(mh), **params}
    else:
        raise TypeError("lens must be a name, a config dict or an object with amplification(f)")
    if name not in LENSES:
        raise ValueError(f"unknown lens {name!r}; available: {', '.join(sorted(LENSES))}")
    return LENSES[name](**params)


def make_lens(spec, mass_lz, y, zs, zl_frac=0.5):
    """On-axis lens for lisa_mbhb.fisher (kept for backward compatibility; see get_lens).
    spec: 'pm' | 'nfw:<M200 Msun>' (PM inside an NFW halo) | 'halo:<M200 Msun>' (halo alone)."""
    if spec in ("pm", "point", "point_mass"):
        return get_lens("pm", mass_lz=mass_lz, y=y)
    if ":" not in spec:
        raise ValueError(f"bad lens spec {spec!r}; use 'pm', 'nfw:<M200>' (point mass + halo) or 'halo:<M200>' (halo only)")
    return get_lens(spec, mass_lz=mass_lz, y=y, zs=zs, zl_frac=zl_frac)


class LensedMBHB:
    """MBHB signal in LISA, optionally lensed.  `params` as in lisa_mbhb.waveform.generate
    plus psi, ecl_lon, ecl_lat (detector frame).  `lens`: None (unlensed), a lens object,
    a lens name ('pm', 'nfw', ...) or a config dict {'name': ..., <parameters>} (see get_lens)."""

    def __init__(self, params, lens=None, cfg=None, channels=("A", "E", "T"), tdi=1):
        self.params = params
        self.lens = None if lens is None else get_lens(lens)
        self.cfg = cfg or waveform.WaveformConfig(tc_ref=params["tc"])
        self.f = waveform.design_grid(params, self.cfg)
        self.model = inference.TDIModel(self.f, self.cfg, channels, tdi)
        self._F = None

    def amplification(self):
        if self._F is None:
            self._F = (np.ones_like(self.f, dtype=complex) if self.lens is None
                       else self.lens.amplification(self.f))
        return self._F

    def polarizations(self, lensed=True):
        hp, hc, _ = waveform.generate(self.params, self.f, self.cfg)
        F = self.amplification()
        return (hp * F, hc * F, F) if lensed else (hp, hc, np.ones_like(F))

    def tdi(self, lensed=True):
        if not lensed:
            return self.model(self.params)
        return self.model(self.params, amplification=lambda f: self.amplification())

    def snr(self, lensed=True):
        s = self.tdi(lensed)
        return float(np.sqrt(self.model.inner(s, s)))

    def mismatch(self, max_dt=0.0, n_dt=1):
        """1 - overlap(lensed, unlensed), maximised over a global phase and, if
        max_dt > 0, a time shift in [-max_dt, max_dt] seconds (n_dt trial values)."""
        return overlap.mismatch(self.model, self.tdi(True), self.tdi(False), max_dt, n_dt)
