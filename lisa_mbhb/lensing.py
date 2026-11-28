"""Lensed MBHB signals.  All lensing is computed by the GWLENS solvers
(``gwlens.solver.solver1d.Solver1D`` / ``gwlens.solver.solver2d.Solver2D``).

This module only does what GWLENS does not:
  1. units: GWLENS works in dimensionless w, x, y; here f [Hz] -> w = W_CONST * mass_lz * f, and a
     physical NFW halo (M200, z_L, z_S) -> GWLENS's (psi0, x_s)  (``nfw_units``);
  2. solver settings: every solver call uses ``solver_kw()`` (PE_SETTINGS, time_origin=None);
  3. the LISA signal: ``LensedMBHB`` applies F(f) to the TDI response.

    from gwlens.solver import solver1d as S1
    lens = Lens(S1.point_mass(1.0) + S1.sis(0.2), y=0.5, mass_lz=1e7)   # any GWLENS potential
    lens = get_lens("nfw", halo_mass=1e11, zs=2.0)                      # or a named one (defaults for the rest)
    lens = get_lens({"name": "pm", "y": 0.3})                           # or a config dict
    sig  = LensedMBHB(params, lens)                                     # lens may also be a name / config dict

Units: x and y are in units of xi0, the Einstein radius of a point mass `mass_lz` (redshifted,
Msun), for every lens, so different lenses compare point by point.  ``available_lenses()`` lists
the named lenses with their parameters and defaults.

GWLENS (``gwlens`` on PyPI) is a regular dependency.  To use a local GWLENS checkout instead,
set the GWLENS_PATH environment variable to its root directory.
"""
import inspect
import os
import sys

import numpy as np
from scipy.integrate import quad

from . import inference, overlap, waveform
from .constants import C, G, MSUN, MPC

_GWLENS = os.environ.get("GWLENS_PATH")
if _GWLENS and os.path.isdir(_GWLENS) and _GWLENS not in sys.path:
    sys.path.insert(0, _GWLENS)     # explicit local checkout takes precedence over the installed gwlens

W_CONST = 8 * np.pi * G * MSUN / C**3     # w = W_CONST * M_Lz[Msun] * f

# ------------------------------------------------------------------ units (physical -> GWLENS)
H0, OM, OL, DELTA = 70e3 / MPC, 0.3, 0.7, 200     # flat LCDM; halo overdensity 200 x rho_crit(z_L)


def _hubble(z):
    return H0 * np.sqrt(OM * (1 + z) ** 3 + OL)


def d_eff(zl, zs):
    """D_L D_LS / ((1 + z_L) D_S) [m], so that xi0^2 = 4 G d_eff M_Lz / c^2 with the redshifted mass."""
    dc = lambda a, b: C * quad(lambda z: 1 / _hubble(z), a, b)[0]
    d_l, d_s, d_ls = dc(0, zl) / (1 + zl), dc(0, zs) / (1 + zs), dc(zl, zs) / (1 + zs)
    return d_l * d_ls / ((1 + zl) * d_s)


def xi0(mass_lz, zl, zs):
    """Einstein radius [m] of a point mass of redshifted mass mass_lz [Msun]."""
    return np.sqrt(4 * G * d_eff(zl, zs) * mass_lz * MSUN) / C


def rho_crit(z):
    return 3 * _hubble(z) ** 2 / (8 * np.pi * G)


def m_of_c(c):
    return np.log1p(c) - c / (1 + c)


def c200_mean(halo_mass, zl):
    """Mean concentration-mass relation c_200 = 8/(1+z_L) (M200 h / 1e14 Msun)^-0.13."""
    return 8 / (1 + zl) * (halo_mass * 0.7 / 1e14) ** (-0.13)


def nfw_units(halo_mass, mass_lz, zl, zs, conc=None):
    """GWLENS NFW parameters of a halo of physical M200 = halo_mass [Msun], in the units of a point
    mass mass_lz:  psi(x) = psi0 * NFW(x / x_s),  GWLENS NFW(v) = [ln^2(v/2) + arccos^2(1/v)] / 2.

    x_s = r_s / xi0(mass_lz),  r_s = R_200 / c_200,
    psi0 = (1 + z_L) M200 / (m(c_200) mass_lz)   (projected mass of the halo = M200 g(v) / m(c)).
    Returns dict(psi0, xs, c200, m_c).
    """
    c = conc or c200_mean(halo_mass, zl)
    r200 = (3 * halo_mass * MSUN / (4 * np.pi * DELTA * rho_crit(zl))) ** (1 / 3)
    m_c = m_of_c(c)
    return dict(psi0=(1 + zl) * halo_mass / (m_c * mass_lz), xs=(r200 / c) / xi0(mass_lz, zl, zs), c200=c, m_c=m_c)


# ------------------------------------------------------------------ GWLENS solver
def solver_kw(S=None, **extra):
    """Keyword arguments for every GWLENS solver call: PE_SETTINGS of the solver module `S`
    (default: gwlens.solver.solver1d; pass solver2d for Solver2D) updated with `extra`.

    time_origin=None keeps every lens in the same time convention as the analytic point mass
    (gwlens.lens.point.F_analytic).  gwlens >= 0.2.2 defaults to time_origin='first_image', which
    gives each lens a different, lens-dependent time origin.
    """
    if S is None:
        from gwlens.solver import solver1d as S
    kw = dict(getattr(S, "PE_SETTINGS", {}), **extra)
    cls = getattr(S, "Solver1D", None) or getattr(S, "Solver2D")
    if "time_origin" in inspect.signature(cls.__init__).parameters:   # older gwlens: no option
        kw.setdefault("time_origin", None)
    return kw


def _is_gwlens_potential(obj):
    from gwlens.solver import solver1d as S1, solver2d as S2
    return isinstance(obj, (S1.Potential, S2.Lens))


class Lens:
    """A GWLENS lens: `potential` is a solver1d Potential (axisymmetric, on-axis source y) or a
    solver2d Lens (e.g. with external shear; y = (y1, y2) or y1).  `mass_lz` [Msun] sets the units:
    w = W_CONST * mass_lz * f, x and y in units of xi0(mass_lz).  `solver_opts` override solver_kw().
    `info` holds the physical parameters a named lens was built from."""

    def __init__(self, potential, y, mass_lz, info=None, **solver_opts):
        from gwlens.solver import solver2d as S2
        self.potential, self.y, self.mass_lz = potential, y, float(mass_lz)
        self.two_d = isinstance(potential, S2.Lens)
        self.info, self.solver_opts = dict(info or {}), solver_opts

    def w(self, f):
        return W_CONST * self.mass_lz * np.asarray(f, dtype=float)

    def amplification(self, f):
        """Complex F(f) for GW frequencies f [Hz] (detector frame)."""
        w = self.w(f)
        if self.two_d:
            from gwlens.solver import solver2d as S
            y = tuple(np.atleast_1d(self.y).astype(float)) if np.ndim(self.y) else (float(self.y), 0.0)
            Solver, kw = S.Solver2D, solver_kw(S, **{"y_radius": 0.0, **self.solver_opts})
        else:
            from gwlens.solver import solver1d as S
            y, Solver, kw = float(self.y), S.Solver1D, solver_kw(S, **self.solver_opts)
        with Solver(self.potential, y=y, wmin=w.min() * 0.99, wmax=w.max() * 1.01, **kw) as s:
            return np.asarray(s.spectrum(w), dtype=complex)

    def __repr__(self):
        return f"Lens({self.potential!r}, y={self.y}, mass_lz={self.mass_lz:g}{', ' + repr(self.info) if self.info else ''})"


# ------------------------------------------------------------------ named lenses (GWLENS potentials)
_REQUIRED = object()      # marks a parameter with no default


class LensType:
    """A named lens: `build(**params)` returns a Lens; `defaults` maps every parameter to its
    default value (or to _REQUIRED if it must be given)."""

    def __init__(self, name, build, defaults, doc=""):
        self.name, self.build, self.defaults, self.doc = name, build, dict(defaults), doc

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
        return self.build(**self.resolve(params))

    def __repr__(self):
        d = {k: ("<required>" if v is _REQUIRED else v) for k, v in self.defaults.items()}
        return f"LensType({self.name!r}, {d})"


LENSES = {}


def register_lens(name, defaults, doc="", aliases=()):
    """Register a named lens.  Decorate a function taking the parameters in `defaults` as keyword
    arguments and returning a Lens (or any object with ``amplification(f)``)."""
    def deco(build):
        t = LensType(name, build, defaults, doc or (build.__doc__ or "").strip().split("\n")[0])
        for n in (name, *aliases):
            LENSES[n] = t
        return build
    return deco


def _S1():
    from gwlens.solver import solver1d as S1
    return S1


@register_lens("pm", dict(mass_lz=1e7, y=1.0), aliases=("point", "point_mass"))
def _pm(mass_lz, y):
    """point mass"""
    return Lens(_S1().point_mass(1.0), y, mass_lz)


@register_lens("sis", dict(mass_lz=1e7, y=1.0))
def _sis(mass_lz, y):
    """singular isothermal sphere (mass_lz = mass inside its Einstein radius)"""
    return Lens(_S1().sis(1.0), y, mass_lz)


@register_lens("cis", dict(mass_lz=1e7, y=1.0, rc=0.05))
def _cis(mass_lz, y, rc):
    """cored isothermal sphere (core radius rc in units of xi0)"""
    return Lens(_S1().cis(1.0, rc), y, mass_lz)


_HALO_DEFAULTS = dict(mass_lz=1e7, y=1.0, halo_mass=1e11, zs=1.0, zl_frac=0.5, conc=None)


def _halo_lens(with_pm, mass_lz, y, halo_mass, zs, zl_frac, conc):
    S1, zl = _S1(), zl_frac * zs
    u = nfw_units(halo_mass, mass_lz, zl, zs, conc)
    pot = S1.nfw(k=1.0 / u["xs"], psi0=u["psi0"])
    if with_pm:
        pot = S1.point_mass(1.0) + pot
    return Lens(pot, y, mass_lz, info=dict(halo_mass=halo_mass, zl=zl, zs=zs, **u))


@register_lens("nfw", _HALO_DEFAULTS, aliases=("pm_nfw",))
def _nfw(**p):
    """point mass mass_lz at the centre of an NFW halo of M200 = halo_mass (z_L = zl_frac * zs)"""
    return _halo_lens(True, **p)


@register_lens("halo", _HALO_DEFAULTS, aliases=("nfw_only",))
def _halo(**p):
    """NFW halo alone, in the units of a point mass mass_lz (mass_lz is then only the unit)"""
    return _halo_lens(False, **p)


def available_lenses():
    """{name: {parameter: default}} for every named lens (aliases included)."""
    return {n: {k: ("<required>" if v is _REQUIRED else v) for k, v in t.defaults.items()}
            for n, t in sorted(LENSES.items())}


def get_lens(lens="pm", **params):
    """A lens from a name, a config dict, a GWLENS potential, or a ready lens object.

        get_lens("pm")                                    # all defaults
        get_lens("nfw", halo_mass=1e9, zs=2.0)            # override some parameters
        get_lens({"name": "halo", "halo_mass": 1e9})      # config dict (e.g. from JSON / YAML)
        get_lens("nfw:1e9", mass_lz=1e7, y=0.5)           # 'kind:<M200>' spec
        get_lens(S1.point_mass(1.0) + S1.sis(0.3), y=0.5, mass_lz=1e7)   # any GWLENS potential

    Keyword arguments override values in a config dict.  Unknown parameter names raise
    ValueError, so typos do not silently fall back to defaults.
    """
    if _is_gwlens_potential(lens):
        unknown = set(params) - {"y", "mass_lz"}
        if unknown:
            raise ValueError(f"a GWLENS potential takes only y and mass_lz, not {sorted(unknown)}")
        return Lens(lens, params.get("y", 1.0), params.get("mass_lz", 1e7))
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
        if sep:                                        # 'nfw:1e9' / 'halo:1e9'
            if kind not in ("nfw", "halo") or not mh:
                raise ValueError(f"bad lens spec {lens!r}; use 'pm', 'nfw:<M200>' or 'halo:<M200>'")
            name, params = kind, {"halo_mass": float(mh), **params}
    else:
        raise TypeError("lens must be a name, a config dict, a GWLENS potential or an object with amplification(f)")
    if name not in LENSES:
        raise ValueError(f"unknown lens {name!r}; available: {', '.join(sorted(LENSES))}")
    return LENSES[name](**params)


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
