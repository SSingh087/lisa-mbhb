"""Lensed MBHB waveforms: wave-optics amplification factors from GWLENS applied to
the LISA MBHB signal.

    lens = PointMassLens(mass_lz=1e6, y=0.5)          # exact point-mass F(w, y)
    sig  = LensedMBHB(params, lens)
    hp, hc, F = sig.polarizations()                    # lensed h+, hx and F(f) on sig.f
    tdi = sig.tdi()                                    # lensed A, E, T
    sig.snr(lensed=False), sig.snr()                   # unlensed vs lensed SNR

Lens classes only need an ``amplification(f)`` method returning the complex F(f)
for GW frequencies f [Hz] (detector frame, lens mass redshifted).
GWLENS (``gwlens`` on PyPI, >=0.2.2) is a regular dependency. To use a local GWLENS
checkout instead, set the GWLENS_PATH environment variable to its root directory.
"""
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

from . import inference, overlap, waveform
from .constants import C, G, MSUN, MPC

_GWLENS = os.environ.get("GWLENS_PATH")
if _GWLENS and os.path.isdir(_GWLENS) and _GWLENS not in sys.path:
    sys.path.insert(0, _GWLENS)     # explicit local checkout takes precedence over the installed gwlens

W_CONST = 8 * np.pi * G * MSUN / C**3     # w = W_CONST * M_Lz[Msun] * f


class PointMassLens:
    """Point-mass lens, exact F(w, y) (Takahashi & Nakamura 2003) from gwlens.lens.point.

    F is evaluated on a uniform w grid (spacing `dw`) and spline-interpolated.
    """

    def __init__(self, mass_lz, y, dw=0.05):
        self.mass_lz, self.y, self.dw = mass_lz, y, dw

    def w(self, f):
        return W_CONST * self.mass_lz * np.asarray(f)

    def amplification(self, f):
        from gwlens.lens.point import F_analytic
        w = self.w(f)
        grid = np.arange(max(w.min() - 2 * self.dw, 1e-3), w.max() + 2 * self.dw, self.dw)
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
    Used by lisa_mbhb.fisher, which needs fast on-axis lenses for finite-difference derivatives;
    the off-axis Solver2DLens family above (build_lens_set) is the general-purpose version.

    Dimensionless variables are the *point-mass* ones, so a vanishing halo recovers PointMassLens:
        w = W_CONST * mass_lz * f,   x = xi / xi0(mass_lz),   y in units of xi0(mass_lz).
    Potential psi(x) = ln x + psi0 * NFW(k x), with the repo's convention (scripts/Compositecontour.py,
    psi0_NFW = M_halo/M_PM when xi0 is built from M_halo):
        psi0 = (1 + zl) * halo_mass / mass_lz,   k = xi0 / r_s,
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
        self.xs = self._r_s() / self.xi0
        self.psi0 = (1 + zl) * halo_mass / mass_lz

    @classmethod
    def _d_eff(cls, zl, zs):
        E = lambda z: np.sqrt(cls.OM * (1 + z) ** 3 + cls.OL)
        dh = C / cls.H0
        dc = lambda a, b: dh * quad(lambda z: 1 / E(z), a, b)[0]
        d_l, d_s, d_ls = dc(0, zl) / (1 + zl), dc(0, zs) / (1 + zs), dc(zl, zs) / (1 + zs)
        return d_l * d_ls / ((1 + zl) * d_s)

    def _r_s(self):
        c200 = self.conc or 8 / (1 + self.zl) * (self.halo_mass * 0.7 / 1e14) ** (-0.13)
        h_z = self.H0 * np.sqrt(self.OM * (1 + self.zl) ** 3 + self.OL)
        rho_c = 3 * h_z ** 2 / (8 * np.pi * G)
        r200 = (3 * self.halo_mass * MSUN / (4 * np.pi * self.DELTA * rho_c)) ** (1 / 3)
        return r200 / c200

    def amplification(self, f):
        from gwlens.solver import solver1d as S1
        w = W_CONST * self.mass_lz * np.asarray(f)
        pot = S1.point_mass(1.0) + S1.nfw(k=1.0 / self.xs, psi0=self.psi0)
        with S1.Solver1D(pot, y=self.y, wmin=w.min() * 0.99, wmax=w.max() * 1.01, **S1.PE_SETTINGS) as s:
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
        with S1.Solver1D(pot, y=self.y, wmin=w.min() * 0.99, wmax=w.max() * 1.01, **S1.PE_SETTINGS) as s:
            return np.asarray(s.spectrum(w), dtype=complex)


def make_lens(spec, mass_lz, y, zs, zl_frac=0.5):
    """On-axis lens for lisa_mbhb.fisher.  spec: 'pm' | 'nfw:<M200 Msun>' (PM inside an NFW halo
    at zl = zl_frac * zs) | 'halo:<M200 Msun>' (the halo alone)."""
    if spec == "pm":
        return PointMassLens(mass_lz, y)
    kind, _, mh = spec.partition(":")
    if kind not in ("nfw", "halo") or not mh:
        raise ValueError(f"bad lens spec {spec!r}; use 'pm', 'nfw:<M200>' (point mass + halo) or 'halo:<M200>' (halo only)")
    return (PMInHaloLens1D if kind == "nfw" else HaloOnlyLens1D)(mass_lz, y, float(mh), zl_frac * zs, zs)


class LensedMBHB:
    """MBHB signal in LISA, optionally lensed.  `params` as in lisa_mbhb.waveform.generate
    plus psi, ecl_lon, ecl_lat (detector frame)."""

    def __init__(self, params, lens=None, cfg=None, channels=("A", "E", "T"), tdi=1):
        self.params, self.lens = params, lens
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
