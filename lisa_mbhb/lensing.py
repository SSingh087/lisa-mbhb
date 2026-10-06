"""Lensed MBHB waveforms: wave-optics amplification factors from GWLENS applied to
the LISA MBHB signal.

    lens = PointMassLens(mass_lz=1e6, y=0.5)          # exact point-mass F(w, y)
    sig  = LensedMBHB(params, lens)
    hp, hc, F = sig.polarizations()                    # lensed h+, hx and F(f) on sig.f
    tdi = sig.tdi()                                    # lensed A, E, T
    sig.snr(lensed=False), sig.snr()                   # unlensed vs lensed SNR

Lens classes only need an ``amplification(f)`` method returning the complex F(f)
for GW frequencies f [Hz] (detector frame, lens mass redshifted).
GWLENS is found via the GWLENS_PATH environment variable (default ~/GWLENS).
"""
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

from . import inference, waveform
from .constants import C, G, MSUN

_GWLENS = os.environ.get("GWLENS_PATH", os.path.expanduser("~/GWLENS"))
if os.path.isdir(_GWLENS) and _GWLENS not in sys.path:
    sys.path.append(_GWLENS)

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

    def mismatch(self):
        """1 - overlap between lensed and unlensed signals (0 = indistinguishable)."""
        a, b = self.tdi(True), self.tdi(False)
        return 1 - self.model.inner(a, b) / np.sqrt(self.model.inner(a, a) * self.model.inner(b, b))
