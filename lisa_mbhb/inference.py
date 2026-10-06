"""Injection and likelihood for LISA MBHBs in the TDI A, E, T channels."""
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import CubicSpline

from . import noise, response, waveform
from .waveform import WaveformConfig

# parameters sampled in recovery (order matters)
SAMPLED = ["log10_mchirp", "q", "spin1z", "spin2z", "log10_dl", "cos_inc",
           "coa_phase", "psi", "ecl_lon", "sin_ecl_lat", "tc"]


def theta_to_params(theta):
    """Sampled vector -> waveform / response parameter dict (detector frame)."""
    th = dict(zip(SAMPLED, theta))
    mc, q = 10 ** th["log10_mchirp"], th["q"]
    m1 = mc * (1 + q) ** 0.2 / q**0.6
    return dict(mass1=m1, mass2=q * m1, spin1z=th["spin1z"], spin2z=th["spin2z"],
                distance=10 ** th["log10_dl"], inclination=np.arccos(th["cos_inc"]),
                coa_phase=th["coa_phase"], psi=th["psi"], ecl_lon=th["ecl_lon"],
                ecl_lat=np.arcsin(th["sin_ecl_lat"]), tc=th["tc"])


def params_to_theta(p):
    m1, m2 = p["mass1"], p["mass2"]
    return np.array([np.log10(waveform.chirp_mass(m1, m2)), m2 / m1, p["spin1z"], p["spin2z"],
                     np.log10(p["distance"]), np.cos(p["inclination"]), p["coa_phase"], p["psi"],
                     p["ecl_lon"], np.sin(p["ecl_lat"]), p["tc"]])


class TDIModel:
    """Maps parameters -> complex TDI channels on a fixed frequency grid."""

    def __init__(self, f, cfg, channels=("A", "E", "T"), tdi=1, n_coarse=1500):
        self.f, self.cfg, self.channels, self.tdi = f, cfg, tuple(channels), tdi
        self.fc = np.geomspace(f[0], f[-1], n_coarse)
        # trapezoid weights
        w = np.zeros_like(f)
        w[1:] += 0.5 * np.diff(f)
        w[:-1] += 0.5 * np.diff(f)
        self.w = w
        psds = {"A": noise.psd_tdi_A(f, tdi=tdi), "E": noise.psd_tdi_A(f, tdi=tdi),
                "T": noise.psd_tdi_T(f, tdi=tdi)}
        self.psd = {c: psds[c] for c in self.channels}
        self._idx = {"A": 0, "E": 1, "T": 2}

    def __call__(self, p, amplification=None):
        """TDI channels for parameters p.  `amplification`: optional callable F(f)
        (complex lensing amplification factor) applied to h+ and hx."""
        hp, hc, t_f = waveform.generate(p, self.f, self.cfg)
        if amplification is not None:
            F = amplification(self.f)
            hp, hc = hp * F, hc * F
        t_c = np.interp(self.fc, self.f, t_f)
        ones = np.ones_like(self.fc)
        zeros = np.zeros_like(self.fc)
        args = (p["ecl_lon"], p["ecl_lat"], p["psi"])
        Fp = response.tdi_aet(self.fc, t_c, ones, zeros, *args, tdi=self.tdi)
        Fc = response.tdi_aet(self.fc, t_c, zeros, ones, *args, tdi=self.tdi)
        out = {}
        lf, lfc = np.log(self.f), np.log(self.fc)
        for c in self.channels:
            i = self._idx[c]
            fp = CubicSpline(lfc, Fp[i])(lf)
            fcr = CubicSpline(lfc, Fc[i])(lf)
            out[c] = fp * hp + fcr * hc
        return out

    def inner(self, a, b):
        """Noise-weighted inner product summed over channels, 4 Re int a b* / S df."""
        s = 0.0
        for c in self.channels:
            s += 4 * np.sum(self.w * np.real(a[c] * np.conj(b[c])) / self.psd[c])
        return s


@dataclass
class Injection:
    model: TDIModel
    params: dict
    data: dict
    signal: dict
    snr: float
    snr_channels: dict = field(default_factory=dict)


def make_injection(params, cfg=None, channels=("A", "E", "T"), tdi=1, add_noise=False, seed=None):
    """Simulate data containing an MBHB.  `params` are detector-frame (see waveform)."""
    cfg = cfg or WaveformConfig(tc_ref=params["tc"])
    f = waveform.design_grid(params, cfg)
    model = TDIModel(f, cfg, channels, tdi)
    sig = model(params)
    data = {c: sig[c].copy() for c in model.channels}
    if add_noise:
        rng = np.random.default_rng(seed)
        for c in model.channels:
            sigma = np.sqrt(model.psd[c] / (4 * model.w))
            data[c] += sigma * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
    snr_c = {c: float(np.sqrt(4 * np.sum(model.w * np.abs(sig[c]) ** 2 / model.psd[c])))
             for c in model.channels}
    return Injection(model, params, data, sig, float(np.sqrt(model.inner(sig, sig))), snr_c)


class Likelihood:
    """ln L(theta) = -1/2 (d - h | d - h) (up to a constant)."""

    def __init__(self, inj):
        self.inj = inj
        self.model = inj.model
        self.dd = self.model.inner(inj.data, inj.data)

    def logl_params(self, p):
        h = self.model(p)
        d = self.inj.data
        dh = self.model.inner(d, h)
        hh = self.model.inner(h, h)
        return dh - 0.5 * hh   # = ln L - ln L(noise-only)

    def __call__(self, theta):
        try:
            return self.logl_params(theta_to_params(theta))
        except Exception:
            return -np.inf
