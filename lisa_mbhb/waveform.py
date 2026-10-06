"""MBHB frequency-domain waveforms from PyCBC on an adaptive, non-uniform grid.

A single-uniform-df FFT grid would need ~1e7 points for a year-long signal;
instead we evaluate the waveform at a fixed non-uniform frequency array whose
spacing resolves the phase evolution (including a time-shift margin).
"""
from dataclasses import dataclass
import numpy as np
from pycbc.types import Array
from pycbc.waveform import get_fd_waveform_sequence

from .constants import MTSUN, YEAR


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def time_to_merger_newtonian(f, mchirp):
    """Leading-order time to merger [s] from GW frequency f [Hz]; mchirp in Msun (detector frame)."""
    return 5.0 / 256.0 * (mchirp * MTSUN) ** (-5.0 / 3) * (np.pi * f) ** (-8.0 / 3)


def frequency_from_time_to_merger(tau, mchirp):
    return (5.0 / 256.0 * (mchirp * MTSUN) ** (-5.0 / 3) / tau) ** (3.0 / 8) / np.pi


def source_to_detector(m1, m2, z, cosmology=None):
    """Source-frame masses + redshift -> detector-frame masses [Msun], d_L [Mpc]."""
    if cosmology is None:
        from astropy.cosmology import Planck18 as cosmology
    dl = float(cosmology.luminosity_distance(z).value)
    return m1 * (1 + z), m2 * (1 + z), dl


@dataclass
class WaveformConfig:
    approximant: str = "IMRPhenomD"   # IMRPhenomD | IMRPhenomXHM | IMRPhenomHM ...
    t_obs: float = 1.0 * YEAR          # data stretch ending at merger/after [s]
    f_max: float = 0.1                 # Hz (capped by the merger-ringdown cutoff)
    time_margin: float = 2e5           # |tc - tc_ref| the grid must resolve [s]
    mc_margin: float = 0.6             # grid is designed for mchirp * mc_margin
    phase_step: float = 0.5            # max phase change between grid points [rad]
    tc_ref: float = 0.0                # reference merger time removed from the FD phase


def design_grid(ref_params, cfg):
    """Fixed frequency grid adapted to the reference system."""
    m1, m2 = ref_params["mass1"], ref_params["mass2"]
    mc = chirp_mass(m1, m2) * cfg.mc_margin
    f_lo = max(frequency_from_time_to_merger(cfg.t_obs, mc), 1e-5)
    f_hi = min(cfg.f_max, 0.25 / ((m1 + m2) * MTSUN))
    ff = np.logspace(np.log10(f_lo), np.log10(f_hi), 400000)
    tau = time_to_merger_newtonian(ff, mc) + cfg.time_margin
    df = np.minimum(cfg.phase_step / (2 * np.pi * tau), ff / 200.0)
    n = np.concatenate([[0.0], np.cumsum(0.5 * (1 / df[1:] + 1 / df[:-1]) * np.diff(ff))])
    return np.interp(np.arange(0, n[-1], 1.0), n, ff)


def generate(params, f, cfg):
    """Return hp(f), hc(f) [1/Hz] and the time t(f) [s] at which each frequency is emitted.

    params keys: mass1, mass2 (detector frame, Msun), spin1z, spin2z,
    distance (Mpc), inclination, coa_phase, tc (s).
    """
    kw = dict(approximant=cfg.approximant, mass1=params["mass1"], mass2=params["mass2"],
              spin1z=params.get("spin1z", 0.0), spin2z=params.get("spin2z", 0.0),
              distance=params["distance"], sample_points=Array(f))
    hp, hc = get_fd_waveform_sequence(inclination=params["inclination"],
                                      coa_phase=params.get("coa_phase", 0.0), **kw)
    hp, hc = hp.numpy(), hc.numpy()
    # time-frequency map from the (face-on, 22-dominated) phase: t = -(1/2pi) dtheta/df
    ref, _ = get_fd_waveform_sequence(inclination=0.0, coa_phase=0.0, **kw)
    theta = np.unwrap(np.angle(ref.numpy()))
    t_f = -np.gradient(theta, f) / (2 * np.pi)
    # PyCBC/LAL place the peak near t=0 -> shift to absolute merger time
    t_f = np.minimum(np.maximum.accumulate(t_f), 0.0) + params["tc"]   # clean post-merger noise
    shift = np.exp(-2j * np.pi * f * (params["tc"] - cfg.tc_ref))
    return hp * shift, hc * shift, t_f
