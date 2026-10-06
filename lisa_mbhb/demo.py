"""Default example source shared by the scripts, tests and notebook."""
from .constants import YEAR, MTSUN
from . import waveform


def default_params(m1=5e5, m2=2.5e5, z=1.0, **over):
    """Source-frame masses [Msun] and redshift -> detector-frame parameter dict."""
    d1, d2, dl = waveform.source_to_detector(m1, m2, z)
    p = dict(mass1=d1, mass2=d2, spin1z=0.3, spin2z=0.1, distance=dl, inclination=0.9,
             coa_phase=0.3, psi=0.7, ecl_lon=1.2, ecl_lat=0.4, tc=0.9 * YEAR)
    p.update(over)
    return p


def shift_range(mass_lz, step=1.0, cap=2500):
    """Time-shift search window [s] and number of trial shifts for a lens of mass_lz."""
    max_dt = 40 * MTSUN * mass_lz
    return max_dt, int(min(cap, 2 * max_dt / step + 1))
