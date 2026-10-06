"""Frequency-domain LISA response: single links -> TDI (X,Y,Z) -> (A,E,T).

Finite-arm-length, time-dependent response in the stationary-phase /
"rigid adiabatic" approximation (Cutler 1998; Marsat et al. 2021; Katz et al. 2022).
Each frequency is evaluated with the constellation at the time t(f) at which
the signal passes through that frequency.
"""
import numpy as np
from .constants import C, LISA_ARM
from .orbits import positions


def sky_basis(lon, lat, psi):
    """Propagation vector k and polarisation tensors e+, ex (ecliptic frame)."""
    cl, sl, cb, sb = np.cos(lon), np.sin(lon), np.cos(lat), np.sin(lat)
    n = np.array([cb * cl, cb * sl, sb])            # towards the source
    e_lon = np.array([-sl, cl, 0.0])
    e_lat = np.array([-sb * cl, -sb * sl, cb])
    cp, sp = np.cos(psi), np.sin(psi)
    p = cp * e_lat + sp * e_lon
    q = -sp * e_lat + cp * e_lon
    ep = np.outer(p, p) - np.outer(q, q)
    ex = np.outer(p, q) + np.outer(q, p)
    return -n, ep, ex


def single_link(f, t_f, hp, hc, lon, lat, psi, r, s):
    """Fractional-frequency response of the link s -> r, y_rs(f)."""
    k, ep, ex = sky_basis(lon, lat, psi)
    pos = positions(t_f)                    # (3, N, 3)
    pr, ps = pos[r], pos[s]
    n = (pr - ps) / np.linalg.norm(pr - ps, axis=-1, keepdims=True)  # photon direction
    kn = n @ k
    nPn = (np.einsum('ni,ij,nj->n', n, ep, n) * hp
           + np.einsum('ni,ij,nj->n', n, ex, n) * hc)
    x = np.pi * f * LISA_ARM / C
    kp = (pr + ps) @ k
    phase = np.exp(-1j * np.pi * f * (LISA_ARM + kp) / C * 1.0)
    # NB: k.p has units of metres -> divide by c
    return 1j * x * np.sinc(x * (1 - kn) / np.pi) * phase * nPn


def tdi_xyz(f, t_f, hp, hc, lon, lat, psi, tdi=1):
    """First-generation Michelson X, Y, Z (equal-arm); tdi=2 adds the TDI-2 factor."""
    y = {}
    for r, s in [(2, 0), (0, 2), (1, 0), (0, 1), (1, 2), (2, 1)]:
        y[(r, s)] = single_link(f, t_f, hp, hc, lon, lat, psi, r, s)
    D = np.exp(-2j * np.pi * f * LISA_ARM / C)
    fac = 1 - D**2
    if tdi == 2:
        fac = fac * (1 - D**4)
    # craft labels 1,2,3 -> 0,1,2 ; y31 = y[(2,0)] etc.
    X = fac * ((y[(2, 0)] + D * y[(0, 2)]) - (y[(1, 0)] + D * y[(0, 1)]))
    Y = fac * ((y[(0, 1)] + D * y[(1, 0)]) - (y[(2, 1)] + D * y[(1, 2)]))
    Z = fac * ((y[(1, 2)] + D * y[(2, 1)]) - (y[(0, 2)] + D * y[(2, 0)]))
    return X, Y, Z


def tdi_aet(f, t_f, hp, hc, lon, lat, psi, tdi=1):
    X, Y, Z = tdi_xyz(f, t_f, hp, hc, lon, lat, psi, tdi)
    A = (Z - X) / np.sqrt(2)
    E = (X - 2 * Y + Z) / np.sqrt(6)
    T = (X + Y + Z) / np.sqrt(3)
    return A, E, T
