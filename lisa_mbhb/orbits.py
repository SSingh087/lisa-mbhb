"""LISA spacecraft orbits in the ecliptic frame (Cutler 1998, O(e) expansion)."""
import numpy as np
from .constants import AU, E_ORBIT, YEAR


def positions(t, kappa=0.0, lam=0.0):
    """Spacecraft positions [m], shape (3, ..., 3): craft index, time, xyz.

    t     : times [s] (scalar or array) since the reference epoch.
    kappa : initial ecliptic longitude of the guiding centre [rad].
    lam   : initial orientation of the constellation [rad].
    """
    t = np.asarray(t, float)
    alpha = 2 * np.pi * t / YEAR + kappa
    sa, ca = np.sin(alpha), np.cos(alpha)
    out = []
    for k in (1, 2, 3):
        beta = 2 * np.pi * (k - 1) / 3 + lam
        sb, cb = np.sin(beta), np.cos(beta)
        x = AU * (ca + E_ORBIT * (sa * ca * sb - (1 + sa**2) * cb))
        y = AU * (sa + E_ORBIT * (sa * ca * cb - (1 + ca**2) * sb))
        z = -np.sqrt(3) * AU * E_ORBIT * np.cos(alpha - beta)
        out.append(np.stack([x, y, z], axis=-1))
    return np.stack(out)
