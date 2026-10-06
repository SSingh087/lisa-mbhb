"""Overlaps and mismatches between TDI signals."""
import numpy as np


def _cross(model, a, b, dts, chunk=64):
    """z(dt) = sum_c 4 int a_c b_c^* e^{+2 pi i f dt} / S_c df  (complex)."""
    z = np.zeros(len(dts), complex)
    for c in model.channels:
        wgt = 4 * model.w * a[c] * np.conj(b[c]) / model.psd[c]
        for i in range(0, len(dts), chunk):
            ph = np.exp(2j * np.pi * np.outer(dts[i:i + chunk], model.f))
            z[i:i + chunk] += ph @ wgt
    return z


def overlap(model, a, b, max_dt=0.0, n_dt=1):
    """Normalised overlap maximised over a global phase and a time shift in [-max_dt, max_dt].

    max_dt = 0 only maximises over phase.  Returns (overlap, best_dt).
    """
    dts = np.array([0.0]) if max_dt == 0 else np.linspace(-max_dt, max_dt, n_dt)
    norm = np.sqrt(model.inner(a, a) * model.inner(b, b))
    z = np.abs(_cross(model, a, b, dts)) / norm
    i = int(np.argmax(z))
    return float(z[i]), float(dts[i])


def mismatch(model, a, b, max_dt=0.0, n_dt=1):
    return 1.0 - overlap(model, a, b, max_dt, n_dt)[0]
