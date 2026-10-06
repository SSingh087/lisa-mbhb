"""Reusable steps shared by scripts/ and the notebook (compute only, no plotting)."""
import numpy as np

from . import demo, inference as I, lensing as L, overlap, waveform as W


def compare_approximants(params, approximants, tdi=1, max_dt=300.0, n_dt=601):
    """Steps 1-3: LISA TDI signal and SNR for each approximant, plus mismatch to the first.

    Returns {name: dict(inj=Injection, snr, snr_channels, mismatch_vs_ref)}.
    Approximants use different time origins, so the mismatch is maximised over a shift.
    """
    out, ref = {}, None
    for name in approximants:
        cfg = W.WaveformConfig(approximant=name, tc_ref=params["tc"])
        inj = I.make_injection(params, cfg, tdi=tdi)
        out[name] = dict(inj=inj, snr=inj.snr, snr_channels=inj.snr_channels, mismatch_vs_ref=None)
        if ref is None:
            ref = name
        else:
            m = out[ref]["inj"].model
            other = I.TDIModel(m.f, cfg, tdi=tdi)(params)
            out[name]["mismatch_vs_ref"] = overlap.mismatch(m, out[ref]["inj"].signal, other, max_dt, n_dt)
    return out


def lensed_snr_scan(params, mass_lz, ys, approximant="IMRPhenomD"):
    """Steps 4-5: unlensed SNR and lensed SNR / F(f) / h+ for each impact parameter y."""
    cfg = W.WaveformConfig(approximant=approximant, tc_ref=params["tc"])
    res = {}
    for y in ys:
        s = L.LensedMBHB(params, L.PointMassLens(mass_lz, y), cfg)
        hp, _, F = s.polarizations()
        res[y] = dict(sig=s, snr=s.snr(), snr_unlensed=s.snr(lensed=False), F=F, hp=hp)
    return res


def mismatch_grid(params, mass_lzs, ys, approximant="IMRPhenomD", cap=2501, verbose=True):
    """Step 6: mismatch (max over phase and time shift) vs y for each lens mass."""
    cfg = W.WaveformConfig(approximant=approximant, tc_ref=params["tc"])
    out = {}
    for M in mass_lzs:
        max_dt, n_dt = demo.shift_range(M, cap=cap)
        vals = []
        for y in ys:
            s = L.LensedMBHB(params, L.PointMassLens(M, y), cfg)
            vals.append(s.mismatch(max_dt, n_dt))
            if verbose:
                print(f"M_Lz={M:.0e} y={y}: mismatch={vals[-1]:.3e}", flush=True)
        out[M] = vals
    return out
