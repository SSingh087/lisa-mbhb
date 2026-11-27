"""Write tests/data/reference.json: exact reference values for the regression tests.

    python scripts/make_reference.py

Records waveforms (h+, hx samples and norms) for each approximant, LISA TDI SNRs, approximant
mismatches, point-mass lensed SNRs, and F(f) of every registered lens, together with the
versions of the libraries that produced them.  Regenerate only on purpose (a physics change or
a dependency upgrade you have checked), and say why in the commit message.
"""
import json, os, sys
from importlib.metadata import version

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lisa_mbhb import demo, lensing as L, waveform as W, workflow   # noqa: E402

APPROXIMANTS = ["IMRPhenomD", "IMRPhenomXHM", "IMRPhenomHM"]
LENSED_Y = [0.3, 0.5, 1.0, 2.0]
LENS_F = [2e-4, 1e-3, 3e-3, 1e-2]
LENS_CASES = {
    "pm": {"name": "pm"},
    "pm_y0.3": {"name": "pm", "y": 0.3},
    "nfw_1e9": {"name": "nfw", "halo_mass": 1e9},
    "nfw_1e11": {"name": "nfw", "halo_mass": 1e11},
    "halo_1e9": {"name": "halo", "halo_mass": 1e9},
    "halo_1e11": {"name": "halo", "halo_mass": 1e11},
    "sis": {"name": "sis"},
    "cis": {"name": "cis"},
}


def cplx(a):
    a = np.asarray(a, dtype=complex)
    return [[float(x.real), float(x.imag)] for x in np.atleast_1d(a)]


def build():
    p = demo.default_params()
    ref = {"versions": {k: version(k) for k in ("gwlens", "pycbc", "lalsuite", "numpy", "scipy")},
           "params": p, "waveform": {}, "snr": {}, "lensed_snr": {}, "lens_F": {"f": LENS_F}}
    for ap in APPROXIMANTS:
        cfg = W.WaveformConfig(approximant=ap, tc_ref=p["tc"])
        f = W.design_grid(p, cfg)
        hp, hc, _ = W.generate(p, f, cfg)
        idx = [0, len(f) // 4, len(f) // 2, 3 * len(f) // 4, len(f) - 1]
        ref["waveform"][ap] = dict(n=len(f), f_first=float(f[0]), f_last=float(f[-1]), idx=idx,
                                   hp=cplx(hp[idx]), hc=cplx(hc[idx]),
                                   norm2_hp=float(np.sum(np.abs(hp) ** 2)), norm2_hc=float(np.sum(np.abs(hc) ** 2)))
    for ap, r in workflow.compare_approximants(p, APPROXIMANTS).items():
        ref["snr"][ap] = dict(snr=r["snr"], channels=r["snr_channels"], mismatch_vs_IMRPhenomD=r["mismatch_vs_ref"])
    scan = workflow.lensed_snr_scan(p, 1e7, LENSED_Y)
    ref["lensed_snr"] = {"mass_lz": 1e7, "unlensed": scan[LENSED_Y[0]]["snr_unlensed"],
                         "lensed": {str(y): r["snr"] for y, r in scan.items()}}
    f = np.array(LENS_F)
    for key, cfg in LENS_CASES.items():
        ref["lens_F"][key] = dict(config=cfg, F=cplx(L.get_lens(cfg).amplification(f)))
    return ref


if __name__ == "__main__":
    out = os.path.join(ROOT, "tests", "data", "reference.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(build(), fh, indent=1)
    print("wrote", out)
