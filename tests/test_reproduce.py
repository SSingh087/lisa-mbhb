"""Regression tests: reproduce exact reference values (tests/data/reference.json, written by
scripts/make_reference.py) and the published results in figures/*.json.

If a dependency upgrade changes these numbers, the failure message names the versions the
reference was made with, so the change can be checked before the reference is regenerated.
"""
import json
import os
from importlib.metadata import version

import numpy as np
import pytest

from lisa_mbhb import demo, lensing as L, waveform as W, workflow

HERE = os.path.dirname(os.path.abspath(__file__))
REF = json.load(open(os.path.join(HERE, "data", "reference.json")))
FIG = os.path.join(HERE, "..", "figures")
RTOL = 1e-9


def _versions_note():
    now = {k: version(k) for k in REF["versions"]}
    return f"reference made with {REF['versions']}; installed {now}"


def _c(pairs):
    return np.array([complex(a, b) for a, b in pairs])


@pytest.fixture(scope="module")
def params():
    p = demo.default_params()
    assert p == pytest.approx(REF["params"], rel=1e-15)
    return p


@pytest.mark.parametrize("ap", list(REF["waveform"]))
def test_waveform_generation_reproduces_reference(params, ap):
    r = REF["waveform"][ap]
    cfg = W.WaveformConfig(approximant=ap, tc_ref=params["tc"])
    f = W.design_grid(params, cfg)
    assert len(f) == r["n"]
    assert f[0] == pytest.approx(r["f_first"], rel=1e-13) and f[-1] == pytest.approx(r["f_last"], rel=1e-13)
    hp, hc, _ = W.generate(params, f, cfg)
    np.testing.assert_allclose(hp[r["idx"]], _c(r["hp"]), rtol=RTOL, err_msg=_versions_note())
    np.testing.assert_allclose(hc[r["idx"]], _c(r["hc"]), rtol=RTOL, err_msg=_versions_note())
    assert np.sum(np.abs(hp) ** 2) == pytest.approx(r["norm2_hp"], rel=RTOL), _versions_note()
    assert np.sum(np.abs(hc) ** 2) == pytest.approx(r["norm2_hc"], rel=RTOL), _versions_note()


@pytest.fixture(scope="module")
def approximant_results(params):
    return workflow.compare_approximants(params, list(REF["snr"]))


def test_snr_and_mismatch_reproduce_reference(approximant_results):
    for ap, r in REF["snr"].items():
        got = approximant_results[ap]
        assert got["snr"] == pytest.approx(r["snr"], rel=RTOL), (ap, _versions_note())
        for c, v in r["channels"].items():
            assert got["snr_channels"][c] == pytest.approx(v, rel=RTOL), (ap, c, _versions_note())
        if r["mismatch_vs_IMRPhenomD"] is not None:
            assert got["mismatch_vs_ref"] == pytest.approx(r["mismatch_vs_IMRPhenomD"], rel=1e-7), ap


def test_published_step01_results(approximant_results):
    """figures/lisa_mbhb_01_results.json (README numbers)."""
    pub = json.load(open(os.path.join(FIG, "lisa_mbhb_01_results.json")))
    for ap, r in pub.items():
        assert approximant_results[ap]["snr"] == pytest.approx(r["snr"], rel=1e-8), ap
        assert len(approximant_results[ap]["inj"].model.f) == r["n_freq"]


def test_lensed_snr_reproduces_reference_and_published(params):
    r = REF["lensed_snr"]
    pub = json.load(open(os.path.join(FIG, "lisa_mbhb_02_results.json")))
    ys = [float(y) for y in r["lensed"]]
    scan = workflow.lensed_snr_scan(params, r["mass_lz"], ys)
    for y in ys:
        assert scan[y]["snr_unlensed"] == pytest.approx(r["unlensed"], rel=RTOL)
        assert scan[y]["snr"] == pytest.approx(r["lensed"][str(y)], rel=RTOL), y
        assert scan[y]["snr"] == pytest.approx(pub["lensed"][str(y)], rel=1e-8), y   # README numbers


@pytest.mark.parametrize("key", [k for k in REF["lens_F"] if k != "f"])
def test_lens_amplification_reproduces_reference(key):
    case = REF["lens_F"][key]
    F = L.get_lens(case["config"]).amplification(np.array(REF["lens_F"]["f"]))
    np.testing.assert_allclose(F, _c(case["F"]), rtol=1e-7, atol=1e-9, err_msg=_versions_note())
