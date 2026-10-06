import numpy as np
import pytest

from lisa_mbhb import demo, inference as I, lensing as L, noise, overlap, response, waveform as W
from lisa_mbhb.constants import C, F_STAR, LISA_ARM, YEAR


@pytest.fixture(scope="module")
def params():
    return demo.default_params()


@pytest.fixture(scope="module")
def injection(params):
    return I.make_injection(params)


# ---------------------------------------------------------------- noise
def test_noise_matches_pycbc_formulas():
    import pycbc.psd.analytical_space as a
    f = np.logspace(-4, -0.5, 40)
    acc, oms = a.lisa_psd_components(f)
    x = a.omega_length(f, LISA_ARM)
    A = 8 * np.sin(x) ** 2 * (4 * (1 + np.cos(x) + np.cos(x) ** 2) * acc + (2 + np.cos(x)) * oms)
    T = 32 * np.sin(x) ** 2 * np.sin(x / 2) ** 2 * (4 * acc * np.sin(x / 2) ** 2 + oms)
    np.testing.assert_allclose(noise.psd_tdi_A(f), A, rtol=1e-10)
    np.testing.assert_allclose(noise.psd_tdi_T(f), T, rtol=1e-10)


def test_sensitivity_reproduces_scird_at_low_frequency():
    f = np.array([1e-3, 3e-3])
    sci = 10 / (3 * LISA_ARM**2) * (noise.p_oms(f) + 2 * (1 + np.cos(f / F_STAR) ** 2)
                                    * noise.p_acc(f) / (2 * np.pi * f) ** 4) * (1 + 0.6 * (f / F_STAR) ** 2)
    np.testing.assert_allclose(noise.sensitivity_curve(f), sci, rtol=0.01)


# ------------------------------------------------------------- response
def test_averaged_response_and_null_channel():
    rng = np.random.default_rng(0)
    f = np.array([1e-3])
    x = 2 * np.pi * f * LISA_ARM / C
    accA = accT = 0.0
    n = 600
    for _ in range(n):
        lon, lat = rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1))
        t = np.array([rng.uniform(0, YEAR)])
        A, E, T = response.tdi_aet(f, t, np.ones(1), np.zeros(1), lon, lat, rng.uniform(0, np.pi))
        accA += abs(A[0]) ** 2 / n
        accT += abs(T[0]) ** 2 / n
    assert accA / (4 * x[0] ** 2 * np.sin(x[0]) ** 2) == pytest.approx(0.9, rel=0.1)
    assert accT < 1e-6 * accA


def test_snr_matches_sky_averaged_estimate():
    rng = np.random.default_rng(3)
    snr2 = []
    for _ in range(50):
        p = demo.default_params(psi=rng.uniform(0, np.pi), ecl_lon=rng.uniform(0, 2 * np.pi),
                                ecl_lat=np.arcsin(rng.uniform(-1, 1)), tc=rng.uniform(0.5, 1) * YEAR)
        inj = I.make_injection(p)
        snr2.append(inj.snr**2)
    m = inj.model
    hp, hc, _ = W.generate(p, m.f, m.cfg)
    avg = np.sqrt(4 * np.sum(m.w * (abs(hp) ** 2 + abs(hc) ** 2) / noise.sensitivity_curve(m.f)))
    assert np.sqrt(np.mean(snr2)) == pytest.approx(avg, rel=0.1)


def test_snr_scales_inverse_with_distance(params, injection):
    far = I.make_injection(dict(params, distance=2 * params["distance"]), injection.model.cfg)
    assert far.snr == pytest.approx(injection.snr / 2, rel=1e-3)


def test_time_frequency_map_monotonic(params, injection):
    _, _, t = W.generate(params, injection.model.f, injection.model.cfg)
    assert np.all(np.diff(t) >= 0) and t[0] < params["tc"] - 0.3 * YEAR and t[-1] == pytest.approx(params["tc"])


# -------------------------------------------------------------- lensing
def test_no_lens_is_identity(params):
    s = L.LensedMBHB(params, None)
    assert s.snr() == pytest.approx(s.snr(lensed=False))
    assert s.mismatch() < 1e-10


def test_amplification_hook_scales_signal(injection, params):
    a = injection.model(params)
    b = injection.model(params, amplification=lambda f: 2 * np.ones_like(f))
    np.testing.assert_allclose(b["A"], 2 * a["A"])


def test_point_lens_geometric_optics_limit():
    from gwlens.lens.point import F_analytic
    y = 0.8
    w = np.linspace(150, 160, 400)
    f2 = np.mean([abs(F_analytic(wi, y)) ** 2 for wi in w])
    assert f2 == pytest.approx((y**2 + 2) / (y * np.sqrt(y**2 + 4)), rel=0.03)
    assert abs(F_analytic(1e-3, y)) == pytest.approx(1.0, abs=0.02)   # wave limit F -> 1


def test_point_lens_interpolation_matches_exact():
    from gwlens.lens.point import F_analytic
    lens = L.PointMassLens(1e7, 0.5)
    f = np.array([2e-3, 7e-3, 1.3e-2])
    F = lens.amplification(f)
    ex = np.array([F_analytic(w, 0.5) for w in lens.w(f)])
    np.testing.assert_allclose(F, ex, atol=2e-3)


def test_lookup_table_lens_matches_point_lens(tmp_path):
    import h5py
    from gwlens.lens.point import F_analytic
    w = np.linspace(0.05, 3.0, 60)
    y = np.linspace(0.4, 0.6, 5)
    res = np.array([[[F_analytic(wi, yi).real, F_analytic(wi, yi).imag] for yi in y] for wi in w])
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as h:
        h["w_values"], h["y_values"], h["results"] = w, y, res
    pm = L.PointMassLens(1e7, 0.5)
    tab = L.LookupTableLens(str(path), pm.w, 0.5)
    f = np.linspace(1e-3, 2.3e-3, 20)   # w = 1.2 .. 2.9, inside the table
    np.testing.assert_allclose(tab.amplification(f), pm.amplification(f), atol=5e-3)


def test_lensing_changes_signal_and_mismatch_decreases_with_y(params):
    mm = []
    for y in (0.5, 2.0):
        s = L.LensedMBHB(params, L.PointMassLens(1e7, y))
        mm.append(s.mismatch(max_dt=400, n_dt=401))
    assert mm[0] > mm[1] > 0


# -------------------------------------------------------------- overlap
def test_mismatch_recovers_pure_time_shift(injection):
    m = injection.model
    shifted = {c: injection.signal[c] * np.exp(-2j * np.pi * m.f * 37.0) for c in m.channels}
    assert overlap.mismatch(m, injection.signal, shifted) > 1e-3
    assert overlap.mismatch(m, injection.signal, shifted, max_dt=60, n_dt=121) < 1e-5


def test_approximants_agree_after_time_shift(params, injection):
    m = injection.model
    hm = I.TDIModel(m.f, W.WaveformConfig(approximant="IMRPhenomHM", tc_ref=params["tc"]))(params)
    assert overlap.mismatch(m, injection.signal, hm, max_dt=300, n_dt=601) < 0.05
