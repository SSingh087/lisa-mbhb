"""Lenses: GWLENS solvers only; named lenses / config dicts with defaults; physical units."""
import json

import numpy as np
import pytest

from lisa_mbhb import lensing as L

F = np.array([2e-3, 5e-3, 1e-2])


def S1():
    from gwlens.solver import solver1d as S
    return S


def test_available_lenses_lists_names_and_defaults():
    av = L.available_lenses()
    for name in ("pm", "nfw", "halo", "sis", "cis"):
        assert name in av
    assert av["pm"] == {"mass_lz": 1e7, "y": 1.0}
    assert av["nfw"]["halo_mass"] == 1e11


def test_every_named_lens_is_a_gwlens_solver_lens():
    from gwlens.solver import solver1d as S
    for name in L.LENSES:
        lens = L.get_lens(name)
        assert isinstance(lens, L.Lens) and isinstance(lens.potential, S.Potential), name


def test_name_only_uses_defaults():
    lens = L.get_lens("pm")
    assert (lens.mass_lz, lens.y) == (1e7, 1.0)


def test_partial_override_keeps_other_defaults():
    lens = L.get_lens("pm", y=0.3)
    assert (lens.mass_lz, lens.y) == (1e7, 0.3)


def test_config_dict_and_kwargs_override():
    cfg = json.loads('{"name": "nfw", "halo_mass": 1e9, "zs": 2.0}')
    lens = L.get_lens(cfg, y=0.7)
    assert (lens.info["halo_mass"], lens.info["zs"], lens.info["zl"], lens.y, lens.mass_lz) == (1e9, 2.0, 1.0, 0.7, 1e7)
    assert cfg == {"name": "nfw", "halo_mass": 1e9, "zs": 2.0}      # input not mutated


def test_aliases_and_kind_spec():
    assert L.get_lens("point").potential is not None
    a = L.get_lens("halo:1e9", mass_lz=1e7, y=0.5, zs=1.0)
    assert a.info["halo_mass"] == 1e9 and a.info["zl"] == 0.5


def test_errors_are_explicit():
    with pytest.raises(ValueError, match="unknown lens"):
        L.get_lens("pointmass")
    with pytest.raises(ValueError, match="no parameter"):
        L.get_lens("pm", masslz=1e7)                 # typo is not silently ignored
    with pytest.raises(ValueError, match="'name'"):
        L.get_lens({"y": 0.5})
    with pytest.raises(ValueError, match="bad lens spec"):
        L.get_lens("pm:1e9")
    with pytest.raises(ValueError, match="only y and mass_lz"):
        L.get_lens(S1().sis(1.0), halo_mass=1e9)


def test_lens_object_passes_through():
    lens = L.get_lens("pm", mass_lz=1e6, y=0.5)
    assert L.get_lens(lens) is lens
    with pytest.raises(ValueError):
        L.get_lens(lens, y=0.3)


def test_any_gwlens_potential_can_be_used_directly():
    S = S1()
    a = L.get_lens(S.point_mass(1.0), y=0.5, mass_lz=1e7).amplification(F)
    np.testing.assert_allclose(a, L.get_lens("pm", y=0.5).amplification(F))
    b = L.get_lens(S.point_mass(1.0) + S.sis(0.3), y=0.5).amplification(F)
    assert np.all(np.isfinite(b)) and np.max(np.abs(b - a)) > 1e-3


def test_two_d_potential_with_shear_uses_solver2d():
    from gwlens.solver import solver2d as S2
    lens = L.Lens(S2.point_mass(1.0) + S2.external_shear(gamma1=0.0, gamma2=0.0, kappa=0.0), y=0.5, mass_lz=1e7)
    assert lens.two_d
    np.testing.assert_allclose(lens.amplification(F), L.get_lens("pm", y=0.5).amplification(F), rtol=0, atol=1e-5)


@pytest.mark.parametrize("name", ["nfw", "halo", "sis", "cis"])
def test_solver_lenses_evaluate(name):
    Fv = L.get_lens(name, y=1.0).amplification(F)
    assert Fv.shape == F.shape and np.all(np.isfinite(Fv))


def test_register_custom_lens():
    @L.register_lens("_sis_scaled", dict(mass_lz=1e7, y=1.0, psi0=0.5))
    def build(mass_lz, y, psi0):
        return L.Lens(S1().sis(psi0), y, mass_lz)

    try:
        a = L.get_lens("_sis_scaled").amplification(F)
        b = L.get_lens({"name": "_sis_scaled", "psi0": 1.0}).amplification(F)
        np.testing.assert_allclose(b, L.get_lens("sis").amplification(F))
        assert np.max(np.abs(a - b)) > 1e-3
    finally:
        L.LENSES.pop("_sis_scaled")


def test_lensed_mbhb_accepts_name_and_config():
    from lisa_mbhb import demo
    p = demo.default_params()
    s = L.LensedMBHB(p, {"name": "pm", "mass_lz": 1e7, "y": 0.5})
    assert isinstance(s.lens, L.Lens) and s.lens.y == 0.5
    assert isinstance(L.LensedMBHB(p, "pm").lens, L.Lens)


def test_solver_lenses_use_point_mass_time_convention():
    assert L.solver_kw().get("time_origin", None) is None


def test_nfw_with_vanishing_halo_recovers_point_mass():
    """Full complex F: a vanishing halo must give exactly the point mass."""
    f = np.geomspace(2e-4, 1e-2, 15)
    np.testing.assert_allclose(L.get_lens("nfw", halo_mass=1e-6).amplification(f),
                               L.get_lens("pm").amplification(f), rtol=0, atol=1e-6)


def test_point_mass_matches_exact_over_wide_w_range():
    from gwlens.lens.point import F_analytic
    w = np.geomspace(2e-3, 30, 120)
    F_ = L.get_lens("pm", mass_lz=1e7, y=1.0).amplification(w / (L.W_CONST * 1e7))
    np.testing.assert_allclose(F_, [F_analytic(x, 1.0) for x in w], rtol=0, atol=1e-7)


@pytest.mark.parametrize("halo_mass", [1e9, 1e11])
def test_nfw_psi0_gives_the_halo_mass(halo_mass):
    """Projected mass from the GWLENS potential (x dpsi/dx) equals a direct integral of an NFW
    density normalised so that the mass inside r200 is halo_mass (checks the 1/m(c) in psi0)."""
    from scipy.integrate import quad
    from lisa_mbhb.constants import MSUN
    lens = L.get_lens("halo", halo_mass=halo_mass)
    i = lens.info
    x_unit = L.xi0(lens.mass_lz, i["zl"], i["zs"])
    rs = i["xs"] * x_unit
    r200 = i["c200"] * rs
    shape = lambda r: 1.0 / ((r / rs) * (1 + r / rs) ** 2)
    rho_s = halo_mass * MSUN / quad(lambda r: 4 * np.pi * r * r * shape(r), 0, r200)[0]
    sigma = lambda R: 2 * quad(lambda z: rho_s * shape(np.hypot(R, z)), 0, 200 * r200, limit=500)[0]
    m_ref = lens.mass_lz * MSUN / (1 + i["zl"])
    x = i["xs"]                                     # at R = r_s
    m2d = quad(lambda R: 2 * np.pi * R * sigma(R), 0, x * x_unit, limit=200)[0]
    h = 1e-5 * x
    dpsi = (lens.potential.real(x + h) - lens.potential.real(x - h)) / (2 * h)
    assert m2d / (m_ref * x * dpsi) == pytest.approx(1.0, rel=1e-4)
