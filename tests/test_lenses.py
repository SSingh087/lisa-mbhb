"""Lens registry: lenses by name / config dict, with defaults."""
import json

import numpy as np
import pytest

from lisa_mbhb import lensing as L

F = np.array([2e-3, 5e-3, 1e-2])


def test_available_lenses_lists_names_and_defaults():
    av = L.available_lenses()
    for name in ("pm", "nfw", "halo", "sis", "cis", "lookup"):
        assert name in av
    assert av["pm"] == {"mass_lz": 1e7, "y": 1.0, "dw": 0.05}
    assert av["lookup"]["table_path"] == "<required>"


def test_name_only_uses_defaults():
    lens = L.get_lens("pm")
    assert isinstance(lens, L.PointMassLens)
    assert (lens.mass_lz, lens.y) == (1e7, 1.0)


def test_partial_override_keeps_other_defaults():
    lens = L.get_lens("pm", y=0.3)
    assert (lens.mass_lz, lens.y) == (1e7, 0.3)


def test_config_dict_and_kwargs_override():
    cfg = json.loads('{"name": "nfw", "halo_mass": 1e9, "zs": 2.0}')
    lens = L.get_lens(cfg, y=0.7)
    assert isinstance(lens, L.PMInHaloLens1D)
    assert (lens.halo_mass, lens.zs, lens.zl, lens.y, lens.mass_lz) == (1e9, 2.0, 1.0, 0.7, 1e7)
    assert cfg == {"name": "nfw", "halo_mass": 1e9, "zs": 2.0}      # input not mutated


def test_aliases_and_legacy_spec():
    assert isinstance(L.get_lens("point"), L.PointMassLens)
    a = L.get_lens("halo:1e9", mass_lz=1e7, y=0.5, zs=1.0)
    assert isinstance(a, L.HaloOnlyLens1D) and a.halo_mass == 1e9
    b = L.make_lens("nfw:1e9", 1e7, 0.5, 1.0)
    assert isinstance(b, L.PMInHaloLens1D) and b.zl == 0.5


def test_errors_are_explicit():
    with pytest.raises(ValueError, match="unknown lens"):
        L.get_lens("pointmass")
    with pytest.raises(ValueError, match="no parameter"):
        L.get_lens("pm", masslz=1e7)                 # typo is not silently ignored
    with pytest.raises(ValueError, match="needs"):
        L.get_lens("lookup")                          # required parameter missing
    with pytest.raises(ValueError, match="'name'"):
        L.get_lens({"y": 0.5})
    with pytest.raises(ValueError, match="bad lens spec"):
        L.get_lens("pm:1e9")


def test_lens_object_passes_through():
    lens = L.PointMassLens(1e6, 0.5)
    assert L.get_lens(lens) is lens
    with pytest.raises(ValueError):
        L.get_lens(lens, y=0.3)


def test_registry_lens_matches_direct_class():
    np.testing.assert_allclose(L.get_lens("pm", mass_lz=1e7, y=0.5).amplification(F),
                               L.PointMassLens(1e7, 0.5).amplification(F))


@pytest.mark.parametrize("name", ["nfw", "halo", "sis", "cis"])
def test_solver_lenses_evaluate(name):
    Fv = L.get_lens(name, y=1.0).amplification(F)
    assert Fv.shape == F.shape and np.all(np.isfinite(Fv))


def test_register_custom_lens():
    @L.register_lens("_unit", dict(scale=2.0))
    class Unit:
        def __init__(self, scale):
            self.scale = scale

        def amplification(self, f):
            return self.scale * np.ones_like(f, dtype=complex)

    try:
        assert L.get_lens("_unit").amplification(F)[0] == 2.0
        assert L.get_lens({"name": "_unit", "scale": 3.0}).amplification(F)[0] == 3.0
    finally:
        L.LENSES.pop("_unit")


def test_lensed_mbhb_accepts_name_and_config():
    from lisa_mbhb import demo
    p = demo.default_params()
    s = L.LensedMBHB(p, {"name": "pm", "mass_lz": 1e7, "y": 0.5})
    assert isinstance(s.lens, L.PointMassLens) and s.lens.y == 0.5
    assert isinstance(L.LensedMBHB(p, "pm").lens, L.PointMassLens)


def test_solver_lenses_use_point_mass_time_convention():
    """Solver lenses must use time_origin=None, the convention of the analytic point mass."""
    assert L.solver_kw().get("time_origin", None) is None


def test_nfw_with_vanishing_halo_recovers_point_mass():
    """Full complex F, not just |F|: a vanishing halo must give exactly the point mass."""
    f = np.geomspace(2e-4, 1e-2, 15)
    pm = L.get_lens("pm").amplification(f)
    a = L.get_lens("nfw", halo_mass=1e-6).amplification(f)
    np.testing.assert_allclose(a, pm, rtol=0, atol=1e-6)


def test_point_mass_spline_matches_exact_at_small_w():
    from gwlens.lens.point import F_analytic
    w = np.geomspace(2e-3, 30, 120)
    F = L.PointMassLens(1e7, 1.0).amplification(w / (L.W_CONST * 1e7))
    np.testing.assert_allclose(F, [F_analytic(x, 1.0) for x in w], rtol=0, atol=1e-6)


@pytest.mark.parametrize("halo_mass", [1e9, 1e11])
def test_nfw_psi0_gives_the_halo_mass(halo_mass):
    """Projected mass from the gwlens potential (x dpsi/dx) equals a direct integral of an NFW
    density normalised so that the mass inside r200 is halo_mass (checks the 1/m(c) in psi0)."""
    from scipy.integrate import quad
    from gwlens.solver import solver1d as S1
    from lisa_mbhb.constants import MSUN
    lens = L.get_lens("nfw", halo_mass=halo_mass)
    rs = lens.xs * lens.xi0
    r200 = lens.c200 * rs
    shape = lambda r: 1.0 / ((r / rs) * (1 + r / rs) ** 2)
    rho_s = halo_mass * MSUN / quad(lambda r: 4 * np.pi * r * r * shape(r), 0, r200)[0]
    sigma = lambda R: 2 * quad(lambda z: rho_s * shape(np.hypot(R, z)), 0, 200 * r200, limit=500)[0]
    pot = S1.nfw(k=1 / lens.xs, psi0=lens.psi0)
    m_ref = lens.mass_lz * MSUN / (1 + lens.zl)
    x = lens.xs                                    # at R = r_s
    m2d = quad(lambda R: 2 * np.pi * R * sigma(R), 0, x * lens.xi0, limit=200)[0]
    h = 1e-5 * x
    dpsi = (pot.real(x + h) - pot.real(x - h)) / (2 * h)
    assert m2d / (m_ref * x * dpsi) == pytest.approx(1.0, rel=1e-4)
