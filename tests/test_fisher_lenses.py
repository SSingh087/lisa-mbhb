"""Checks of lisa_mbhb.fisher_lenses (local, small w).  GWLENS from $GWLENS_PATH."""
import numpy as np
import pytest

from lisa_mbhb import lensing as L        # noqa: F401  (puts GWLENS on sys.path)
from lisa_mbhb.fisher_lenses import FisherLens, derived_jacobian

ZS, MREF = 1.0, 1e6
F = np.geomspace(2e-4, 2e-3, 60)          # w ~ 0.25 - 2.5 for m_ref = 1e6


@pytest.mark.parametrize("model", ["pml", "nfw", "pmlnfw"])
def test_mass_sheet_identity(model):
    """F(w, y; kappa) = (1-kappa)^-1 exp(i w C) F(w, y'; 0), y' = y/sqrt(1-k), x_s' = x_s sqrt(1-k), gamma' = gamma/(1-k).
    This is the exact mass-sheet degeneracy that makes kappa unmeasurable when D_L and t_c are free."""
    Lm = FisherLens(model, MREF, ZS)
    v = Lm.default_values(y=0.7, m200=1e9, kappa=0.2)
    k = v["kappa"]; s = 1 - k
    v0 = dict(v, kappa=0.0, y=v["y"] / np.sqrt(s), gamma1=v["gamma1"] / s, gamma2=v["gamma2"] / s)
    if "log10_xs" in v:
        v0["log10_xs"] = v["log10_xs"] + 0.5 * np.log10(s)
    a = 10 ** v["log10_mlz"] / MREF if "log10_mlz" in v else 0.0
    Cc = -k * v["y"] ** 2 / (2 * s) + 0.5 * a * np.log(s)
    w = Lm.w(F)
    Fk, F0 = Lm.amplification(v, F), Lm.amplification(v0, F)
    pred = F0 * np.exp(1j * w * Cc) / s
    assert np.max(np.abs(Fk - pred) / np.abs(Fk)) < 1e-5


def test_halo_roundtrip_and_normalisation():
    """(M200, c) -> (psi0, x_s) -> (M200, c), and psi0 equals lensing.PMInHaloLens1D (fixed m(c) normalisation) when m_ref = BH mass."""
    Lm = FisherLens("pmlnfw", MREF, ZS)
    for m200 in (1e9, 1e11):
        lp, lx = Lm.halo_params(m200)
        m, c = Lm.halo_physical(lp, lx)
        assert abs(m / m200 - 1) < 1e-8
        cl = L.PMInHaloLens1D(MREF, 1.0, m200, Lm.zl, ZS)
        assert abs(10 ** lp / cl.psi0 - 1) < 1e-10 and abs(10 ** lx / cl.xs - 1) < 1e-10
        assert abs(c / cl.c200 - 1) < 1e-8


def test_tiny_halo_is_pml():
    a = FisherLens("pml", MREF, ZS); b = FisherLens("pmlnfw", MREF, ZS)
    va = a.default_values(y=0.5); vb = b.default_values(y=0.5, m200=1e-3)
    assert np.max(np.abs(a.amplification(va, F) - b.amplification(vb, F))) < 1e-5


def test_derived_jacobian_kappa_direction():
    """Mass-sheet invariants do not change along the exact degeneracy direction (kappa, y, gamma, D_L, t_c moved together)."""
    Lm = FisherLens("pml", MREF, ZS)
    v = Lm.default_values(y=0.7, kappa=0.1); binary = dict(log10_dl=3.8, tc=1e7)
    d0 = Lm.derived(v, binary)
    k2 = 0.15; s1, s2 = 1 - v["kappa"], 1 - k2
    v2 = dict(v, kappa=k2, y=d0["y_ms"] * np.sqrt(s2), gamma1=d0["gamma1_ms"] * s2, gamma2=d0["gamma2_ms"] * s2)
    b2 = dict(log10_dl=d0["log10_dl_ms"] - np.log10(s2), tc=0.0)
    a = 1.0; C2 = -k2 * v2["y"] ** 2 / (2 * s2) + 0.5 * a * np.log(s2); b2["tc"] = d0["tc_ms"] + Lm.t_unit * C2
    d2 = Lm.derived(v2, b2)
    for key in ("y_ms", "gamma1_ms", "gamma2_ms", "log10_dl_ms", "tc_ms"):
        assert abs(d2[key] - d0[key]) < 1e-9 * max(1, abs(d0[key]))
    dn, _, J = derived_jacobian(Lm, v, binary, ["log10_mlz", "y", "kappa", "gamma1", "gamma2", "log10_dl", "tc"])
    assert J.shape == (len(dn), 7)
