"""The three lens models of the Fisher study, all with external convergence and shear, in ONE common unit.

Models (free lens parameters; the 11 binary parameters are added by lisa_mbhb.fisher):
    pml    : log10_mlz, y, kappa, gamma1, gamma2
    nfw    : log10_psi0, log10_xs, y, kappa, gamma1, gamma2
    pmlnfw : log10_mlz, log10_psi0, log10_xs, y, kappa, gamma1, gamma2

Common unit (all models): xi0 = Einstein radius of the reference point mass m_ref_z (= the BH mass of the true lens),
    xi0^2 = 4 G d_eff m_ref_z / c^2,  d_eff = D_L D_LS / ((1+z_L) D_S),   w = 8 pi G m_ref_z f / c^3,
so y, x_s and w mean the same physical offset / scale radius / frequency in every model and the corner plots overlay.
Lens potential (GWLENS solver2d, source at (y, 0)):
    psi(x) = a ln|x|  +  psi0 * psi_NFW(|x| / x_s)  +  [kappa |x|^2 + gamma1 (x1^2 - x2^2) + 2 gamma2 x1 x2] / 2,
    a = M_Lz / m_ref_z (PM term, pml and pmlnfw),  psi_NFW = GWLENS nfw(k = 1/x_s, psi0)  (= 4 kappa_s / k^2 normalisation).
For the NFW model m_ref_z is only the unit (the halo has no point mass); it is NOT a parameter.

Derived quantities (lisa_mbhb.fisher_lenses.derived; drawn dashed in the corner plots):
    M200, c200 from (psi0, x_s):  r_s = x_s xi0,  kappa_s = psi0 / (4 x_s^2),  rho_s = kappa_s Sigma_cr / r_s,
        rho_s = (200/3) rho_crit(z_L) c^3 / m(c)  ->  c,   M200 = 4 pi rho_s r_s^3 m(c),   m(c) = ln(1+c) - c/(1+c).
    f_NFW = M200 / (M200 + M_BH)  (pmlnfw).
    Mass-sheet invariants (exact for every model, see test_lens3.py):  F(w, y; kappa) = (1-kappa)^-1 e^{i w C} F(w, y'; 0) with
        y' = y / sqrt(1-kappa),  x_s' = x_s sqrt(1-kappa),  gamma' = gamma / (1-kappa),  psi0 and a unchanged,
        C = -kappa y^2 / (2 (1-kappa)) + (a/2) ln(1-kappa);
    the amplitude is degenerate with D_L and the phase with t_c, so the invariant combinations are
        D_L (1-kappa),  y',  x_s',  gamma',  t_c - t_unit C   (t_unit = 4 G m_ref_z / c^3).
Lens cosmology: flat LCDM H0=70, Om=0.3 (as PMInHaloLens1D).  GWLENS time_origin pinned to None (lensing.solver_kw).
"""
import numpy as np
from scipy.optimize import brentq

from . import lensing as L
from .constants import C, G, MSUN

MODELS = ("pml", "nfw", "pmlnfw")
NAMES = dict(pml=["log10_mlz", "y", "kappa", "gamma1", "gamma2"],
             nfw=["log10_psi0", "log10_xs", "y", "kappa", "gamma1", "gamma2"],
             pmlnfw=["log10_mlz", "log10_psi0", "log10_xs", "y", "kappa", "gamma1", "gamma2"])
BOUNDS = dict(log10_mlz=(-np.inf, np.inf), log10_psi0=(-np.inf, np.inf), log10_xs=(-np.inf, np.inf),
              y=(1e-4, 50.0), kappa=(-0.9, 0.9), gamma1=(-0.9, 0.9), gamma2=(-0.9, 0.9))
STEPS = dict(log10_mlz=3e-4, log10_psi0=1e-3, log10_xs=1e-3, y=3e-4, kappa=1e-3, gamma1=1e-3, gamma2=1e-3)
GAMMA_TRUE = (0.1 * np.cos(2 * np.pi / 8), 0.1 * np.sin(2 * np.pi / 8))   # |gamma| = 0.1 at 22.5 deg (both components non-zero)


def m_of_c(c):
    return np.log(1 + c) - c / (1 + c)


class FisherLens:
    def __init__(self, model, m_ref_z, zs, zl_frac=0.5, solver_opts=None):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        self.model, self.m_ref_z, self.zs, self.zl = model, float(m_ref_z), float(zs), zl_frac * zs
        self.names = list(NAMES[model])
        self.bounds = {n: BOUNDS[n] for n in self.names}
        self.steps = {n: STEPS[n] for n in self.names}
        self.priors = {}                                       # Fisher only: no priors
        self.solver_opts = dict(solver_opts or {})
        CL = L.PMInHaloLens1D
        self.d_eff = CL._d_eff(self.zl, self.zs)                                   # [m], includes 1/(1+z_L)
        self.xi0 = np.sqrt(4 * G * self.d_eff * self.m_ref_z * MSUN) / C          # [m]
        self.m_ref_phys = self.m_ref_z / (1 + self.zl)                             # [Msun]
        self.sigma_cr_xi0sq = self.m_ref_phys * MSUN / np.pi                       # Sigma_cr * xi0^2 [kg]
        hz = CL.H0 * np.sqrt(CL.OM * (1 + self.zl) ** 3 + CL.OL)
        self.rho_crit = 3 * hz ** 2 / (8 * np.pi * G)                              # [kg m^-3]
        self.t_unit = 4 * G * self.m_ref_z * MSUN / C ** 3                         # [s]; w = 2 pi f t_unit

    # ---------- units ----------
    def w(self, f):
        return L.W_CONST * self.m_ref_z * np.asarray(f)

    # ---------- truth from physical halo ----------
    def halo_params(self, m200, c200=None):
        """(log10_psi0, log10_xs) of a halo of physical M200 [Msun] (c200 default: the mean M-c relation of lensing.PMInHaloLens1D)."""
        c = c200 or 8 / (1 + self.zl) * (m200 * 0.7 / 1e14) ** (-0.13)
        r200 = (3 * m200 * MSUN / (4 * np.pi * 200 * self.rho_crit)) ** (1 / 3)
        rs = r200 / c; xs = rs / self.xi0
        psi0 = m200 / (m_of_c(c) * self.m_ref_phys)            # = 4 kappa_s x_s^2
        return float(np.log10(psi0)), float(np.log10(xs))

    def default_values(self, y=1.0, m200=1e11, c200=None, kappa=0.0, gamma=GAMMA_TRUE, **over):
        v = dict(log10_mlz=np.log10(self.m_ref_z), y=y, kappa=kappa, gamma1=gamma[0], gamma2=gamma[1])
        v["log10_psi0"], v["log10_xs"] = self.halo_params(m200, c200)
        v.update({k: float(x) for k, x in over.items() if k in BOUNDS})
        return {n: float(v[n]) for n in self.names}

    # ---------- lens ----------
    def lens2d(self, vals):
        from gwlens.solver import solver2d as S2
        parts = []
        if "log10_mlz" in vals:
            parts.append(S2.point_mass(10 ** vals["log10_mlz"] / self.m_ref_z))
        if "log10_psi0" in vals:
            parts.append(S2.nfw(k=10 ** -vals["log10_xs"], psi0=10 ** vals["log10_psi0"]))
        parts.append(S2.external_shear(gamma1=vals["gamma1"], gamma2=vals["gamma2"], kappa=vals["kappa"]))
        lens = parts[0]
        for p in parts[1:]:
            lens = lens + p
        return lens

    def amplification(self, vals, f):
        from gwlens.solver import solver2d as S2
        w = self.w(f)
        opts = dict(self.solver_opts); acc = opts.pop("accuracy", "pe")
        if acc == "full":       # GWLENS default (validated) settings, ~1e-12 in F; interpolation checked to 1e-9
            kw = L.solver_kw(S2, y_radius=0.0, map_ext=1.0, interp_tol=1e-9, interpolate=True, n_workers=1)
            for k in getattr(S2, "PE_SETTINGS", {}):
                kw.pop(k, None)
        else:                   # PE_SETTINGS, ~1e-7 in F
            kw = L.solver_kw(S2, y_radius=0.0, map_ext=1.0, interp_tol=1e-6, interpolate=True, n_workers=1)
        kw.update(opts)
        with S2.Solver2D(self.lens2d(vals), y=(vals["y"], 0.0), wmin=w.min() * 0.99, wmax=w.max() * 1.01, **kw) as s:
            return np.asarray(s.spectrum(w), dtype=complex)

    # ---------- derived quantities ----------
    def halo_physical(self, log10_psi0, log10_xs):
        """(M200 [Msun], c200) from (psi0, x_s) in this unit."""
        psi0, xs = 10 ** log10_psi0, 10 ** log10_xs
        rs = xs * self.xi0
        rho_s = (psi0 / (4 * xs ** 2)) * (self.sigma_cr_xi0sq / self.xi0 ** 2) / rs
        g = lambda lc: np.log((200 / 3) * self.rho_crit * np.exp(3 * lc) / m_of_c(np.exp(lc))) - np.log(rho_s)
        c = float(np.exp(brentq(g, np.log(1e-3), np.log(1e5))))
        return float(4 * np.pi * rho_s * rs ** 3 * m_of_c(c) / MSUN), c

    def derived(self, vals, binary=None):
        """{name: value} of derived quantities at (lens vals [+ binary theta dict for the mass-sheet invariants])."""
        d = {}
        if "log10_psi0" in vals:
            m200, c = self.halo_physical(vals["log10_psi0"], vals["log10_xs"])
            d["log10_m200"], d["log10_c200"] = np.log10(m200), np.log10(c)
            d["log10_kappa_s"] = vals["log10_psi0"] - 2 * vals["log10_xs"] - np.log10(4.0)   # inner convergence scale psi0/(4 x_s^2)
            if "log10_mlz" in vals:
                mbh = 10 ** vals["log10_mlz"] / (1 + self.zl)
                d["f_nfw"] = m200 / (m200 + mbh)
                d["log10_mbh_m200"] = np.log10(mbh) - np.log10(m200)        # relative mass BH / halo
        k = vals["kappa"]; s = 1 - k
        a = 10 ** vals["log10_mlz"] / self.m_ref_z if "log10_mlz" in vals else 0.0
        d["y_ms"] = vals["y"] / np.sqrt(s)
        d["gamma1_ms"], d["gamma2_ms"] = vals["gamma1"] / s, vals["gamma2"] / s
        d["gamma_abs"] = float(np.hypot(vals["gamma1"], vals["gamma2"]))      # shear strength
        if "log10_xs" in vals:
            d["log10_xs_ms"] = vals["log10_xs"] + 0.5 * np.log10(s)
        if binary is not None:
            d["log10_dl_ms"] = binary["log10_dl"] + np.log10(s)
            Cc = -k * vals["y"] ** 2 / (2 * s) + 0.5 * a * np.log(s)
            d["tc_ms"] = binary["tc"] - self.t_unit * Cc
        return d


def derived_jacobian(lens, vals, binary, names, h=1e-6):
    """(derived names, J[d, p]) by central differences of lens.derived w.r.t. the Fisher parameter list `names`."""
    base = lens.derived(vals, binary); dn = list(base)
    J = np.zeros((len(dn), len(names)))
    for j, n in enumerate(names):
        vp, vm, bp, bm = dict(vals), dict(vals), dict(binary), dict(binary)
        src_p, src_m = (vp, vm) if n in vals else (bp, bm)
        if n not in src_p:
            continue
        hh = h * max(1.0, abs(src_p[n])); src_p[n] += hh; src_m[n] -= hh
        dp, dm = lens.derived(vp, bp), lens.derived(vm, bm)
        J[:, j] = [(dp[k] - dm[k]) / (2 * hh) for k in dn]
    return dn, np.array([base[k] for k in dn]), J


def mass_sheet_tangent(lens, vals, binary, names, h=1e-6):
    """d(parameter)/d(kappa) along the EXACT mass-sheet transformation through (vals, binary), for the Fisher parameter list `names`
    (zero for parameters the transformation leaves unchanged).  The invariants y/sqrt(1-k), gamma/(1-k), x_s sqrt(1-k), D_L (1-k),
    t_c - t_unit C are held fixed while kappa moves."""
    inv = lens.derived(vals, binary); a_ = 10 ** vals["log10_mlz"] / lens.m_ref_z if "log10_mlz" in vals else 0.0

    def point(k):
        s = 1 - k; v = dict(vals); b = dict(binary); v["kappa"] = k
        v["y"] = inv["y_ms"] * np.sqrt(s); v["gamma1"], v["gamma2"] = inv["gamma1_ms"] * s, inv["gamma2_ms"] * s
        if "log10_xs" in v: v["log10_xs"] = inv["log10_xs_ms"] - 0.5 * np.log10(s)
        b["log10_dl"] = inv["log10_dl_ms"] - np.log10(s)
        b["tc"] = inv["tc_ms"] + lens.t_unit * (-k * v["y"] ** 2 / (2 * s) + 0.5 * a_ * np.log(s))
        return np.array([v[n] if n in v else b.get(n, binary.get(n, 0.0)) for n in names])
    k0 = vals["kappa"]
    return (point(k0 + h) - point(k0 - h)) / (2 * h)
