"""Shared tools for analysing the lens-model Fisher rows (scripts/fisher_study.py output): covariance with the exact mass-sheet
null direction projected out, parameters held fixed ('known'), derived quantities, 90% CIs, significances, and the plot style.

Conventions
  * 'known' parameters: their rows/columns are removed from F before inversion (conditional Fisher), e.g. known=("kappa",).
  * Free parameters and derived quantities that move by more than their own sigma when sliding along a null direction
    (delta kappa = 0.1) are 'unconstrained' and reported as NaN.
  * 90% CI half-width = 1.645 sigma (Gaussian Fisher).
"""
import glob
import json
import os

import numpy as np

from .fisher_lenses import FisherLens, derived_jacobian, mass_sheet_tangent

Z90 = 1.6448536269514722
LN10 = np.log(10.0)
COND_MS = 1e11
MODEL_COL = dict(unlensed="#7f7f7f", pml="#d62728", nfw="#1f77b4", pmlnfw="#2ca02c")
MODEL_LAB = dict(unlensed="unlensed", pml="PML", nfw="NFW", pmlnfw="PML+NFW")
LABEL = dict(log10_mchirp=r"$\log_{10}\mathcal{M}_z$", q="$q$", spin1z=r"$\chi_1$", spin2z=r"$\chi_2$", log10_dl=r"$\log_{10}D_L$",
             cos_inc=r"$\cos\iota$", coa_phase=r"$\phi_c$", psi=r"$\psi$", ecl_lon=r"$\lambda$", sin_ecl_lat=r"$\sin\beta$", tc=r"$t_c$",
             log10_mlz=r"$\log_{10}M_{Lz}$", y="$y$", kappa=r"$\kappa$", gamma1=r"$\gamma_1$", gamma2=r"$\gamma_2$",
             log10_psi0=r"$\log_{10}\psi_0$", log10_xs=r"$\log_{10}x_s$", log10_m200=r"$\log_{10}M_{200}$", log10_c200=r"$\log_{10}c_{200}$",
             log10_kappa_s=r"$\log_{10}\kappa_s$", f_nfw=r"$f_{\rm NFW}$", log10_mbh_m200=r"$\log_{10}(M_{\rm BH}/M_{200})$",
             gamma_abs=r"$|\gamma|$", y_ms=r"$y/\sqrt{1-\kappa}$", gamma1_ms=r"$\gamma_1/(1-\kappa)$", gamma2_ms=r"$\gamma_2/(1-\kappa)$",
             log10_xs_ms=r"$\log_{10}x_s\sqrt{1-\kappa}$", log10_dl_ms=r"$\log_{10}D_L(1-\kappa)$", tc_ms=r"$t_c-t_uC$")


def style(scale=1.0):
    """Serif (Computer Modern) style, inward ticks on all sides, minor ticks, large fonts."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 18 * scale, "axes.labelsize": 22 * scale,
                         "axes.titlesize": 22 * scale, "xtick.labelsize": 18 * scale, "ytick.labelsize": 18 * scale, "legend.fontsize": 18 * scale,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.size": 7, "ytick.major.size": 7,
                         "xtick.minor.size": 3.5, "ytick.minor.size": 3.5, "axes.linewidth": 1.3, "savefig.dpi": 150})


def load(rows_dir, prefix):
    """[(row dict, file dict)] of every ok row in rows_dir/fisher_study_<prefix>*.json."""
    out = []
    for f in sorted(glob.glob(os.path.join(rows_dir, f"fisher_study_{prefix}*.json"))):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        out += [(r, d) for r in d["rows"] if r.get("ok")]
    return out


def load_unlensed(fisher_dir, source):
    f = os.path.join(fisher_dir, f"fisher_study_unlensed_{source}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); return d["rows"][0], d


def _inv(F):
    """Covariance from a Fisher via the diagonally-normalised pseudo-inverse (physical near-degeneracies -> large, finite sigma)."""
    d = np.sqrt(np.clip(np.diag(F), 1e-300, None))
    return np.linalg.pinv(0.5 * (F + F.T) / np.outer(d, d), rcond=1e-15, hermitian=True) / np.outer(d, d)


EIG_FLOOR = 1e-14     # normalised eigenvalues below this (x largest) are within ~100x of double precision: never resolved
CONV_TOL = 0.1        # an eigenvalue is converged if it changes by < 10% between step h and h/2
SENS_TOL = 0.1        # a quantity is unconstrained if an unresolved direction adds > 10% to its variance at eigenvalue
                      # max(measured upper bound, EIG_FLOOR) (finite-difference noise only adds information; below EIG_FLOOR
                      # nothing can be resolved, so quantities that only feel the direction there are reported)


def eigen_split(F, F_half=None):
    """Eigen-decomposition of the diagonally normalised Fisher N = F/(d d^T), d = sqrt(diag F), with each direction marked
    resolved or not.  Unresolved: lambda < EIG_FLOOR * lambda_max, or (F_half given) the Rayleigh quotient of N_half along the
    eigenvector differs from lambda by > CONV_TOL (the eigenvalue still depends on the finite-difference step).
    Unresolved eigenvalues are replaced by min(lambda_h, lambda_h/2), the tighter of the two upper bounds.
    Returns d, eigenvalues (clipped at 1e-16 lambda_max), eigenvectors (columns), boolean mask of unresolved directions."""
    d = np.sqrt(np.clip(np.diag(F), 1e-300, None)); N = 0.5 * (F + F.T) / np.outer(d, d)
    ev, U = np.linalg.eigh(N); lmax = ev[-1]; evc = np.clip(ev, 1e-16 * lmax, None)
    unres = ev < EIG_FLOOR * lmax
    if F_half is not None:
        Nh = 0.5 * (F_half + F_half.T) / np.outer(d, d); lh = np.clip(np.einsum("ij,ik,kj->j", U, Nh, U), 1e-16 * lmax, None)
        unres |= np.abs(lh / evc - 1) > CONV_TOL
        evc = np.where(unres, np.minimum(evc, lh), evc)
    return d, evc, U, unres


def analyse(r, model, info=None, m_ref=None, known=(), tol=1e-4):
    """{name: (mu, sigma or nan)} of free parameters (minus `known`) and derived quantities of one Fisher row.

    kappa free (kappa in the Fisher and not in `known`): the mass-sheet degeneracy is EXACT, so the variance of any quantity with a
    component along the analytic mass-sheet direction v (fisher_lenses.mass_sheet_tangent) is infinite -> NaN; the exact invariants
    (gradient . v = 0) do not depend on how kappa is fixed, so their covariance is the one with kappa held fixed.
    A quantity counts as moving along v when |g.v| > tol * sum|g_i v_i| (tol absorbs finite-difference round-off).

    Unresolved directions (eigen_split; rows with F_half, i.e. NFW models): their true eigenvalue is below what finite
    differences in double precision can measure; the measured value is only an upper bound.  Any quantity to whose variance such a
    direction contributes > SENS_TOL even at that upper bound is unconstrained -> NaN.  The others keep sigma from the full inverse
    (which includes the bounded unresolved contribution); they would grow only if the true eigenvalue were far below the bound."""
    names0, F0 = r["names"], np.array(r["F"]); vals = r.get("vals", {}); binary = r["binary"]
    kfree = "kappa" in names0 and "kappa" not in known
    drop = set(known) | ({"kappa"} if kfree else set())
    keep = [i for i, n in enumerate(names0) if n not in drop]; names = [names0[i] for i in keep]
    Fh = np.array(r["F_half"])[np.ix_(keep, keep)] if r.get("F_half") is not None else None
    d, evc, U, unres = eigen_split(F0[np.ix_(keep, keep)], Fh)
    C = (U / evc) @ U.T / np.outer(d, d); sig = np.sqrt(np.clip(np.diag(C), 0, None))

    def depends(G):   # rows of G = gradients (w.r.t. the kept parameters) -> True where sigma depends on an unresolved direction
        if not unres.any(): return np.zeros(len(G), bool)
        lref = np.maximum(evc[unres], EIG_FLOOR * evc[-1])   # never below what double precision could ever resolve
        A = (G / d) @ U; vu = (A[:, unres] ** 2 / lref).sum(1); vr = (A[:, ~unres] ** 2 / evc[~unres]).sum(1)
        return vu > SENS_TOL * vr

    dep = depends(np.eye(len(keep)))
    mu = {n: (vals[n] if n in vals else binary[n]) for n in names0}
    v = None; unc = {n for i, n in enumerate(names) if dep[i]}
    if kfree:
        lens = FisherLens(model, m_ref, info["z"])
        v = mass_sheet_tangent(lens, vals, {k: binary[k] for k in binary}, names0)
        unc |= {n for n, vi in zip(names0, v) if abs(vi) > 0}
    out = {n: (mu[n], np.nan if n in unc else sig[i]) for i, n in enumerate(names)}
    if kfree: out["kappa"] = (mu["kappa"], np.nan)
    if model != "unlensed":
        lens = FisherLens(model, m_ref, info["z"])
        dn, dv, J = derived_jacobian(lens, dict(vals), {k: binary[k] for k in ("log10_dl", "tc")}, names0)
        Jk = J[:, keep]; sd = np.sqrt(np.clip(np.diag(Jk @ C @ Jk.T), 0, None)); depd = depends(Jk)
        for q, nm in enumerate(dn):
            bad = depd[q] or (kfree and abs(J[q] @ v) > tol * (np.abs(J[q] * v).sum() + 1e-300))
            out[nm] = (dv[q], np.nan if bad else sd[q])
    out["_C"], out["_names"], out["_n_unresolved"] = C, names, int(unres.sum())
    return out


def hw90(res, name):
    """90% CI half-width of `name` (NaN if unconstrained or absent)."""
    return Z90 * res[name][1] if name in res else np.nan


def significance_log(res, name):
    """Detection significance of a positive amplitude A from sigma(log10 A): A / sigma(A) = 1 / (ln10 sigma_log10)."""
    s = res.get(name, (0, np.nan))[1]
    return 1.0 / (LN10 * s) if np.isfinite(s) and s > 0 else np.nan


def marginal_lens_fisher(r, lens_names):
    """Lens-block Fisher with the binary parameters marginalised (Schur complement)."""
    names, F = r["names"], np.array(r["F"])
    L = [names.index(n) for n in lens_names]; B = [i for i in range(len(names)) if i not in L]
    FLL, FLB, FBB = F[np.ix_(L, L)], F[np.ix_(L, B)], F[np.ix_(B, B)]
    return FLL - FLB @ np.linalg.pinv(FBB, rcond=1e-15) @ FLB.T
