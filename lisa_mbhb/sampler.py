"""Minimal CPU tools for recovery: boxed priors, Fisher matrix, stretch-move ensemble MCMC."""
import numpy as np

from .inference import SAMPLED

# default half-widths of the uniform prior box around the truth
_HALF_WIDTH = dict(log10_mchirp=0.05, q=0.25, spin1z=0.5, spin2z=0.5, log10_dl=0.3, cos_inc=0.6,
                   coa_phase=np.pi, psi=np.pi / 2, ecl_lon=0.5, sin_ecl_lat=0.5, tc=5e4)
_HARD = dict(q=(0.05, 1.0), spin1z=(-0.99, 0.99), spin2z=(-0.99, 0.99), cos_inc=(-1, 1),
             sin_ecl_lat=(-1, 1))


class BoxPrior:
    def __init__(self, truth, widths=None):
        w = dict(_HALF_WIDTH, **(widths or {}))
        lo, hi = [], []
        for name, x in zip(SAMPLED, truth):
            a, b = x - w[name], x + w[name]
            if name in _HARD:
                a, b = max(a, _HARD[name][0]), min(b, _HARD[name][1])
            lo.append(a)
            hi.append(b)
        self.lo, self.hi = np.array(lo), np.array(hi)

    def __call__(self, theta):
        return 0.0 if np.all((theta >= self.lo) & (theta <= self.hi)) else -np.inf

    def sample(self, n, rng):
        return rng.uniform(self.lo, self.hi, size=(n, len(self.lo)))


def fisher_matrix(logl, theta, steps=None):
    """Gaussian-approximation Fisher matrix = -Hessian of ln L, by central differences."""
    theta = np.asarray(theta, float)
    n = len(theta)
    h = np.array([1e-5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1.0]) if steps is None else steps
    f0 = logl(theta)
    H = np.zeros((n, n))
    fp = np.zeros(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = h[i]
        fp[i] = logl(theta + e)
        H[i, i] = (fp[i] - 2 * f0 + logl(theta - e)) / h[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (logl(theta + ei + ej) - logl(theta + ei - ej)
                                 - logl(theta - ei + ej) + logl(theta - ei - ej)) / (4 * h[i] * h[j])
    return -H


def _lnprob(theta, logl, prior):
    lp = prior(theta)
    return -np.inf if not np.isfinite(lp) else lp + logl(theta)


def ensemble_mcmc(logl, prior, p0, nsteps, pool=None, a=2.0, seed=None, progress=True):
    """Goodman & Weare (2010) stretch move.  Returns chain (nsteps, nwalkers, ndim), lnprob."""
    rng = np.random.default_rng(seed)
    nw, nd = p0.shape
    half = nw // 2
    mapf = pool.starmap if pool is not None else (lambda f, it: [f(*x) for x in it])
    x = p0.copy()
    lp = np.array(mapf(_lnprob, [(xi, logl, prior) for xi in x]))
    chain, lnps = np.empty((nsteps, nw, nd)), np.empty((nsteps, nw))
    for s in range(nsteps):
        for grp in (slice(0, half), slice(half, nw)):
            comp = x[slice(half, nw)] if grp.start == 0 else x[slice(0, half)]
            idx = np.arange(*grp.indices(nw))
            z = ((a - 1) * rng.uniform(size=len(idx)) + 1) ** 2 / a
            partner = comp[rng.integers(len(comp), size=len(idx))]
            prop = partner + z[:, None] * (x[idx] - partner)
            lnew = np.array(mapf(_lnprob, [(pi, logl, prior) for pi in prop]))
            acc = np.log(rng.uniform(size=len(idx))) < (nd - 1) * np.log(z) + lnew - lp[idx]
            x[idx[acc]], lp[idx[acc]] = prop[acc], lnew[acc]
        chain[s], lnps[s] = x, lp
        if progress and (s + 1) % max(1, nsteps // 10) == 0:
            print(f"step {s + 1}/{nsteps}  max lnP {lp.max():.1f}", flush=True)
    return chain, lnps
