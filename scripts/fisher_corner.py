"""True corner plots from the Fisher matrices: the full multivariate Gaussian N(theta_true, F^-1) over ALL parameters
(11 MBH + lens), sampled, with the 1D marginals and 90% 2D regions in ABSOLUTE parameter values and the injected
(true) values marked.  Samples outside the physical bounds (q<=1, |spins|<0.99, |cos i|<=1, |sin lat|<=1, y>0) are rejected.

    python scripts/fisher_corner.py --shards DIR --out DIR [--n 3] [--nsamp 400000]
Writes corner_full_<tag>.png (all parameters) and corner_lens_<tag>.png (lens + chirp mass, q, distance, inclination).
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lisa_mbhb import demo, fisher as FI, inference as I  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=3); ap.add_argument("--nsamp", type=int, default=400000)
ap.add_argument("--cond-max", type=float, default=1e8)
ap.add_argument("--all", action="store_true", help="one corner pair for EVERY Fisher point, into <out>/<lens>/ (use --shard/--nshards to parallelise)")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--only", nargs="+", default=["full", "lens"], choices=["full", "lens"])
ap.add_argument("--match", nargs="+", default=None, help="with --all: only points whose tag contains one of these strings (exact tags like pm_z1_M3e06_q0.3_L1e08_y0.3)")
ap.add_argument("--compact", nargs="+", default=["log10_mlz", "log10_mhalo", "y", "log10_mchirp", "q", "log10_dl", "cos_inc"])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

LABEL = dict(log10_mchirp=r"$\log_{10}\mathcal{M}_z$", q="$q$", spin1z=r"$\chi_1$", spin2z=r"$\chi_2$", log10_dl=r"$\log_{10}D_L$",
             cos_inc=r"$\cos\iota$", coa_phase=r"$\phi_c$", psi=r"$\psi$", ecl_lon=r"$\lambda$", sin_ecl_lat=r"$\sin\beta$",
             tc="$t_c$ [s]", log10_mlz=r"$\log_{10}M_{Lz}$", y="$y$", log10_mhalo=r"$\log_{10}M_{halo}$")
LENSN = ("log10_mlz", "y", "log10_mhalo")
UP = dict(q=1.0, spin1z=0.99, spin2z=0.99, cos_inc=1.0, sin_ecl_lat=1.0)
LO = dict(q=0.02, spin1z=-0.99, spin2z=-0.99, cos_inc=-1.0, sin_ecl_lat=-1.0, y=0.0)

pts = []
for f in sorted(glob.glob(os.path.join(a.shards, "fisher*.json"))):
    d = json.load(open(f))
    for r, m in zip(d["rows"], d["fisher"]):
        if r.get("ok") and m: pts.append((r, m["names"], np.array(m["F"])))
if not pts:
    fig = plt.figure(figsize=(5, 2)); fig.text(.5, .5, "no Fisher points", ha="center"); fig.savefig(os.path.join(a.out, "corner_none.png")); sys.exit("no points")


def tag(r): return f"{r['lens']}_z{r['z']:g}_M{r['m_tot']:.0e}_q{r['q']:g}_L{r['mass_lz']:.0e}_y{r['y']:g}".replace("+", "")


def truth(r, names):
    m1 = r["m_tot"] / (1 + r["q"]); p = demo.default_params(m1=m1, m2=r["q"] * m1, z=r["z"], inclination=r["inc"])
    t = dict(zip(I.SAMPLED, I.params_to_theta(p))); t["log10_mlz"] = np.log10(r["mass_lz"]); t["y"] = r["y"]
    if r["lens"].startswith(("nfw", "halo")): t["log10_mhalo"] = np.log10(float(r["lens"].split(":")[1]))
    return np.array([t[n] for n in names])


def ticks(t, s):
    """3 ticks (truth, +/-2 sigma) with just enough digits to resolve sigma."""
    dig = int(max(0, 2 - np.floor(np.log10(s)))) if s > 0 else 3
    xs = np.array([t - 2 * s, t, t + 2 * s]); return xs, [f"{x:.{dig}f}" for x in xs]


Z90 = 1.6448536269514722          # 90% central interval of a Gaussian = +/- Z90 sigma
def med90(col):
    """median and the 5%/95% percentiles of a sample column -> (median, +upper error, -lower error)"""
    lo, med, hi = np.percentile(col, [5, 50, 95]); return med, hi - med, med - lo


def fmt90(med, up, lo, tex=True):
    d = int(max(0, 2 - np.floor(np.log10(max(min(up, lo), 1e-300)))))
    return (f"{med:.{d}f}$^{{+{up:.2g}}}_{{-{lo:.2g}}}$" if tex else f"{med:.{d}f} +{up:.2g} -{lo:.2g}")


BLUE, ORANGE, TEAL = "#2f6db5", "#e8710a", "#2a9d8f"


def hpd_level(H, frac):
    """Density level enclosing `frac` of the probability (highest-posterior-density contour)."""
    h = np.sort(H.ravel())[::-1]; c = np.cumsum(h); return h[min(np.searchsorted(c, frac * c[-1]), len(h) - 1)]


def plot(r, names, F, use, fname, rng):
    from scipy.ndimage import gaussian_filter
    C, cond, _ = FI.covariance(F); mu = truth(r, names)
    ix = [names.index(p) for p in use]; mu_u, C_u = mu[ix], C[np.ix_(ix, ix)]
    # sample the FULL Gaussian (so marginalisation over the dropped parameters is exact), keep the requested columns
    x = rng.multivariate_normal(mu, C, size=a.nsamp, method="eigh")
    ok = np.ones(len(x), bool)
    for k, nm in enumerate(names):
        if nm in UP: ok &= x[:, k] <= UP[nm]
        if nm in LO: ok &= x[:, k] >= LO[nm]
    x = x[ok][:, ix]; sig = np.sqrt(np.diag(C_u)); n = len(use)
    rngs = [(m - 4 * s_, m + 4 * s_) for m, s_ in zip(mu_u, sig)]
    m90 = [med90(x[:, k]) for k in range(n)]
    lens = [k for k, p_ in enumerate(use) if p_ in LENSN]
    plt.rcParams.update({"font.size": 13, "axes.linewidth": 0.8, "mathtext.fontset": "cm"})
    fig = plt.figure(figsize=(2.3 * n + 1.6, 2.3 * n + 1.6))
    gs = fig.add_gridspec(n, n, left=0.09, right=0.985, bottom=0.10, top=0.93, wspace=0.07, hspace=0.07)
    ax = np.empty((n, n), object)
    for i in range(n):
        for j in range(i + 1):
            ax[i, j] = fig.add_subplot(gs[i, j])
            li, lj = i in lens, j in lens
            col = ORANGE if (li and lj) else (TEAL if (li or lj) else BLUE)
            a_ = ax[i, j]
            if i == j:
                hh, ee = np.histogram(x[:, j], bins=70, range=rngs[j], density=True); c_ = 0.5 * (ee[1:] + ee[:-1])
                hh = gaussian_filter(hh, 1.0)
                a_.fill_between(c_, hh, color=col, alpha=0.30, lw=0); a_.plot(c_, hh, color=col, lw=1.3)
                q5, q95 = np.percentile(x[:, j], [5, 95])
                for qq in (q5, q95): a_.axvline(qq, color=col, ls="--", lw=0.9)
                a_.axvline(mu_u[j], color="crimson", lw=1.1)
                rel = f", {100 * sig[j] / abs(mu_u[j]):.2g}%" if abs(mu_u[j]) > 0 else ""
                a_.set_title(fmt90(*m90[j]), fontsize=11, color=col if col != BLUE else "0.2", pad=4)
                a_.set_yticks([]); a_.set_ylim(0, hh.max() * 1.12)
            else:
                H, xe, ye = np.histogram2d(x[:, j], x[:, i], bins=80, range=[rngs[j], rngs[i]])
                H = gaussian_filter(H.T, 1.4); xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                l90, l68, l39 = hpd_level(H, 0.90), hpd_level(H, 0.68), hpd_level(H, 0.39)
                a_.set_facecolor("#fff4e6" if (li or lj) else "white")
                a_.imshow(np.ma.masked_less(H, l90 * 0.05), origin="lower", extent=[*rngs[j], *rngs[i]], aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("c", ["white", col]), alpha=0.85, interpolation="bilinear")
                a_.contourf(xc, yc, H, levels=[l90, H.max() * 1.01], colors=[col], alpha=0.12)
                a_.contour(xc, yc, H, levels=[l90], colors=[col], linewidths=1.6)
                a_.contour(xc, yc, H, levels=[l68], colors=[col], linewidths=0.9, linestyles="--")
                a_.axvline(mu_u[j], color="crimson", lw=0.8, alpha=.8); a_.axhline(mu_u[i], color="crimson", lw=0.8, alpha=.8)
                a_.plot(mu_u[j], mu_u[i], "o", ms=4, mfc="crimson", mec="white", mew=0.7, zorder=5)
            a_.set_xlim(rngs[j]);
            if i != j: a_.set_ylim(rngs[i])
            if li and lj:
                for sp in a_.spines.values(): sp.set_edgecolor(ORANGE); sp.set_linewidth(2.0)
            xt, xl = ticks(mu_u[j], sig[j]); a_.set_xticks(xt)
            if i == n - 1: a_.set_xticklabels(xl, rotation=40, ha="right", fontsize=10); a_.set_xlabel(LABEL.get(use[j], use[j]), fontsize=15, color=ORANGE if j in lens else "k", fontweight="bold" if j in lens else "normal")
            else: a_.set_xticklabels([])
            if j == 0 and i > 0:
                yt, yl = ticks(mu_u[i], sig[i]); a_.set_yticks(yt); a_.set_yticklabels(yl, fontsize=10)
                a_.set_ylabel(LABEL.get(use[i], use[i]), fontsize=15, color=ORANGE if i in lens else "k", fontweight="bold" if i in lens else "normal")
            elif i != j: a_.set_yticks(ticks(mu_u[i], sig[i])[0]); a_.set_yticklabels([])
            a_.tick_params(length=3.5, width=0.8)
    txt = [f"{r['lens']}   z={r['z']:g}   M_tot={r['m_tot']:.0e}   q={r['q']:g}", f"M_Lz={r['mass_lz']:.0e}   y={r['y']:g}",
           f"SNR = {r['snr_lensed']:.0f}   (lensing imprint {r['snr_diff']:.0f})", f"cond(F) = {cond:.1e}", ""]
    txt += ["median  +upper -lower   (90% range = 5%..95%):"]
    txt += [f"{p_:<13s} " + fmt90(*m_, tex=False) + ("   <- lens" if p_ in LENSN else "") for p_, m_ in zip(use, m90)]
    fig.text(0.975, 0.905, "\n".join(txt), ha="right", va="top", multialignment="left", family="monospace", fontsize=12.5, color="0.15")
    fig.suptitle("Fisher Gaussian posterior: titles = median (+upper -lower) of the 90% range;  solid: 90% region, dashed: 68%;  1D dashed: 5%/95%;  red: injected;  orange: lens",
                 fontsize=12, y=0.985, color="0.25")
    fig.savefig(os.path.join(a.out, fname), dpi=130); plt.close(fig)


rng = np.random.default_rng(a.shard)
if a.all:
    cand = [p for p in pts if (a.match is None or any(m == tag(p[0]) for m in a.match))]
    mine = sorted(cand, key=lambda p: (p[0]["lens"], tag(p[0])))[a.shard::a.nshards]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} points", flush=True)
    base_out = a.out
    for k, (r, names, F) in enumerate(mine):
        a.out = os.path.join(base_out, r["lens"].replace(":", "")); os.makedirs(a.out, exist_ok=True)
        t = tag(r).replace(":", "")
        try:
            if "full" in a.only: plot(r, names, F, list(names), f"corner_full_{t}.png", rng)
            if "lens" in a.only: plot(r, names, F, [p_ for p_ in a.compact if p_ in names], f"corner_lens_{t}.png", rng)
        except Exception as e: print("FAILED (non-fatal)", t, repr(e), flush=True)
        print(f"  [{k + 1}/{len(mine)}] {t}", flush=True)
    print("done ->", base_out); sys.exit(0)

stable = sorted([p for p in pts if p[0]["cond"] < a.cond_max], key=lambda p: p[0]["sig_y"])
pick = stable[:a.n]
if stable: pick.append(max(stable, key=lambda p: p[0]["sig_y"] / p[0]["y"]))
pick.append(max(pts, key=lambda p: p[0]["cond"]))
rng = np.random.default_rng(0); done = set()
for r, names, F in pick:
    if tag(r) in done: continue
    done.add(tag(r))
    plot(r, names, F, [p for p in names], f"corner_full_{tag(r)}.png", rng)
    plot(r, names, F, [p for p in a.compact if p in names], f"corner_lens_{tag(r)}.png", rng)
    print("corner", tag(r), flush=True)
print("done ->", a.out)
