"""Fisher confidence contours (1 and 2 sigma Gaussian ellipses) from the stored Fisher matrices.

  corner_*.png     : corner plot for one (source, lens) point: lens params + key MBH params, joint (marginalised over
                     everything) in blue; MBH-only/lens-known in orange where both parameters are MBH ones.
  lensplane_*.png  : (log10 M_Lz, y) 1/2-sigma ellipses of every lens grid point for one source (ellipses magnified
                     by the stated factor, otherwise invisible).
    python scripts/fisher_contours.py --shards DIR --out DIR [--n 3]
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys; sys.path.insert(0, ROOT)
from lisa_mbhb.fisher import covariance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=3, help="corner plots for the n best-measured stable points (+ extras)")
ap.add_argument("--params", nargs="+", default=["log10_mlz", "log10_mhalo", "y", "log10_mchirp", "q", "log10_dl", "cos_inc"])
ap.add_argument("--cond-max", type=float, default=1e8)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
DCHI = {1: 2.30, 2: 6.18}                       # 68% / 95% for 2 dof

pts = []
for f in sorted(glob.glob(os.path.join(a.shards, "fisher*.json"))):
    d = json.load(open(f))
    for r, m in zip(d["rows"], d["fisher"]):
        if r.get("ok") and m: pts.append((r, m["names"], np.array(m["F"])))
if not pts:
    fig = plt.figure(figsize=(5, 2)); fig.text(.5, .5, "no Fisher points", ha="center"); fig.savefig(os.path.join(a.out, "corner_none.png")); raise SystemExit("no points")


def ellipse(ax, cov2, color, **kw):
    w, v = np.linalg.eigh(cov2); w = np.clip(w, 0, None); ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    for k, ls in ((2, "--"), (1, "-")):
        ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(w[1] * DCHI[k]), 2 * np.sqrt(w[0] * DCHI[k]), angle=ang, fill=False, color=color, ls=ls, **kw))


def tag(r): return f"{r['lens']}_z{r['z']:g}_M{r['m_tot']:.0e}_q{r['q']:g}_L{r['mass_lz']:.0e}_y{r['y']:g}".replace("+", "")


def corner(r, names, F):
    P = [p for p in a.params if p in names]; ix = [names.index(p) for p in P]; n = len(P)
    C, cond, _ = covariance(F); Cj = C[np.ix_(ix, ix)]
    nm = [i for i, k in enumerate(names) if k not in ("log10_mlz", "y", "log10_mhalo")]
    Ck, _, _ = covariance(F[np.ix_(nm, nm)]); mpos = {names[i]: j for j, i in enumerate(nm)}   # lens known
    fig, axs = plt.subplots(n, n, figsize=(2.3 * n, 2.3 * n), squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            if j > i: ax.axis("off"); continue
            s = np.sqrt(Cj[j, j]); lim = 3.2 * s
            if i == j:
                x = np.linspace(-lim, lim, 200); ax.plot(x, np.exp(-x**2 / (2 * s**2)), "C0")
                if P[i] in mpos: sk = np.sqrt(Ck[mpos[P[i]], mpos[P[i]]]); ax.plot(x, np.exp(-x**2 / (2 * sk**2)), "C1")
                ax.set_yticks([]); ax.set_xlim(-lim, lim); ax.set_title(f"$\\sigma$={s:.2g}", fontsize=7)
            else:
                ellipse(ax, Cj[np.ix_([j, i], [j, i])], "C0")
                if P[i] in mpos and P[j] in mpos:
                    ellipse(ax, Ck[np.ix_([mpos[P[j]], mpos[P[i]]], [mpos[P[j]], mpos[P[i]]])], "C1")
                ax.plot(0, 0, "k+"); si = np.sqrt(Cj[i, i]); ax.set_xlim(-lim, lim); ax.set_ylim(-3.2 * si, 3.2 * si)
            if i == n - 1: ax.set_xlabel(f"$\\delta$ {P[j]}", fontsize=7)
            else: ax.set_xticklabels([])
            if j == 0 and i > 0: ax.set_ylabel(f"$\\delta$ {P[i]}", fontsize=7)
            elif j > 0: ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
            try: ax.ticklabel_format(style="sci", scilimits=(-2, 2))
            except AttributeError: pass            # axes whose tick labels were blanked have a fixed formatter
    fig.suptitle(f"{r['lens']} z={r['z']:g} M={r['m_tot']:.0e} q={r['q']:g} | M_Lz={r['mass_lz']:.0e} y={r['y']:g} | SNR={r['snr_lensed']:.0f} cond={cond:.1e}\n"
                 "blue: joint (marginalised over lens + MBH), orange: lens known; solid 1$\\sigma$, dashed 2$\\sigma$", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"corner_{tag(r)}.png"), dpi=130); plt.close(fig)


stable = sorted([p for p in pts if p[0]["cond"] < a.cond_max], key=lambda p: p[0]["sig_y"])
pick = stable[:a.n]
if stable: pick.append(max(stable, key=lambda p: p[0]["sig_y"] / p[0]["y"]))                    # worst-measured stable
pick.append(max(pts, key=lambda p: p[0]["cond"]))                                                 # most degenerate overall
done = set()
for r, names, F in pick:
    if tag(r) in done: continue
    done.add(tag(r)); corner(r, names, F); print("corner", tag(r))

# lens plane: ellipses of all lens points of one source
srcs = sorted({(r["lens"], r["z"], r["m_tot"], r["q"], r["inc"]) for r, _, _ in pts})
for s in srcs:
    sel = [(r, n, F) for r, n, F in pts if (r["lens"], r["z"], r["m_tot"], r["q"], r["inc"]) == s]
    if len(sel) < 2 or "log10_mlz" not in sel[0][1]: continue
    ells = []
    for r, names, F in sel:
        C, _, _ = covariance(F); i = [names.index("log10_mlz"), names.index("y")]; ells.append((r, C[np.ix_(i, i)]))
    big = max(np.sqrt(np.linalg.eigvalsh(c)[-1]) for _, c in ells)
    xs = np.log10(sorted({r["mass_lz"] for r, _ in ells})); ys = sorted({r["y"] for r, _ in ells})
    mag = 0.12 * min(np.ptp(xs) if len(xs) > 1 else 1, np.ptp(ys) if len(ys) > 1 else 1) / (np.sqrt(DCHI[2]) * big)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for r, c in ells:
        w, v = np.linalg.eigh(c); ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        for k, ls in ((2, "--"), (1, "-")):
            ax.add_patch(Ellipse((np.log10(r["mass_lz"]), r["y"]), 2 * mag * np.sqrt(max(w[1], 0) * DCHI[k]), 2 * mag * np.sqrt(max(w[0], 0) * DCHI[k]), angle=ang, fill=(k == 1), alpha=.5 if k == 1 else 1, ec="C0", fc="C0", ls=ls))
        ax.plot(np.log10(r["mass_lz"]), r["y"], "k.", ms=3)
    ax.set_xlim(xs.min() - .5, xs.max() + .5); ax.set_ylim(min(ys) - .5, max(ys) + .5)
    ax.set_xlabel("log10 $M_{Lz}$"); ax.set_ylabel("y"); ax.set_title(f"{s[0]} z={s[1]:g} M={s[2]:.0e} q={s[3]:g}: 1/2$\\sigma$ ellipses x{mag:.0f}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"lensplane_{s[0]}_z{s[1]:g}_M{s[2]:.0e}_q{s[3]:g}.png".replace("+", "")), dpi=130); plt.close(fig)
print("done ->", a.out)
