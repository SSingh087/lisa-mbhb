"""How well the LENS parameters are constrained across the whole parameter space (from fisher_90pct.csv; no new computation).
Three lens types: point mass (pm), point mass + NFW halo (nfw:*), halo only (halo:*).
Quantity: 90% half-width = (p95 - p05)/2 of the lens-parameter posterior; y in relative terms (%), log10 masses in dex.

  ppd_bands_<type>.png : half-width vs each injected parameter (y, M_Lz, M_tot, z); line = median over all other parameters,
                         dark band = 16-84%, light band = min-max (the spread over the rest of the space)
  ppd_density_<type>.png: probability density of log10(half-width) over the whole parameter space, per lens parameter,
                         split by redshift (filled violins) - i.e. the "posterior distribution of the constraint"
  ppd_summary.png      : all three lens types side by side
    python scripts/fisher_ppd.py --table fisher_90pct.csv --out DIR
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ap = argparse.ArgumentParser(); ap.add_argument("--table", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rows = [{k: (float(v) if k not in ("lens", "param") else v) for k, v in r.items()} for r in csv.DictReader(open(a.table))]
TYPES = {"pm": ("Point mass", ["pm"], ["log10_mlz", "y"]),
         "nfw": ("Point mass + NFW halo", ["nfw:1e9", "nfw:1e11"], ["log10_mlz", "y", "log10_mhalo"]),
         "halo": ("NFW halo only", ["halo:1e9", "halo:1e11"], ["y", "log10_mhalo"])}
LAB = dict(log10_mlz=r"$\log_{10}M_{Lz}$ [dex]", y=r"$y$ [relative, %]", log10_mhalo=r"$\log_{10}M_{halo}$ [dex]")
XV = [("y_src", "injected $y$"), ("mass_lz", "injected $M_{Lz}$"), ("m_tot", "source $M_{tot}$"), ("z", "source $z$")]
COL = {"pm": "#2f6db5", "nfw:1e9": "#e8710a", "nfw:1e11": "#b34700", "halo:1e9": "#2a9d8f", "halo:1e11": "#14625a"}


def hw(r, p):
    h = 0.5 * (r["p95"] - r["p05"])
    return 100 * h / abs(r["truth"]) if p == "y" else h        # y in %, log10 masses in dex


# per (lens, param) list of point dicts
D = {}
for r in rows:
    D.setdefault((r["lens"], r["param"]), []).append(dict(r, hw=hw(r, r["param"])))
plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8})


def bands(tkey):
    title, lenses, params = TYPES[tkey]
    fig, axs = plt.subplots(len(params), len(XV), figsize=(4.2 * len(XV), 3.4 * len(params)), squeeze=False, sharey="row")
    for i, p in enumerate(params):
        for j, (xk, xl) in enumerate(XV):
            ax = axs[i, j]
            for l in lenses:
                pts = D.get((l, p), [])
                if not pts: continue
                xs = sorted({q[xk] for q in pts}); c = COL[l]
                g = [np.array([q["hw"] for q in pts if q[xk] == x]) for x in xs]
                lo, hi = [np.percentile(v, 16) for v in g], [np.percentile(v, 84) for v in g]
                ax.fill_between(xs, [v.min() for v in g], [v.max() for v in g], color=c, alpha=0.13, lw=0)
                ax.fill_between(xs, lo, hi, color=c, alpha=0.30, lw=0)
                ax.plot(xs, [np.median(v) for v in g], "o-", color=c, lw=2, ms=5, label=l)
            ax.set_xscale("log" if xk in ("mass_lz", "m_tot") else "linear"); ax.set_yscale("log")
            if xk == "y_src": ax.set_xscale("log")
            ax.grid(alpha=0.25, which="both"); ax.set_xlabel(xl)
            if j == 0: ax.set_ylabel("90% half-width\n" + LAB[p], fontsize=10)
            if i == 0 and j == 0: ax.legend(fontsize=9)
    fig.suptitle(f"{title}: lens-parameter constraint across the parameter space\nline = median over all other parameters, dark band = 16-84%, light band = min-max", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"ppd_bands_{tkey}.png"), dpi=130); plt.close(fig)


def density(ax, tkey, p, zs=(1.0, 3.0)):
    title, lenses, _ = TYPES[tkey]; ypos = 0
    ticks, labs = [], []
    allv = [np.log10(q["hw"]) for l in lenses for q in D.get((l, p), []) if q["hw"] > 0]
    if not allv: return
    grid = np.linspace(min(allv) - 0.3, max(allv) + 0.3, 300)
    for l in lenses:
        for z, off in zip(zs, (0.0, 1.0)):
            v = np.array([np.log10(q["hw"]) for q in D.get((l, p), []) if q["z"] == z and q["hw"] > 0])
            if len(v) < 3: continue
            k = gaussian_kde(v)(grid); k = k / k.max() * 0.42
            ax.fill_between(grid, ypos - k, ypos + k, color=COL[l], alpha=0.35 + 0.3 * off, lw=0)
            ax.plot(grid, ypos + k, color=COL[l], lw=1); ax.plot(grid, ypos - k, color=COL[l], lw=1)
            ax.plot(np.median(v), ypos, "w|", ms=14, mew=2.5); ax.plot(np.median(v), ypos, "k|", ms=12, mew=1.2)
            ticks.append(ypos); labs.append(f"{l}  z={z:g}"); ypos += 1
        ypos += 0.3
    ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=9); ax.invert_yaxis(); ax.grid(alpha=0.25, axis="x")
    ax.set_xlabel("log$_{10}$ of 90% half-width  (" + LAB[p].split("[")[1].rstrip("]") + ")")
    ax.set_title(LAB[p].split(" [")[0], fontsize=11)


def densities(tkey):
    title, lenses, params = TYPES[tkey]
    fig, axs = plt.subplots(1, len(params), figsize=(6.2 * len(params), 3.6), squeeze=False)
    for ax, p in zip(axs[0], params): density(ax, tkey, p)
    fig.suptitle(f"{title}: distribution of the lens-parameter constraint over the whole parameter space (median marked)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"ppd_density_{tkey}.png"), dpi=130); plt.close(fig)


for t in TYPES:
    bands(t); densities(t); print("wrote", t, flush=True)
# summary: y constraint for all three types together + the mass constraints
fig, axs = plt.subplots(1, 3, figsize=(18, 4.2))
for ax, (p, ttl) in zip(axs, [("y", "$y$"), ("log10_mlz", "$\\log_{10}M_{Lz}$"), ("log10_mhalo", "$\\log_{10}M_{halo}$")]):
    k = 0; ticks = []; labs = []
    allv = [np.log10(q["hw"]) for l in COL for q in D.get((l, p), []) if q["hw"] > 0]
    if not allv: continue
    grid = np.linspace(min(allv) - .3, max(allv) + .3, 300)
    for l in COL:
        v = np.array([np.log10(q["hw"]) for q in D.get((l, p), []) if q["hw"] > 0])
        if len(v) < 3: continue
        d = gaussian_kde(v)(grid); d = d / d.max() * 0.42
        ax.fill_between(grid, k - d, k + d, color=COL[l], alpha=0.4); ax.plot(grid, k + d, color=COL[l]); ax.plot(grid, k - d, color=COL[l])
        ax.plot(np.median(v), k, "k|", ms=12, mew=1.5); ticks.append(k); labs.append(l); k += 1
    ax.set_yticks(ticks); ax.set_yticklabels(labs); ax.invert_yaxis(); ax.set_title(ttl); ax.grid(alpha=.25, axis="x")
    ax.set_xlabel("log$_{10}$ 90% half-width (" + ("relative %" if p == "y" else "dex") + ")")
fig.suptitle("Lens-parameter constraint over the whole parameter space, all lens types (median marked)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "ppd_summary.png"), dpi=130); plt.close(fig); print("done")
