"""Corner plots for the fisher_lenses Fisher study (Fisher only, Gaussian ellipses, marginalised).  CLUSTER JOB (light).

    python scripts/fisher_study_corner.py --fisher DIR --out DIR --source caliskan --m-ref 1e7 [--part all|lens|binary] [--ncpu 4]

For every truth point (y, M200, kappa) of the given source and m_ref:
  corner_<model>_<tag>.png    one model: all binary + lens parameters + derived quantities
  corner_overlay_<tag>.png    pml, nfw, pmlnfw overlaid over the union of all their parameters
Style: free (sampled) parameters solid, DERIVED quantities dashed (and marked 'derived' on the axis label);
       single-model plots: lens panels orange, binary panels blue;  overlays: colour = model, axis-label colour = lens / binary.
Derived quantities (lisa_mbhb.fisher_lenses.FisherLens.derived): log10 M200, log10 c200, f_NFW, and the mass-sheet invariants
  y/sqrt(1-k), gamma_i/(1-k), log10 x_s + 0.5 log10(1-k), log10 D_L + log10(1-k), t_c - t_unit C.
Their covariance is J C J^T; when the matrix is singular along the exact mass-sheet direction (cond > --cond-ms) C is the
pseudo-inverse without that direction (the invariants have no component along it), and the free parameters that move with
kappa are drawn with their (huge) regular-inverse sigma and labelled 'unconstrained'.
Diagonal titles: 90% half-width (1.645 sigma).  Red: injected values.
"""
import argparse, textwrap, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lisa_mbhb import fisher as FI  # noqa: E402
from lisa_mbhb.fisher_lenses import FisherLens, derived_jacobian  # noqa: E402
from lisa_mbhb import fisher_analysis as FA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fisher", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--source", required=True); ap.add_argument("--m-ref", type=float, required=True)
ap.add_argument("--ncpu", type=int, default=1); ap.add_argument("--cond-ms", type=float, default=1e11)
ap.add_argument("--binary", nargs="+", default=None, help="binary parameters to show (default: all)")
ap.add_argument("--chunk", default="0/1", help="i/n: draw only every n-th truth point starting at i (several small jobs)")
ap.add_argument("--point", nargs=3, type=float, default=None, metavar=("Y", "M200", "KAPPA"), help="only this truth point (M200 ignored for pml)")
ap.add_argument("--models", nargs="+", default=None, help="only these lens models")
ap.add_argument("--free-only", action="store_true", help="free (Fisher) parameters only, no derived quantities")
ap.add_argument("--fontscale", type=float, default=1.0, help="multiplies every font size and the panel size")
ap.add_argument("--unlensed-only", action="store_true", help="draw only the unlensed corner of the source, then stop")
ap.add_argument("--part", default="all", choices=["all", "lens", "binary"], help="all parameters, lens parameters only (binary marginalised), or binary parameters only (lens marginalised)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

LABEL = dict(log10_mchirp=r"$\log_{10}\mathcal{M}_z$", q="$q$", spin1z=r"$\chi_1$", spin2z=r"$\chi_2$", log10_dl=r"$\log_{10}D_L$",
             cos_inc=r"$\cos\iota$", coa_phase=r"$\phi_c$", psi=r"$\psi$", ecl_lon=r"$\lambda$", sin_ecl_lat=r"$\sin\beta$",
             tc=r"$t_c$ [s]", log10_mlz=r"$\log_{10}M_{Lz}$", y="$y$", kappa=r"$\kappa$", gamma1=r"$\gamma_1$", gamma2=r"$\gamma_2$",
             log10_psi0=r"$\log_{10}\psi_0$", log10_xs=r"$\log_{10}x_s$",
             log10_m200=r"$\log_{10}M_{200}$", log10_kappa_s=r"$\log_{10}\kappa_s$", log10_c200=r"$\log_{10}c_{200}$", f_nfw=r"$f_{\rm NFW}$", log10_mbh_m200=r"$\log_{10}(M_{\rm BH}/M_{200})$", gamma_abs=r"$|\gamma|$",
             y_ms=r"$y/\sqrt{1-\kappa}$", gamma1_ms=r"$\gamma_1/(1-\kappa)$", gamma2_ms=r"$\gamma_2/(1-\kappa)$",
             log10_xs_ms=r"$\log_{10}x_s\sqrt{1-\kappa}$", log10_dl_ms=r"$\log_{10}D_L(1-\kappa)$", tc_ms=r"$t_c-t_uC$ [s]")
LENS = {"log10_mbh_m200", "gamma_abs", "log10_mlz", "y", "kappa", "gamma1", "gamma2", "log10_psi0", "log10_xs", "log10_m200", "log10_c200", "log10_kappa_s", "f_nfw",
        "y_ms", "gamma1_ms", "gamma2_ms", "log10_xs_ms"}
DERIVED = {"log10_mbh_m200", "gamma_abs", "log10_m200", "log10_c200", "log10_kappa_s", "f_nfw", "y_ms", "gamma1_ms", "gamma2_ms", "log10_xs_ms", "log10_dl_ms", "tc_ms"}
MODEL_COL = dict(pml="#d62728", nfw="#1f77b4", pmlnfw="#2ca02c", unlensed="#7f7f7f")
LENS_COL, BIN_COL = "#e8710a", "#2f6db5"
Z90, R68, R90 = 1.6448536269514722, np.sqrt(2.2788), np.sqrt(4.6052)


def cov_with_derived(r, model, zl_frac=0.5):
    """(names, mu, C, unconstrained set) for free + derived quantities of one Fisher row (lisa_mbhb.fisher_analysis.analyse):
    kappa free = exact mass-sheet degeneracy; quantities moving along the ANALYTIC mass-sheet direction are unconstrained, the exact
    invariants get the covariance with kappa held fixed (they do not depend on it)."""
    res = FA.analyse(r, model, {"z": r_zs}, r_mref)
    names, C = res["_names"], res["_C"]; unc = {n for n, v in res.items() if not n.startswith("_") and not np.isfinite(v[1])}
    mu = np.array([res[n][0] for n in names])
    if model == "unlensed":
        return names, mu, C, unc
    Lm = FisherLens(model, r_mref, r_zs, zl_frac)
    dn, dv, J = derived_jacobian(Lm, r["vals"], {k: r["binary"][k] for k in ("log10_dl", "tc")}, r["names"])
    Jk = J[:, [r["names"].index(n) for n in names]]; Cx = Jk @ C
    return names + dn, np.concatenate([mu, dv]), np.block([[C, Cx.T], [Cx, Cx @ Jk.T]]), unc


def ell(C2, mu, rad, n=200):
    w, U = np.linalg.eigh(C2); t = np.linspace(0, 2 * np.pi, n)
    xy = (U * np.sqrt(np.clip(w, 0, None))) @ np.array([np.cos(t), np.sin(t)]) * rad
    return mu[0] + xy[0], mu[1] + xy[1]


def fmt(x, lo, hi):
    d = int(np.clip(np.ceil(-np.log10(0.35 * (hi - lo))) + 1, 0, 9)); return f"{x:.{d}f}"


def ci_text(m, hw):
    """'median ± half-width' of the 90% CI (Gaussian Fisher: median = mean) with a common number of decimals."""
    d = int(np.clip(np.ceil(-np.log10(hw)) + 1, 0, 9)) if hw > 0 else 3
    return f"{m:.{d}f}\n± {hw:.{d}f}"


def corner(sets, use, fname, title, outdir=None):
    """sets: list of dict(label, color or None, names, mu, C, unc).  color None -> lens/binary colouring (single model)."""
    dropped = [p for p in use if all(p not in st["names"] or p in st["unc"] for st in sets)]      # unconstrained in every set: not drawn
    use = [p for p in use if p not in dropped]
    if dropped: title += "\nnot shown (unconstrained: mass-sheet degeneracy or unresolved Fisher direction): " + ", ".join(dropped)
    n = len(use); fs = a.fontscale; plt.rcParams.update({"font.size": 18 * fs, "mathtext.fontset": "cm"})
    pw = (2.7 if n <= 10 else 1.9) * fs                    # smaller panels for many parameters -> text relatively larger
    W = pw * n + 2.5
    title = title + "\ncontours: 90% credible region;  diagonal titles: median ± 90% CI (±1.645σ);  marginalised Fisher"
    ncar = max(30, int(W * 72 / (18 * fs * 0.55)))           # characters per line that fit the figure width
    lines = [w for ln in title.split("\n") for w in (textwrap.wrap(ln, ncar, break_on_hyphens=False) or [""])]
    head = len(lines) * 18 * fs * 1.35 / 72                  # inches of title text
    diag = 2 * 20 * fs * 1.35 / 72 + 0.2                     # inches for the two-line diagonal titles of the first row
    lm, bm = 1.45 * fs, 1.75 * fs                            # inches for y labels (left) and rotated tick + x labels (bottom)
    leg = 2 * 14 * fs * 1.5 / 72 if n <= 4 else 0.0          # small corner: one legend band (2 rows) under the title
    W = W + lm - 0.10 * W; H = W + head + leg + diag + bm - 0.10 * W
    top = 1 - (head + leg + diag + 0.2) / H
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(n, n, left=lm / W, right=0.985, bottom=bm / H, top=top, wspace=0.08, hspace=0.08 + (0.25 if n <= 4 else 0))
    def s_of(st, p): i = st["names"].index(p); return np.sqrt(max(st["C"][i, i], 0))
    def m_of(st, p): return st["mu"][st["names"].index(p)]
    rng = []
    for p in use:
        ok = [st for st in sets if p in st["names"] and p not in st["unc"]]
        if not ok:                                   # unconstrained in every set: nominal range around the truth
            mu0 = m_of(next(st for st in sets if p in st["names"]), p); rng.append((mu0 - 1.0, mu0 + 1.0)); continue
        ss = [s_of(st, p) for st in ok]; s = min(max(ss), 20 * min(ss)) if min(ss) > 0 else max(ss)
        mu0 = m_of(ok[0], p); rng.append((mu0 - 3.5 * s, mu0 + 3.5 * s))
    for i in range(n):
        for j in range(i + 1):
            ax = fig.add_subplot(gs[i, j]); pi, pj = use[i], use[j]
            ls = "--" if (pi in DERIVED or pj in DERIVED) else "-"
            if len(sets) == 1 and (pi in sets[0]["unc"] or pj in sets[0]["unc"]):
                ax.text(0.5, 0.5, "unconstrained", ha="center", va="center", transform=ax.transAxes, fontsize=12 * fs, color="0.4")
            for st in sets:
                if pi not in st["names"] or pj not in st["names"] or pi in st["unc"] or pj in st["unc"]: continue
                col = st["color"] or (LENS_COL if (pi in LENS or pj in LENS) else BIN_COL)
                mj, mi = m_of(st, pj), m_of(st, pi)
                if i == j:
                    s = s_of(st, pj); x = np.linspace(*rng[j], 400); g = np.exp(-0.5 * ((x - mj) / max(s, 1e-300)) ** 2)
                    ax.plot(x, g, color=col, lw=2.0, ls=ls); ax.fill_between(x, g, color=col, alpha=0.10)
                else:
                    ii, jj = st["names"].index(pi), st["names"].index(pj)
                    C2 = st["C"][np.ix_([jj, ii], [jj, ii])]
                    ax.plot(*ell(C2, (mj, mi), R90), color=col, lw=2.0, ls=ls)        # 90% credible region only
                    ax.axvline(mj, color="crimson", lw=1.1, zorder=4); ax.axhline(mi, color="crimson", lw=1.1, zorder=4)   # injected: cross hair
            if i == j:
                ax.axvline(m_of(next(st for st in sets if pj in st["names"]), pj), color="crimson", lw=1.2)
                ax.set_yticks([]); ax.set_ylim(0, 1.15)
                tt = []
                for st in sets:
                    if pj in st["names"]:
                        v = s_of(st, pj) * Z90; tt.append(("unconstr." if pj in st["unc"] else ci_text(m_of(st, pj), v)) if len(sets) == 1 else
                                                         f"{st['label'][:4]} " + ("unc." if pj in st["unc"] else f"±{v:.2g}"))
                ax.set_title("  ".join(tt), fontsize=(12 if len(sets) > 1 else 21) * fs, pad=6)
            else:
                ax.set_ylim(rng[i])
            ax.set_xlim(rng[j])
            xt = [rng[j][0] + (rng[j][1] - rng[j][0]) * k for k in (0.15, 0.5, 0.85)]; ax.set_xticks(xt)
            lab = lambda p: LABEL.get(p, p) + ("\n(derived)" if p in DERIVED else "")
            if i == n - 1:
                ax.set_xticklabels([fmt(x, *rng[j]) for x in xt], rotation=40, ha="right", fontsize=19 * fs)
                ax.set_xlabel(lab(pj), fontsize=26 * fs, color=LENS_COL if pj in LENS else BIN_COL)
            else: ax.set_xticklabels([])
            if j == 0 and i > 0:
                yt = [rng[i][0] + (rng[i][1] - rng[i][0]) * k for k in (0.15, 0.5, 0.85)]; ax.set_yticks(yt)
                ax.set_yticklabels([fmt(y, *rng[i]) for y in yt], fontsize=19 * fs)
                ax.set_ylabel(lab(pi), fontsize=26 * fs, color=LENS_COL if pi in LENS else BIN_COL)
            elif i != j: ax.set_yticks([])
    h = ([Line2D([], [], color=st["color"], lw=2.5, label=st["label"]) for st in sets] if sets[0]["color"] else
         [Line2D([], [], color=LENS_COL, lw=2.5, label="lens parameters"), Line2D([], [], color=BIN_COL, lw=2.5, label="binary parameters")])
    h += [Line2D([], [], color="k", lw=2, ls="-", label="free (Fisher) parameter"), Line2D([], [], color="k", lw=2, ls="--", label="derived quantity"),
          Line2D([], [], color="crimson", lw=1.5, label="injected (cross hair)")]
    if n <= 4:   # small corner: the empty upper triangle is too small, put the legend in one row under the title
        fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, top + (diag + 0.05) / H), ncol=3, fontsize=14 * fs, frameon=False)
    else:
        fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.985, top), fontsize=20 * fs, frameon=False)
    fig.text(0.5, 1 - 0.15 / H, "\n".join(lines), ha="center", va="top", fontsize=18 * fs)
    od = outdir or a.out; os.makedirs(od, exist_ok=True)
    fig.savefig(os.path.join(od, fname), dpi=110); plt.close(fig)
    ci = {}                                                           # 90% credible intervals of every plotted parameter
    for st in sets:
        ci[st["label"]] = {p: (dict(unconstrained=True) if p in st["unc"] else dict(mu=float(m_of(st, p)), lo=float(m_of(st, p) - Z90 * s_of(st, p)),
                            hi=float(m_of(st, p) + Z90 * s_of(st, p)), half_width=float(Z90 * s_of(st, p)), derived=p in DERIVED))
                           for p in use if p in st["names"]}
    json.dump(dict(title=title, level=0.90, ci=ci), open(os.path.join(od, fname.replace(".png", ".ci90.json")), "w"), indent=1)


# ---------------- load ----------------
pts, unl = {}, None
for f in sorted(glob.glob(os.path.join(a.fisher, "fisher_study*.json"))):
    d = json.load(open(f))
    if d["info"]["source"] != a.source: continue
    if d["model"] == "unlensed":
        unl = (d["rows"][0], d); continue
    if abs(d["m_ref"] / a.m_ref - 1) > 1e-9: continue
    for r in d["rows"]:
        if r.get("ok"): pts.setdefault((r["y"], r["m200"] if d["model"] != "pml" else None, r["kappa"]), {})[d["model"]] = (r, d)
if not pts:
    fig = plt.figure(figsize=(6, 2)); fig.text(.5, .5, f"no Fisher points for {a.source} m_ref={a.m_ref:g}", ha="center")
    fig.savefig(os.path.join(a.out, f"corner_none_{a.source}_ref{a.m_ref:.0e}.png")); sys.exit("no points")
d0 = next(iter(next(iter(pts.values())).values()))[1]; r_zs, r_mref = d0["info"]["z"], d0["m_ref"]
PART = a.part
sel_names = lambda names: [n for n in names if (PART == "all" and (n in LENS or n in DERIVED or a.binary is None or n in a.binary))
                           or (PART == "lens" and n in LENS)
                           or (PART == "binary" and n not in LENS and (a.binary is None or n in a.binary or n in DERIVED))]
OUT = os.path.join(a.out, PART)
unl_set = None
CI, CN = (int(x) for x in a.chunk.split("/"))
if unl is not None and PART != "lens" and CI == 0 and not a.models:     # the unlensed binary Fisher: one corner per source, and part of every overlay
    un, umu, uC, uunc = cov_with_derived(unl[0], "unlensed")
    unl_set = dict(label="unlensed", color=MODEL_COL["unlensed"], names=un, mu=umu, C=uC, unc=uunc)
    corner([dict(unl_set, color=None)], sel_names(un), f"corner_unlensed_{a.source}.png",
           f"unlensed: {a.source}   SNR {unl[0]['snr_unlensed']:.0f}  cond {unl[0]['cond']:.1e}", outdir=OUT)

if a.unlensed_only: sys.exit(0)

# pml truth has no halo: overlay each (y, M200, kappa) halo truth with the pml point at the same (y, kappa)
jobs = []
for (y, m2, k), models in pts.items():
    if m2 is None: continue
    sel = dict(models); pml = pts.get((y, None, k), {}).get("pml")
    if pml: sel["pml"] = pml
    jobs.append(((y, m2, k), sel))
for (y, m2, k), models in pts.items():
    if m2 is None and not any(kk[0] == y and kk[2] == k and kk[1] is not None for kk in pts): jobs.append(((y, None, k), models))

if a.point: jobs = [j for j in jobs if abs(j[0][0] - a.point[0]) < 1e-9 and abs(j[0][2] - a.point[2]) < 1e-9 and (j[0][1] is None or abs(j[0][1] / a.point[1] - 1) < 1e-6)]
jobs = sorted(jobs, key=lambda j: (j[0][0], j[0][1] or 0, j[0][2]))[CI::CN]


def work(job):
    (y, m2, k), models = job
    tg = f"{a.source}_ref{a.m_ref:.0e}_y{y:g}_k{k:g}" + (f"_m{m2:.0e}" if m2 else "")
    sets, done = [], []
    for model in ("pml", "nfw", "pmlnfw"):
        if model not in models or (a.models and model not in a.models): continue
        r, d = models[model]
        names, mu, C, unc = cov_with_derived(r, model)
        use = [n for n in sel_names(names) if not (a.free_only and n in DERIVED)]
        sets.append(dict(label=model, color=MODEL_COL[model], names=names, mu=mu, C=C, unc=unc))
        fn = (f"corner_{PART}_pml_{a.source}_ref{a.m_ref:.0e}_y{y:g}_k{k:g}.png" if model == "pml" and m2 else f"corner_{PART}_{model}_{tg}.png")
        if model == "pml" and m2 and os.path.exists(os.path.join(OUT, fn)): continue      # the pml-only corner is the same for every M200
        corner([dict(sets[-1], color=None)], use, fn,
               f"{model} ({PART} parameters): {a.source}  m_ref={a.m_ref:.0e}  y={y:g}  kappa={k:g}" + (f"  M200={m2:.0e}" if m2 and model != "pml" else "")
               + f"\nSNR {r['snr_lensed']:.0f}  cond {r['cond']:.1e}"
               , outdir=OUT)
        done.append(fn)
    if len(sets) > 1:
        allsets = sets + ([unl_set] if unl_set else [])
        union = []
        for st in allsets:
            union += [n for n in sel_names(st["names"]) if n not in union]
        corner(allsets, union, f"corner_{PART}_overlay_{tg}.png", f"overlay {'unlensed / ' if unl_set else ''}pml / nfw / pmlnfw ({PART} parameters): {a.source}  "
               f"m_ref={a.m_ref:.0e}  y={y:g}  kappa={k:g}" + (f"  M200={m2:.0e}" if m2 else ""), outdir=OUT)
        done.append(f"corner_{PART}_overlay_{tg}.png")
    return done


if a.ncpu > 1:
    import multiprocessing as mp
    with ProcessPoolExecutor(a.ncpu, mp_context=mp.get_context("fork")) as ex:
        out = list(ex.map(work, jobs))
else:
    out = [work(j) for j in jobs]
print(f"wrote {sum(len(o) for o in out)} corner plots ({PART}) -> {OUT}")
