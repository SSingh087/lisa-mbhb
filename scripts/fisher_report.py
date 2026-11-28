"""Figures, LaTeX tables and number macros for the Fisher write-up (report/fisher_lensing.tex).  CLUSTER JOB (lens/waveform evaluations).
Reads the shard JSONs, fig3 results and validation output; writes $OUT/{figures,tables}/ and $OUT/tables/numbers.tex.
Every figure is wrapped so that one failure does not stop the others.   python scripts/fisher_report.py --fisher DIR --out DIR"""
import argparse, csv, glob, json, os, sys, traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--fisher", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); FIG = os.path.join(a.out, "figures"); TAB = os.path.join(a.out, "tables"); os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8, "mathtext.fontset": "cm", "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})
LN10 = np.log(10.0); NUM = {}
LENSES = ["pm", "nfw:1e9", "nfw:1e11", "halo:1e9", "halo:1e11"]
TAGS = {"pm": "PM", "nfw:1e9": "NFWnine", "nfw:1e11": "NFWeleven", "halo:1e9": "HALOnine", "halo:1e11": "HALOeleven"}
COL = {"pm": "#2f6db5", "nfw:1e9": "#e8710a", "nfw:1e11": "#b34700", "halo:1e9": "#2a9d8f", "halo:1e11": "#14625a"}
NICE = {"pm": "point mass", "nfw:1e9": r"PM + NFW, $M_{200}=10^{9}$", "nfw:1e11": r"PM + NFW, $M_{200}=10^{11}$", "halo:1e9": r"halo only, $10^{9}$", "halo:1e11": r"halo only, $10^{11}$"}
PLAB = dict(log10_mchirp=r"$\log_{10}\mathcal{M}_z$", q="$q$", spin1z=r"$\chi_1$", spin2z=r"$\chi_2$", log10_dl=r"$\log_{10}D_L$", cos_inc=r"$\cos\iota$", coa_phase=r"$\phi_c$",
            psi=r"$\psi$", ecl_lon=r"$\lambda$", sin_ecl_lat=r"$\sin\beta$", tc=r"$t_c$", log10_mlz=r"$\log_{10}M_{Lz}$", y="$y$", log10_mhalo=r"$\log_{10}M_{halo}$")
SAMPLED = ["log10_mchirp", "q", "spin1z", "spin2z", "log10_dl", "cos_inc", "coa_phase", "psi", "ecl_lon", "sin_ecl_lat", "tc"]


def tex(name, body): open(os.path.join(TAB, name), "w").write(body)
def ts(x, d=2):
    """x -> $m \\times 10^{e}$ (plain number when |e| < 2)"""
    if x == 0 or not np.isfinite(x): return "$0$"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    return f"${x:.{max(d - e - 1, 0)}f}$" if -2 < e < 2 else f"${m:.{d - 1}f}\\times10^{{{e}}}$"


def tsm(x, d=1):
    """number -> 'm\\times10^{e}' (for use inside math mode)"""
    e = int(np.floor(np.log10(abs(x)))); return f"{x / 10 ** e:.{d}f}\\times10^{{{e}}}"


def safe(fn):
    try: fn(); print("ok  ", fn.__name__, flush=True)
    except Exception: print("FAIL", fn.__name__); traceback.print_exc()
def sci(x, d=2):
    if x == 0 or not np.isfinite(x): return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    return f"{m:.{d}f}\\times10^{{{e}}}" if abs(e) >= 3 else f"{x:.{max(d - e, 0) if e < 1 else max(d - e, 0)}g}"


rows, mats = [], []
for f in sorted(glob.glob(os.path.join(a.fisher, "shards", "fisher*.json"))):
    d = json.load(open(f))
    for r, m in zip(d["rows"], d["fisher"]):
        if r.get("ok") and m: rows.append(r); mats.append(m)
print(len(rows), "Fisher points")
key = lambda r: (r["z"], r["m_tot"], r["q"], r["inc"], r["mass_lz"], r["y"])
pm = {key(r): r for r in rows if r["lens"] == "pm"}
bylens = {l: [r for r in rows if r["lens"] == l] for l in LENSES}
NUM["nPoints"] = len(rows); NUM["nPerLens"] = len(bylens["pm"])
med = lambda v: float(np.median(v)) if len(v) else float("nan")


# ---------------------------------------------------------------- tables
def table_summary():
    L = []
    for l in LENSES:
        rs = bylens[l]
        if not rs: continue
        sy = [r["sig_y"] / r["y"] * 100 for r in rs]; sM = [r["sig_log10_mlz"] for r in rs if "sig_log10_mlz" in r]; sH = [r["sig_log10_mhalo"] for r in rs if "sig_log10_mhalo" in r]
        ry = [r["sig_y"] / pm[key(r)]["sig_y"] for r in rs if key(r) in pm]; rM = [r["sig_log10_mlz"] / pm[key(r)]["sig_log10_mlz"] for r in rs if "sig_log10_mlz" in r and key(r) in pm]
        rc = [r["sig_log10_mchirp"] / pm[key(r)]["sig_log10_mchirp"] for r in rs if key(r) in pm]
        st = np.mean([(r["cond"] < 1e8) and r["eig_min"] > 0 for r in rs])
        c = lambda v, f="{:.2g}": (f.format(med(v)) if len(v) else "--")
        nm = NICE[l]
        L.append(f"{nm} & {c([r['snr_diff'] for r in rs], '{:.0f}')} & {c(sy)} & {c(sM)} & {c(sH)} & {c(ry, '{:.2f}') if l != 'pm' else '1'} & {c(rM, '{:.2f}') if l != 'pm' and rM else ('1' if l == 'pm' else '--')} & {c(rc, '{:.2f}')} & {100 * st:.0f}\\% \\\\")
    tex("summary.tex", "\\begin{tabular}{lrrrrrrrr}\n\\toprule\nLens model & $\\rho_{\\rm lens}$ & $\\sigma_y/y$ [\\%] & $\\sigma_{\\log M_{Lz}}$ & $\\sigma_{\\log M_{h}}$ & $\\sigma_y/\\sigma_y^{\\rm PM}$ & $\\sigma_{M_{Lz}}/\\sigma_{M_{Lz}}^{\\rm PM}$ & $\\sigma_{\\mathcal{M}}/\\sigma_{\\mathcal{M}}^{\\rm PM}$ & stable \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    for l in LENSES:
        rs = bylens[l]; t = TAGS[l]
        NUM[f"cond{t}"] = f"{med([r['cond'] for r in rs]):.1e}".replace("e+0", "\\times10^{").replace("e+", "\\times10^{") + "}"
        NUM[f"stable{t}"] = f"{100 * np.mean([(r['cond'] < 1e8) for r in rs]):.0f}"
        NUM[f"sy{t}"] = f"{med([r['sig_y'] for r in rs]):.2g}"
        NUM[f"syrel{t}"] = f"{med([r['sig_y'] / r['y'] * 100 for r in rs]):.2g}"
        if l != "halo:1e9" and l != "halo:1e11": NUM[f"sM{t}"] = f"{med([r['sig_log10_mlz'] for r in rs]):.2g}"
        if l != "pm": NUM[f"sH{t}"] = f"{med([r['sig_log10_mhalo'] for r in rs]):.2g}"; NUM[f"hdet{t}"] = f"{100 * np.mean([r['sig_log10_mhalo'] < 0.1 for r in rs]):.0f}"
        NUM[f"snrd{t}"] = f"{med([r['snr_diff'] for r in rs]):.0f}"
        if l != "pm":
            NUM[f"ratioy{t}"] = f"{med([r['sig_y'] / pm[key(r)]['sig_y'] for r in rs]):.2f}"
            NUM[f"ratiochirp{t}"] = f"{med([r['sig_log10_mchirp'] / pm[key(r)]['sig_log10_mchirp'] for r in rs]):.2f}"
            NUM[f"fracbetterchirp{t}"] = f"{100 * np.mean([r['sig_log10_mchirp'] < pm[key(r)]['sig_log10_mchirp'] for r in rs]):.0f}"
        if "sig_log10_mlz" in bylens[l][0] and l != "pm":
            NUM[f"ratioM{t}"] = f"{med([r['sig_log10_mlz'] / pm[key(r)]['sig_log10_mlz'] for r in rs]):.2f}"
    NUM["snrUnlMed"] = f"{med([r['snr_unlensed'] for r in rows]):.0f}"; NUM["snrUnlMin"] = f"{min(r['snr_unlensed'] for r in rows):.0f}"; NUM["snrUnlMax"] = f"{max(r['snr_unlensed'] for r in rows):.0f}"
    for q in (0.3, 0.8):
        rs = [r for r in rows if r["lens"] == "pm" and r["q"] == q]
        NUM[f"stablePMq{ {0.3: 'Low', 0.8: 'High'}[q] }"] = f"{100 * np.mean([r['cond'] < 1e8 for r in rs]):.0f}"; NUM[f"condPMq{ {0.3: 'Low', 0.8: 'High'}[q] }"] = tsm(med([r['cond'] for r in rs]))
    NUM["maxcond"] = tsm(max(r['cond'] for r in rows)); NUM["minEig"] = tsm(min(r['eig_min'] for r in rows))
    NUM["posdef"] = str(int(sum(r["eig_min"] > 0 for r in rows)))


def extra_numbers():
    for l in LENSES:
        t = TAGS[l]
        for z, zn in ((1.0, "One"), (3.0, "Three")):
            rs = [r for r in bylens[l] if r["z"] == z]; NUM[f"syrel{t}z{zn}"] = f"{med([r['sig_y'] / r['y'] * 100 for r in rs]):.2g}"
            if "sig_log10_mhalo" in rs[0]: NUM[f"sH{t}z{zn}"] = f"{med([r['sig_log10_mhalo'] for r in rs]):.2g}"
            if "sig_log10_mlz" in rs[0]: NUM[f"sM{t}z{zn}"] = f"{med([r['sig_log10_mlz'] for r in rs]):.2g}"
        for M, mn in ((1e6, "six"), (1e7, "seven"), (1e8, "eight")):
            rs = [r for r in bylens[l] if r["mass_lz"] == M]; NUM[f"syrel{t}ML{mn}"] = f"{med([r['sig_y'] / r['y'] * 100 for r in rs]):.2g}"
            if "sig_log10_mlz" in rs[0]: NUM[f"sM{t}ML{mn}"] = f"{med([r['sig_log10_mlz'] for r in rs]):.2g}"
            if "sig_log10_mhalo" in rs[0]: NUM[f"sH{t}ML{mn}"] = f"{med([r['sig_log10_mhalo'] for r in rs]):.2g}"
    mu = lambda y: (y * y + 2) / (y * np.sqrt(y * y + 4)); rs = bylens["pm"]
    dv = lambda sel: max(abs(r["snr_lensed"] / r["snr_unlensed"] / np.sqrt(mu(r["y"])) - 1) for r in rs if sel(r))
    NUM["magMaxYgeOneEight"] = f"{100 * dv(lambda r: r['mass_lz'] == 1e8 and r['y'] >= 1):.2f}"; NUM["magMaxYgeOneSeven"] = f"{100 * dv(lambda r: r['mass_lz'] == 1e7 and r['y'] >= 1):.1f}"
    NUM["magMaxYlowEight"] = f"{100 * dv(lambda r: r['mass_lz'] == 1e8 and r['y'] < 1):.0f}"
    NUM["snrPMmin"] = f"{min(r['snr_diff'] for r in bylens['pm']):.0f}"; NUM["snrPMmax"] = f"{max(r['snr_diff'] for r in bylens['pm']):.0f}"
    st = [r for r in bylens["pm"] if r["cond"] < 1e8]; NUM["bestSigY"] = f"{min(r['sig_y'] for r in st):.1e}"; NUM["bestSigM"] = f"{min(r['sig_log10_mlz'] for r in st):.1e}"
    NUM["worstSigYrel"] = f"{max(r['sig_y'] / r['y'] for r in bylens['pm']) * 100:.0f}"; NUM["fracYunmeasuredHaloNine"] = f"{100 * np.mean([r['sig_y'] / r['y'] > 1 for r in bylens['halo:1e9']]):.0f}"
    NUM["fracYunmeasuredPM"] = f"{100 * np.mean([r['sig_y'] / r['y'] > 1 for r in bylens['pm']]):.0f}"
    NUM["tcRatioNFWnine"] = f"{med([r['sig_tc'] / r['sigU_tc'] for r in bylens['nfw:1e9']]):.1f}"


def table_best():
    L = []
    for l, k in (("pm", 5), ("nfw:1e11", 3), ("halo:1e11", 3)):
        rs = sorted([r for r in bylens[l] if r["cond"] < 1e8], key=lambda r: r["sig_y"])[:k]
        for r in rs:
            sM = f"{r['sig_log10_mlz']:.1e}" if "sig_log10_mlz" in r else "--"; sH = f"{r['sig_log10_mhalo']:.1e}" if "sig_log10_mhalo" in r else "--"
            L.append(f"{NICE[l]} & {r['z']:g} & {ts(r['m_tot'], 1)} & {r['q']:g} & {ts(r['mass_lz'], 1)} & {r['y']:g} & {r['snr_lensed']:.0f} & {ts(r['sig_y'], 2)} & {ts(r['sig_log10_mlz'], 2) if 'sig_log10_mlz' in r else '--'} & {ts(r['sig_log10_mhalo'], 2) if 'sig_log10_mhalo' in r else '--'} \\\\")
    tex("best.tex", "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\nLens & $z$ & $M_{\\rm tot}$ & $q$ & $M_{Lz}$ & $y$ & $\\rho$ & $\\sigma_y$ & $\\sigma_{\\log M_{Lz}}$ & $\\sigma_{\\log M_h}$ \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")


def table_steps():
    rec = json.load(open(os.path.join(a.fisher, "recommended_steps.json")))
    from lisa_mbhb import fisher as FI
    dflt = {**FI.STEP_MBH, **FI.STEP_LENS}; L = []
    for k in SAMPLED + ["log10_mlz", "y", "log10_mhalo"]:
        if k in rec: L.append(f"{PLAB[k]} & {ts(dflt[k], 2)} & {ts(rec[k], 3)} & {rec[k] / dflt[k]:.2f} \\\\")
    tex("steps.tex", "\\begin{tabular}{lrrr}\n\\toprule\nParameter & initial step & plateau step & ratio \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")


def table_mbh():
    L = []
    for k in SAMPLED:
        u = med([r[f"sigU_{k}"] for r in bylens["pm"]]); cells = []
        for l in LENSES: cells.append(f"{med([r[f'sig_{k}'] / r[f'sigU_{k}'] for r in bylens[l]]):.2f}")
        L.append(f"{PLAB[k]} & {ts(u, 2)} & {med([r[f'sigK_{k}'] / r[f'sigU_{k}'] for r in bylens['pm']]):.2f} & " + " & ".join(cells) + " \\\\")
    tex("mbh.tex", "\\begin{tabular}{lrrrrrrr}\n\\toprule\n & $\\sigma^{\\rm unl}$ & PM, lens known & PM & NFW$_{9}$ & NFW$_{11}$ & halo$_{9}$ & halo$_{11}$ \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    NUM["mbhRatioPMmed"] = f"{np.median([med([r[f'sig_{k}'] / r[f'sigU_{k}'] for r in bylens['pm']]) for k in SAMPLED]):.2f}"
    NUM["mbhKnownPMmed"] = f"{np.median([med([r[f'sigK_{k}'] / r[f'sigU_{k}'] for r in bylens['pm']]) for k in SAMPLED]):.2f}"
    NUM["dlSNRUnl"] = f"{med([r['snr_unlensed'] * LN10 * r['sigU_log10_dl'] for r in bylens['pm']]):.0f}"
    NUM["dlSNRKnown"] = f"{med([r['snr_lensed'] * LN10 * r['sigK_log10_dl'] for r in bylens['pm']]):.0f}"
    NUM["dlSNRJoint"] = f"{med([r['snr_lensed'] * LN10 * r['sig_log10_dl'] for r in bylens['pm']]):.0f}"
    NUM["dlSNRJointHaloEleven"] = f"{med([r['snr_lensed'] * LN10 * r['sig_log10_dl'] for r in bylens['halo:1e11']]):.0f}"
    NUM["dlSNRJointNFWEleven"] = f"{med([r['snr_lensed'] * LN10 * r['sig_log10_dl'] for r in bylens['nfw:1e11']]):.0f}"
    NUM["mcSNRUnl"] = f"{med([r['snr_unlensed'] * LN10 * r['sigU_log10_mchirp'] for r in bylens['pm']]):.2f}"


def table_check():
    d = json.load(open(os.path.join(a.fisher, "plots", "check", "fisher_check.json")))
    pts = list(dict.fromkeys(x["point"] for x in d)); L = []
    for p in pts:
        e = [x for x in d if x["point"] == p]
        sg = [x["ratio"] for x in e if x["test"].startswith("single")]; r_ = [x["ratio"] for x in e if x["test"].startswith("random")]; w = [x for x in e if x["test"].startswith("weakest")]
        lam = lambda t: float(t.split("lam=")[1].rstrip(")"))
        import re
        mm = re.match(r"(\S+) z=(\S+) M=(\S+) q=(\S+) ML=(\S+) y=(\S+)", p); lens_, zz, mt, qq, ml, yy = mm.groups()
        nm = {"pm": "PM", "nfw:1e11": r"PM+NFW, $M_{200}=10^{11}$", "nfw:1e9": r"PM+NFW, $M_{200}=10^{9}$", "halo:1e11": r"halo only, $10^{11}$", "halo:1e9": r"halo only, $10^{9}$"}[lens_]
        par = "$z=%s$, $M_{\\rm tot}=%s$, $q=%s$, $M_{Lz}=%s$, $y=%s$" % (zz, ts(float(mt), 1).strip("$"), qq, ts(float(ml), 1).strip("$"), yy)
        wk = "; ".join("%.2f (%s)" % (x["ratio"], ts(lam(x["test"]), 1)) for x in w)
        L.append("%s & %s & %d & %.4f--%.4f & %.4f--%.4f & %s \\\\" % (nm, par, len(sg), min(sg), max(sg), min(r_), max(r_), wk))
    tex("check.tex", "\\begin{tabular}{llrrrl}\n\\toprule\nLens & Point & single & single ratio & random ratio & weakest directions: ratio ($\\lambda$) \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    allr = [x["ratio"] for x in d if not x["test"].startswith("weakest")]
    NUM["checkN"] = len(d); NUM["checkNstrong"] = len(allr); NUM["checkMin"] = f"{min(allr):.4f}"; NUM["checkMax"] = f"{max(allr):.4f}"
    wk = [x["ratio"] for x in d if x["test"].startswith("weakest")]; NUM["weakMax"] = f"{max(wk):.0f}"; NUM["weakMed"] = f"{np.median(wk):.1f}"


# ---------------------------------------------------------------- figures
def fig_noise_source():
    from lisa_mbhb import demo, noise, waveform as W
    m1 = 1e6 / 1.5; p = demo.default_params(m1=m1, m2=0.5 * m1, z=1.0, inclination=0.9); cfg = W.WaveformConfig(tc_ref=p["tc"]); f = W.design_grid(p, cfg)
    hp, hc, t = W.generate(p, f, cfg); hcr = 2 * f * np.sqrt(np.abs(hp) ** 2 + np.abs(hc) ** 2)
    ff = np.logspace(-5, -0.5, 600); sn = noise.sensitivity_curve(ff)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].loglog(ff, np.sqrt(ff * sn), "k", label=r"$\sqrt{f S_n}$, SciRDv1 (A+E, sky averaged)"); ax[0].loglog(f, hcr, color="#2f6db5", label="source, $z=1$, $M_{\\rm tot}=10^6$, $q=0.5$")
    ax[0].set_xlabel("f [Hz]"); ax[0].set_ylabel("characteristic strain"); ax[0].set_ylim(1e-23, 1e-15); ax[0].set_xlim(1e-5, 0.3); ax[0].legend(fontsize=8)
    for c, ls in (("A", "-"), ("T", "--")):
        s = noise.psd_tdi_A(ff) if c == "A" else noise.psd_tdi_T(ff); ax[1].loglog(ff, np.sqrt(s), ls, color="k" if c == "A" else "0.5", label=f"TDI-1 {c} channel")
    ax[1].set_xlabel("f [Hz]"); ax[1].set_ylabel(r"$\sqrt{S_n(f)}$ [Hz$^{-1/2}$]"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f01_noise_source.png"), dpi=170); plt.close(fig)
    NUM["gridN"] = f"{len(f):,}".replace(",", "\\,"); NUM["fMinSrc"] = f"{f[0]:.1e}"; NUM["fMaxSrc"] = f"{f[-1]:.2f}"


def fig_amplification():
    from lisa_mbhb import lensing as L
    f = np.logspace(-4.3, -1.6, 500); fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for M, c in zip((1e6, 1e7, 1e8), ("#6baed6", "#2f6db5", "#08306b")): ax[0].semilogx(f, np.abs(L.get_lens("pm", mass_lz=M, y=1.0).amplification(f)), color=c, label=f"$M_{{Lz}}=10^{{{int(np.log10(M))}}}$")
    for y, c in zip((0.3, 1.0, 2.0), ("#fdae6b", "#e6550d", "#7f2704")): ax[1].semilogx(f, np.abs(L.get_lens("pm", mass_lz=1e7, y=y).amplification(f)), color=c, label=f"$y={y:g}$")
    ax[2].semilogx(f, np.abs(L.get_lens("pm", mass_lz=1e7, y=1.0).amplification(f)), color=COL["pm"], label="point mass")
    for l in ("nfw:1e11", "halo:1e11"):
        try: ax[2].semilogx(f, np.abs(L.get_lens(l, mass_lz=1e7, y=1.0, zs=1.0).amplification(f)), color=COL[l], label=NICE[l])
        except Exception as e: print("lens failed", l, e)
    for x, t in zip(ax, ("point mass, $y=1$", "point mass, $M_{Lz}=10^7$", "$M_{Lz}=10^7$ (unit), $y=1$")): x.set_title(t, fontsize=10); x.set_xlabel("f [Hz]"); x.legend(fontsize=8)
    ax[0].set_ylabel("$|F(f)|$"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "f02_amplification.png"), dpi=170); plt.close(fig)


def fig_validation():
    d = json.load(open(os.path.join(a.fisher, "plots", "check", "fisher_check.json"))); pts = list(dict.fromkeys(x["point"] for x in d))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    sym = {"single": "o", "random": "s", "weakest": "D"}; colr = {"single": "#2f6db5", "random": "#e8710a", "weakest": "#c0392b"}
    for i, p in enumerate(pts):
        for kind in sym:
            v = [x["ratio"] for x in d if x["point"] == p and x["test"].startswith(kind)]
            ax[0].plot(np.full(len(v), i) + {"single": -0.15, "random": 0, "weakest": 0.15}[kind], v, sym[kind], color=colr[kind], ms=5, alpha=0.8, label=kind if i == 0 else None)
    ax[0].set_yscale("log"); ax[0].axhline(1, color="k", lw=1); ax[0].set_xticks(range(len(pts))); ax[0].set_xticklabels([p.split(" z=")[0] + "\n" + "q=" + p.split("q=")[1].split(" ")[0] for p in pts], fontsize=8)
    ax[0].set_ylabel(r"exact $\Delta\chi^2$ / Fisher prediction ($\epsilon=0.3$)"); ax[0].legend(fontsize=8)
    w = [x for x in d if x["test"].startswith("weakest")]; lam = [float(x["test"].split("lam=")[1].rstrip(")")) for x in w]; ax[1].loglog(lam, [x["ratio"] for x in w], "D", color="#c0392b")
    ax[1].axhline(1, color="k", lw=1); ax[1].set_xlabel(r"normalised eigenvalue $\lambda$ of the displacement direction"); ax[1].set_ylabel(r"exact / predicted $\Delta\chi^2$")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f03_validation.png"), dpi=170); plt.close(fig)


def fig_correlation():
    from lisa_mbhb.fisher import covariance
    want = (1.0, 1e6, 0.3, 0.9, 1e7, 1.0); sel = []
    for l in ("pm", "nfw:1e11", "halo:1e11"):
        for r, m in zip(rows, mats):
            if r["lens"] == l and key(r) == want: sel.append((l, r, m)); break
    fig = plt.figure(figsize=(19, 4.8)); gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.04, 1.05], wspace=0.35); ax = [fig.add_subplot(gs[0, i]) for i in (0, 1, 2)]; cax = fig.add_subplot(gs[0, 3]); ax4 = fig.add_subplot(gs[0, 4])
    for a_, (l, r, m) in zip(ax, sel):
        F = np.array(m["F"]); C, cond, _ = covariance(F); s_ = np.sqrt(np.diag(C)); R = C / np.outer(s_, s_); nm = m["names"]
        im = a_.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1); a_.set_xticks(range(len(nm))); a_.set_xticklabels([PLAB[n] for n in nm], rotation=90, fontsize=8); a_.set_yticks(range(len(nm))); a_.set_yticklabels([PLAB[n] for n in nm], fontsize=8)
        a_.set_title(f"{NICE[l]}\n(cond = {cond:.1e})", fontsize=9); a_.grid(False)
    fig.colorbar(im, cax=cax, label="posterior correlation")
    for l, r, m in sel:
        F = np.array(m["F"]); d_ = 1 / np.sqrt(np.diag(F)); w = np.sort(np.linalg.eigvalsh(F * np.outer(d_, d_)))[::-1]; ax4.semilogy(range(1, len(w) + 1), w, "o-", color=COL[l], label=NICE[l], ms=4)
    ax4.set_xlabel("eigenvalue index"); ax4.set_ylabel("eigenvalue of the normalised Fisher matrix"); ax4.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "f04_correlation.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    l, r, m = sel[0]; C, _, _ = covariance(np.array(m["F"])); nm = m["names"]; s_ = np.sqrt(np.diag(C)); R = C / np.outer(s_, s_)
    i = nm.index("log10_dl"); NUM["corrDlInc"] = f"{R[i, nm.index('cos_inc')]:.3f}"; NUM["corrDlPsi"] = f"{R[i, nm.index('psi')]:.3f}"; NUM["corrDlPhi"] = f"{R[i, nm.index('coa_phase')]:.3f}"


def fig_snr_scaling():
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for l in LENSES:
        rs = bylens[l]; x = np.array([r["snr_diff"] for r in rs])
        ax[0].loglog(x, [r["sig_y"] for r in rs], "o", ms=3.5, alpha=0.6, color=COL[l], label=NICE[l])
        if l in ("pm", "nfw:1e9", "nfw:1e11"): ax[1].loglog(x, [r["sig_log10_mlz"] for r in rs], "o", ms=3.5, alpha=0.6, color=COL[l], label=NICE[l])
        if l != "pm": ax[2].loglog(x, [r["sig_log10_mhalo"] for r in rs], "o", ms=3.5, alpha=0.6, color=COL[l], label=NICE[l])
    for a_, t in zip(ax, (r"$\sigma_y$", r"$\sigma_{\log_{10}M_{Lz}}$", r"$\sigma_{\log_{10}M_{halo}}$")):
        xx = np.array(a_.get_xlim()); a_.loglog(xx, 3 / xx * (1 if a_ is not ax[1] else 1), "k--", lw=0.8, label=r"$\propto 1/\rho_{\rm lens}$" if a_ is ax[0] else None)
        a_.set_xlabel(r"SNR of the lensing imprint $\rho_{\rm lens}=\|h_L-h_U\|$"); a_.set_ylabel(t); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f05_snr_scaling.png"), dpi=170); plt.close(fig)


def fig_magnification():
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2)); rs = bylens["pm"]
    mu = lambda y: (y * y + 2) / (y * np.sqrt(y * y + 4))
    for M, c in zip((1e6, 1e7, 1e8), ("#9ecae1", "#3182bd", "#08306b")):
        v = [r for r in rs if r["mass_lz"] == M]; ax[0].plot([np.sqrt(mu(r["y"])) for r in v], [r["snr_lensed"] / r["snr_unlensed"] for r in v], "o", color=c, label=f"$M_{{Lz}}=10^{{{int(np.log10(M))}}}$")
    x = np.linspace(1, 2, 10); ax[0].plot(x, x, "k--", lw=0.8); ax[0].set_xlabel(r"$\sqrt{\mu_{\rm tot}(y)}$ (geometric optics)"); ax[0].set_ylabel(r"$\rho_{\rm lensed}/\rho_{\rm unlensed}$"); ax[0].legend(fontsize=8)
    dev = {M: [abs(r["snr_lensed"] / r["snr_unlensed"] / np.sqrt(mu(r["y"])) - 1) for r in rs if r["mass_lz"] == M] for M in (1e6, 1e7, 1e8)}
    for M, c in zip(dev, ("#9ecae1", "#3182bd", "#08306b")): ax[1].plot([r["y"] for r in rs if r["mass_lz"] == M], dev[M], "o", color=c, label=f"$10^{{{int(np.log10(M))}}}$")
    ax[1].set_yscale("log"); ax[1].set_xlabel("$y$"); ax[1].set_ylabel(r"$|\rho_{\rm lensed}/(\rho_{\rm unl}\sqrt{\mu})-1|$"); ax[1].legend(fontsize=8, title="$M_{Lz}$")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f06_magnification.png"), dpi=170); plt.close(fig)
    NUM["magDevEight"] = f"{100 * max(dev[1e8]):.1f}"; NUM["magDevSeven"] = f"{100 * max(dev[1e7]):.1f}"; NUM["magDevSix"] = f"{100 * max(dev[1e6]):.0f}"


def fig_mbh_impact():
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for j, l in enumerate(LENSES):
        m_, lo, hi = [], [], []
        for k in SAMPLED:
            v = np.array([r[f"sig_{k}"] / r[f"sigU_{k}"] for r in bylens[l]]); m_.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
        yy = np.arange(len(SAMPLED)) + (j - 2) * 0.13; ax.errorbar(m_, yy, xerr=[np.array(m_) - lo, np.array(hi) - m_], fmt="o", ms=4, color=COL[l], label=NICE[l], capsize=0, lw=1.2)
    ax.axvline(1, color="k", lw=1); ax.set_xscale("log"); ax.set_yticks(range(len(SAMPLED))); ax.set_yticklabels([PLAB[k] for k in SAMPLED]); ax.invert_yaxis()
    ax.set_xlabel(r"$\sigma$ (joint MBH + lens model) / $\sigma$ (unlensed model)"); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.set_xticks([0.3, 0.5, 1, 2, 5]); ax.set_xlim(0.28, 12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "f07_mbh_impact.png"), dpi=170); plt.close(fig)


def fig_stepscan():
    rec = json.load(open(os.path.join(a.fisher, "recommended_steps.json"))); sc = {}
    for f in glob.glob(os.path.join(a.fisher, "scan", "stepscan_*_f0.json")): d = json.load(open(f)); sc[d["param"]] = d
    names = [k for k in SAMPLED + ["log10_mlz", "y"] if k in sc]; fig, axs = plt.subplots(3, 5, figsize=(16, 7.5)); axs = axs.ravel()
    for ax, k in zip(axs, names):
        d = sc[k]; m = np.array(d["mults"]); v = np.array(d["fkk"]); step = m * d["base_step"]; ax.semilogx(step, v / np.nanmedian(v[v > 0]), "o-", ms=3.5, color="#2f6db5")
        if k in rec: ax.axvline(rec[k], color="#c0392b", ls="--", lw=1)
        ax.set_ylim(0, 1.6); ax.set_title(PLAB[k], fontsize=10); ax.set_xlabel("step")
    for ax in axs[len(names):]: ax.axis("off")
    axs[0].set_ylabel(r"$F_{kk}/{\rm median}$"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "f08_stepscan.png"), dpi=150); plt.close(fig)


def curves(path):
    d = {}
    for r in csv.DictReader(open(path)): d.setdefault(r["config"], []).append((float(r["logM"]), float(r["median"])))
    return {c: np.array(sorted(v)) for c, v in d.items()}


def fig_fig3_before_after():
    v1 = curves(os.path.join(a.fisher, "fig3_v1", "fig3_curves.csv")); v2 = curves(os.path.join(a.fisher, "fig3", "fig3_curves.csv")); pap = {}
    for r in csv.DictReader(open(os.path.join(a.fisher, "fig3", "compare", "paper_fig3_digitised.csv"))): pap.setdefault(r["config"], []).append((float(r["logM"]), float(r["log10_median"])))
    pap = {c: np.array(sorted(v)) for c, v in pap.items()}; fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for a_, c, t in zip(ax, ("mr", "isco"), ("IMRPhenomD with merger and ringdown", "IMRPhenomD truncated at the ISCO")):
        p = pap[c]
        for name, v, st, lab in (("v1", v1, "o--", "first lens grid"), ("v2", v2, "o-", "refined lens grid")):
            o = v[c]; pm_ = np.interp(o[:, 0], p[:, 0], p[:, 1], left=np.nan, right=np.nan); ok = np.isfinite(pm_) & (10 ** pm_ < 0.95)
            a_.semilogx(10 ** o[ok, 0], o[ok, 1] / 10 ** pm_[ok], st, ms=3.5, label=lab)
        a_.axhline(1, color="k", lw=1); a_.axhspan(0.9, 1.1, color="0.88"); a_.set_ylim(0.4, 1.4); a_.set_xlabel(r"$M_{Lz}\ [M_\odot]$"); a_.set_title(t, fontsize=10); a_.legend(fontsize=8)
    ax[0].set_ylabel("ours / paper (median $\\Delta M_{Lz}/M_{Lz}$)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "f09_fig3_before_after.png"), dpi=170); plt.close(fig)


def fig_mcrit():
    PAPER = {"isco": (1.08e5, 0.40e5, 0.64e5), "mr": (4.73e3, 1.75e3, 2.45e3), "hm": (3.19e3, 1.17e3, 1.39e3), "hmp": (1.49e3, 0.49e3, 0.62e3), "hmm": (6.63e3, 2.24e3, 3.46e3)}
    NAMES = {"isco": "ISCO", "mr": "MR", "hm": "MR+HM", "hmp": r"HM, $\chi=+0.99$", "hmm": r"HM, $\chi=-0.99$"}; fig, axs = plt.subplots(1, 5, figsize=(17, 3.4)); L = []
    for ax, c in zip(axs, PAPER):
        crit = []
        for f in sorted(glob.glob(os.path.join(a.fisher, "fig3", "shards", f"fig3_{c}_d*.json"))):
            d = json.load(open(f)); lm = np.array(d["logM"]); row = np.array(d["rel_dM"], float); ok = np.where(row <= 1.0)[0]
            if len(ok) == 0 or ok[0] == 0: continue
            j = ok[0]; crit.append(10 ** np.interp(0.0, [np.log10(row[j]), np.log10(row[j - 1])], [lm[j], lm[j - 1]]))
        crit = np.array(crit); ax.hist(np.log10(crit), bins=18, color="#2f6db5", alpha=0.75); p = PAPER[c]; ax.axvspan(np.log10(p[0] - p[1]), np.log10(p[0] + p[2]), color="#e8710a", alpha=0.25); ax.axvline(np.log10(p[0]), color="#e8710a", lw=2)
        ax.axvline(np.log10(np.median(crit)), color="k", ls="--"); ax.set_title(NAMES[c], fontsize=10); ax.set_xlabel(r"$\log_{10}M^{\rm crit}_{Lz}$")
        m50, m16, m84 = np.percentile(crit, [50, 16, 84]); L.append(f"{NAMES[c]} & {len(crit)} & {ts(m50, 3)} & {ts(m16, 3)}--{ts(m84, 3)} & {ts(p[0], 3)} & {ts(p[0] - p[1], 3)}--{ts(p[0] + p[2], 3)} & {m50 / p[0]:.2f} \\\\")
        NUM[f"mcrit{c}"] = f"{m50:.3g}"; NUM[f"mcritpaper{c}"] = f"{p[0]:.3g}"; NUM[f"mcritratio{c}"] = f"{m50 / p[0]:.2f}"
    axs[0].set_ylabel("draws"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "f10_mcrit_hist.png"), dpi=170); plt.close(fig)
    tex("mcrit.tex", "\\begin{tabular}{lrrrrrr}\n\\toprule\nWaveform & $N$ & $M^{\\rm crit}_{Lz}$ (this work) & 68\\% range (this work) & $M^{\\rm crit}_{Lz}$ (paper) & 68\\% range (paper) & ratio \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")


def fig3_numbers():
    c2 = curves(os.path.join(a.fisher, "fig3", "fig3_curves.csv"))
    for c in ("mr", "isco", "hm", "hmp", "hmm"):
        v = c2[c]; NUM[f"plateau{c}"] = f"{v[-1, 1]:.2e}".replace("e-0", "\\times10^{-") + "}"
    pap = {}
    for r in csv.DictReader(open(os.path.join(a.fisher, "fig3", "compare", "paper_fig3_digitised.csv"))): pap.setdefault(r["config"], []).append((float(r["logM"]), float(r["log10_median"])))
    pap = {c: np.array(sorted(v)) for c, v in pap.items()}
    for c in pap: NUM[f"paperplateau{c}"] = f"{10 ** np.median(pap[c][-40:, 1]):.2e}".replace("e-0", "\\times10^{-") + "}"
    for tag, path in (("one", os.path.join(a.fisher, "fig3_v1", "fig3_curves.csv")), ("two", os.path.join(a.fisher, "fig3", "fig3_curves.csv"))):
        cv = curves(path)
        for c in ("mr", "hm", "isco", "hmp", "hmm"):
            o = cv[c]; p = pap[c]; pm_ = np.interp(o[:, 0], p[:, 0], p[:, 1], left=np.nan, right=np.nan); ok = np.isfinite(pm_) & (10 ** pm_ < 0.95) & (o[:, 0] < 5)
            if ok.any(): NUM[f"lowM{tag}{c}"] = f"{np.median(o[ok, 1] / 10 ** pm_[ok]):.2f}"


def fig_fig3_clean():
    PAPER = {"isco": (1.08e5, 0.40e5, 0.64e5), "mr": (4.73e3, 1.75e3, 2.45e3), "hm": (3.19e3, 1.17e3, 1.39e3), "hmp": (1.49e3, 0.49e3, 0.62e3), "hmm": (6.63e3, 2.24e3, 3.46e3)}
    LAB = {"isco": r"ISCO ($\chi_1,\chi_2=0$)", "mr": r"MR ($\chi_1,\chi_2=0$)", "hm": r"MR + HM ($\chi_1,\chi_2=0$)", "hmp": r"MR + HM ($\chi_1,\chi_2=+0.99$)", "hmm": r"MR + HM ($\chi_1,\chi_2=-0.99$)"}
    CL = {"isco": "#f28e2b", "mr": "#e15759", "hm": "#59a14f", "hmp": "#8c564b", "hmm": "#4e79a7"}
    d = {}
    for r in csv.DictReader(open(os.path.join(a.fisher, "fig3", "fig3_curves.csv"))): d.setdefault(r["config"], []).append((float(r["logM"]), float(r["median"]), float(r["p16"]), float(r["p84"])))
    d = {c: np.array(sorted(v)) for c, v in d.items()}
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, group in zip(axs, (("isco", "mr", "hm"), ("hmm", "hm", "hmp"))):
        for c in group:
            v = d[c]; M = 10 ** v[:, 0]; ax.fill_between(M, v[:, 2], v[:, 3], color=CL[c], alpha=0.25, lw=0); ax.plot(M, v[:, 1], color=CL[c], lw=2, label=LAB[c])
            p = PAPER[c]; ax.errorbar([p[0]], [1.0], xerr=[[p[1]], [p[2]]], fmt="D", color=CL[c], mec="k", ms=6, capsize=3, zorder=6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e3, 1e9); ax.set_ylim(1e-4, 1.3); ax.grid(alpha=0.3, which="both"); ax.set_xlabel(r"$M_{Lz}\ [M_\odot]$"); ax.legend(loc="lower left", fontsize=9)
    axs[0].set_ylabel(r"$\Delta M_{Lz}/M_{Lz}$"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_reproduction_clean.png"), dpi=170); plt.close(fig)
    pap = {}
    for r in csv.DictReader(open(os.path.join(a.fisher, "fig3", "compare", "paper_fig3_digitised.csv"))): pap.setdefault(r["config"], []).append((float(r["logM"]), float(r["log10_median"])))
    pap = {c: np.array(sorted(v)) for c, v in pap.items()}
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.2), gridspec_kw=dict(height_ratios=[1.6, 1]), sharex=True)
    for col, group in ((0, ("isco", "mr", "hm")), (1, ("hmm", "hm", "hmp"))):
        for c in group:
            o = d[c]; p = pap[c]; ax[0, col].loglog(10 ** o[:, 0], o[:, 1], "-", color=CL[c], lw=2.2, label=LAB[c]); ax[0, col].loglog(10 ** p[:, 0], 10 ** p[:, 1], "--", color="k", lw=1.0, alpha=0.8)
            pm_ = np.interp(o[:, 0], p[:, 0], p[:, 1], left=np.nan, right=np.nan); ok = np.isfinite(pm_) & (10 ** pm_ < 0.95); ax[1, col].semilogx(10 ** o[ok, 0], o[ok, 1] / 10 ** pm_[ok], "o-", ms=3.5, color=CL[c])
        ax[0, col].plot([], [], "--", color="k", label="Ref. [5] (digitised)"); ax[0, col].legend(fontsize=8, loc="lower left"); ax[0, col].set_ylim(1e-4, 1); ax[0, col].grid(alpha=0.3, which="both")
        ax[1, col].axhline(1, color="k", lw=1); ax[1, col].axhspan(0.9, 1.1, color="0.88"); ax[1, col].set_ylim(0.4, 1.4); ax[1, col].set_xlabel(r"$M_{Lz}\ [M_\odot]$"); ax[1, col].grid(alpha=0.3, which="both")
    ax[0, 0].set_ylabel(r"$\Delta M_{Lz}/M_{Lz}$ (median)"); ax[1, 0].set_ylabel("ours / paper"); ax[1, 0].set_xlim(1e3, 1e9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_difference_clean.png"), dpi=170); plt.close(fig)


def fig_heatmaps():
    want = dict(z=1.0, m_tot=1e6, q=0.3); fig, axs = plt.subplots(3, 5, figsize=(17, 8.6)); Ms = [1e6, 1e7, 1e8]; ys = [0.3, 1.0, 2.0]
    spec = [("sig_y", r"$\sigma_y/y$ [%]", True), ("sig_log10_mlz", r"$\sigma_{\log_{10}M_{Lz}}$", False), ("sig_log10_mhalo", r"$\sigma_{\log_{10}M_{halo}}$", False)]
    for j, l in enumerate(LENSES):
        for i, (k, t, rel) in enumerate(spec):
            ax = axs[i, j]; Z = np.full((3, 3), np.nan)
            for r in bylens[l]:
                if all(r[a_] == b_ for a_, b_ in want.items()) and k in r: Z[ys.index(r["y"]), Ms.index(r["mass_lz"])] = r[k] / r["y"] * 100 if rel else r[k]
            if np.isfinite(Z).any():
                im = ax.imshow(Z, origin="lower", cmap="viridis_r", norm=matplotlib.colors.LogNorm(vmin=np.nanmin(Z), vmax=np.nanmax(Z) if np.nanmax(Z) > np.nanmin(Z) else np.nanmin(Z) * 2))
                for (a_, b_), v in np.ndenumerate(Z): ax.text(b_, a_, f"{v:.2g}", ha="center", va="center", color="w", fontsize=9)
            else: ax.text(0.5, 0.5, "not a parameter", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.4")
            ax.set_xticks(range(3)); ax.set_xticklabels(["$10^6$", "$10^7$", "$10^8$"]); ax.set_yticks(range(3)); ax.set_yticklabels(["0.3", "1", "2"]); ax.grid(False)
            if i == 0: ax.set_title(NICE[l], fontsize=10)
            if j == 0: ax.set_ylabel(t + "\n$y$")
            if i == 2: ax.set_xlabel(r"$M_{Lz}$ (reference mass)")
    fig.suptitle(r"$z=1$, $M_{\rm tot}=10^6\,M_\odot$, $q=0.3$", fontsize=11, y=0.995); fig.tight_layout(); fig.savefig(os.path.join(FIG, "f11_heatmaps.png"), dpi=150); plt.close(fig)


for fn in (table_summary, extra_numbers, table_best, table_steps, table_mbh, table_check, fig_noise_source, fig_amplification, fig_validation, fig_correlation, fig_snr_scaling,
           fig_magnification, fig_mbh_impact, fig_stepscan, fig_fig3_before_after, fig_fig3_clean, fig_mcrit, fig3_numbers, fig_heatmaps): safe(fn)
open(os.path.join(TAB, "numbers.tex"), "w").write("% generated by scripts/fisher_report.py\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in NUM.items() if k.isalpha()) + "\n")
print("done; macros:", len(NUM)); print({k: v for k, v in NUM.items() if not k.isalpha()})
