"""Analysis figures of the lens-model Fisher study (Fisher only; 90% CI = 1.645 sigma).  CLUSTER JOB (light, one --kind per job).

    python scripts/fisher_study_analysis.py --kind KIND --rows DIR --fisher DIR --out DIR

kinds (row-file prefixes in --rows, written by cluster/make_analysis_dag.py):
  violins    pop_*    90% CI of every parameter over 100 random extrinsic draws: unlensed / PML / NFW / PML+NFW, and split pairs
  ladder     pop_*    information ladder: binary figure of merit gain and lens-block log det per model, over the draws
  pca        pop_*    eigen-decomposition of the binary-marginalised lens Fisher (draw 0): spectrum + eigenvector composition
  corr       pop_*    correlation matrices (draw 0) per model
  ellipses   pop_*    filled 90% regions of shared quantities, all models (draw 0), zoom inset on the tightest
  maps       plane_*  (y, M200) maps for PML+NFW: halo and BH detection significance, 90% CI of log10(M_BH/M200), sigma(log c200)/0.1 dex
  mbhmaps    mbh_*    the same on the (M_BH, M200) plane
  shear      shear_*  shear detection significance vs |gamma|, and how much knowing the shear tightens the halo / BH constraints
  cumul      cumul_*  where in frequency the information comes from (sigma vs upper frequency)
Everything is shown for kappa free (mass sheet: unconstrained quantities omitted) and kappa known.
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from lisa_mbhb import fisher_analysis as FA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--fisher", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--scale", type=float, default=1.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); FA.style(a.scale)
MODELS = ["pml", "nfw", "pmlnfw"]; ALL = ["unlensed"] + MODELS
KNOWN = {"kappa free": (), "kappa known": ("kappa",)}


def save(fig, name, note=""):
    fn = os.path.join(a.out, name); fig.savefig(fn, bbox_inches="tight"); plt.close(fig); print("wrote", fn, note)


def pop_results(known=()):
    """{model: [analyse result per draw]} for the population, plus unlensed per draw."""
    rows = FA.load(a.rows, "pop_"); res = {m: {} for m in ALL}
    for r, d in rows:
        res[d["model"]][d["info"]["draw"]] = FA.analyse(r, d["model"], d["info"], d["m_ref"], known)
    draws = sorted({d["info"]["draw"] for _, d in rows})
    for k in draws:
        u = FA.load_unlensed(a.fisher, f"z1_m1e6_q1_d{k}")
        if u: res["unlensed"][k] = FA.analyse(u[0], "unlensed")
    return res, draws


def param_list(res):
    seen = []
    for m in ALL:
        for rr in res[m].values():
            seen += [n for n in rr if not n.startswith("_") and n not in seen]
    keep = [n for n in seen if any(np.isfinite(FA.hw90(rr, n)) for m in ALL for rr in res[m].values())]
    order = list(FA.LABEL)
    return sorted(keep, key=lambda n: order.index(n) if n in order else 999)


def half_violin(ax, data, pos, color, side, width=0.8):
    data = np.asarray([x for x in data if np.isfinite(x)])
    if len(data) < 2: return
    v = ax.violinplot([data], positions=[pos], widths=width, showextrema=False, showmedians=False)
    for b in v["bodies"]:
        P = b.get_paths()[0].vertices
        P[:, 0] = np.clip(P[:, 0], -np.inf, pos) if side == "left" else (np.clip(P[:, 0], pos, np.inf) if side == "right" else P[:, 0])
        b.set_facecolor(color); b.set_edgecolor("k"); b.set_alpha(0.75)
    q = np.percentile(data, [5, 50, 95]); dx = width / 2 * (-1 if side == "left" else 1 if side == "right" else 0)
    for qq, ls in zip(q, (":", "-", ":")):
        ax.plot([pos, pos + (dx if side != "both" else width / 2)], [qq, qq], color="k", ls=ls, lw=1.2)


def grid(n, ncol=5, w=5.2, h=4.6):
    nr = -(-n // ncol); fig, axs = plt.subplots(nr, ncol, figsize=(w * ncol, h * nr), squeeze=False)
    for ax in axs.ravel()[n:]: ax.axis("off")
    return fig, axs.ravel()


# ----------------------------------------------------------------------------------------------------------------------
if a.kind == "violins":
    for kname, known in KNOWN.items():
        res, draws = pop_results(known); P = param_list(res); tag = kname.replace(" ", "_")
        fig, axs = grid(len(P))
        for ax, p in zip(axs, P):
            for i, m in enumerate(ALL):
                half_violin(ax, [np.log10(FA.hw90(rr, p)) for rr in res[m].values()], i, FA.MODEL_COL[m], "both")
            ax.set_xticks(range(len(ALL))); ax.set_xticklabels([FA.MODEL_LAB[m] for m in ALL], rotation=30); ax.minorticks_off()
            ax.set_title(FA.LABEL.get(p, p)); ax.set_ylabel(r"$\log_{10}$ 90% CI")
        fig.suptitle(f"90% CI half-width over {len(draws)} random sky/inclination draws  ({kname}; z=1, M=1e6, q=1; M_Lz=1e6, y=0.1, "
                     r"$\kappa$=0.1, M200=1e11).  Lines: 5/50/95%.  Missing violin: unconstrained or not in the model", y=1.0)
        fig.tight_layout(); save(fig, f"violins_all_models_{tag}.png")
        pairs = [("unlensed", "pml"), ("pml", "pmlnfw"), ("nfw", "pmlnfw")]
        fig, axs = grid(len(P))
        for ax, p in zip(axs, P):
            for i, (l, rgt) in enumerate(pairs):
                half_violin(ax, [np.log10(FA.hw90(rr, p)) for rr in res[l].values()], i, FA.MODEL_COL[l], "left")
                half_violin(ax, [np.log10(FA.hw90(rr, p)) for rr in res[rgt].values()], i, FA.MODEL_COL[rgt], "right")
            ax.set_xticks(range(len(pairs))); ax.set_xticklabels([f"{FA.MODEL_LAB[l]} | {FA.MODEL_LAB[r_]}" for l, r_ in pairs], rotation=30)
            ax.minorticks_off(); ax.set_title(FA.LABEL.get(p, p)); ax.set_ylabel(r"$\log_{10}$ 90% CI")
        h = [plt.Rectangle((0, 0), 1, 1, color=FA.MODEL_COL[m]) for m in ALL]
        fig.suptitle(f"Split violins: left | right half = two lens models, same {len(draws)} draws ({kname})", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.legend(h, [FA.MODEL_LAB[m] for m in ALL], loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4)
        save(fig, f"violins_split_pairs_{tag}.png")

elif a.kind == "ladder":
    fig, axs = plt.subplots(1, 2, figsize=(28, 10))
    for col, (kname, known) in enumerate(KNOWN.items()):
        res, draws = pop_results(known); ax = axs[col]
        bnames = [n for n in res["unlensed"][draws[0]]["_names"]]
        gains = {m: [] for m in MODELS}; used = {}
        for k in draws:   # ONE parameter set per draw: binary parameters constrained in unlensed AND in every lens model
            if not all(k in res[m] for m in ["unlensed"] + MODELS): continue
            ok = [n for n in bnames if all(n in res[m][k]["_names"] and np.isfinite(res[m][k][n][1]) for m in ["unlensed"] + MODELS)]
            used[tuple(ok)] = used.get(tuple(ok), 0) + 1
            ru = res["unlensed"][k]; su = np.linalg.slogdet(ru["_C"][np.ix_(*[[ru["_names"].index(n) for n in ok]] * 2)])
            for m in MODELS:
                rm = res[m][k]; sm = np.linalg.slogdet(rm["_C"][np.ix_(*[[rm["_names"].index(n) for n in ok]] * 2)])
                gains[m].append(0.5 * (su[1] - sm[1]) / np.log(10) if sm[0] > 0 and su[0] > 0 else np.nan)
        for i, m in enumerate(MODELS):
            half_violin(ax, gains[m], i, FA.MODEL_COL[m], "both")
            g = np.array(gains[m]); g = g[np.isfinite(g)]
            if len(g): ax.text(i + 0.42, np.median(g), f"{np.median(g):+.2f}", va="center", fontsize=18)
        ax.axhline(0, color="k", lw=1); ax.set_xticks(range(3)); ax.set_xticklabels([FA.MODEL_LAB[m] for m in MODELS]); ax.minorticks_off()
        ax.set_xlim(-0.6, 2.9)
        common = max(used, key=used.get) if used else ()
        ax.set_ylabel(r"$\log_{10}$ (FoM$_{\rm lensed}$ / FoM$_{\rm unlensed}$)")
        ax.set_title(f"{kname}\nmost common set ({used.get(common, 0)}/{sum(used.values())} draws): "
                     + ", ".join(FA.LABEL.get(n, n) for n in common), fontsize=16)
    fig.suptitle("Information ladder: change of the binary-parameter figure of merit (det C)$^{-1/2}$ when the lens is fitted, over the draws.\n"
                 "Same parameters for every model in a draw (those constrained in unlensed and in all three lens models).  "
                 "> 0: lensing adds information about the source", y=1.03)
    fig.tight_layout(); save(fig, "ladder_binary_fom.png")

elif a.kind in ("pca", "corr", "ellipses"):
    rows = {d["model"]: (r, d) for r, d in FA.load(a.rows, "pop_") if d["info"]["draw"] == 0}
    if a.kind == "pca":
        # kappa known (kappa free has the exact mass-sheet null).  Lens-block covariance = the binary-marginalised lens covariance,
        # from the same eigen-split covariance as every other plot; a PC is 'unresolved' if the unresolved Fisher direction(s)
        # add > SENS_TOL to its variance (same rule as analyse) -> drawn hatched, its sigma is only a lower bound.
        fig, axs = plt.subplots(2, 3, figsize=(36, 20), gridspec_kw=dict(hspace=0.45, wspace=0.3))
        for j, m in enumerate(MODELS):
            r, d = rows[m]; names = r["names"]; keep = [i for i, n in enumerate(names) if n != "kappa"]; kn = [names[i] for i in keep]
            F = np.array(r["F"])[np.ix_(keep, keep)]; Fh = np.array(r["F_half"])[np.ix_(keep, keep)] if r.get("F_half") is not None else None
            dd, evc, U, un = FA.eigen_split(F, Fh); C = (U / evc) @ U.T / np.outer(dd, dd)
            ln = [n for n in kn if n in r["vals"]]; L = [kn.index(n) for n in ln]
            w, V = np.linalg.eigh(C[np.ix_(L, L)])            # ascending variance: PC1 = best measured
            sig = np.sqrt(np.clip(w, 1e-300, None)); bad = np.zeros(len(w), bool)
            if un.any():
                lref = np.maximum(evc[un], FA.EIG_FLOOR * evc[-1])
                for q in range(len(w)):
                    g = np.zeros(len(kn)); g[L] = V[:, q]; A = (g / dd) @ U
                    bad[q] = (A[un] ** 2 / lref).sum() > FA.SENS_TOL * (A[~un] ** 2 / evc[~un]).sum()
            ls = np.log10(sig); top = (ls[~bad].max() if (~bad).any() else 0) + 1.5
            ax = axs[0, j]
            for q in range(len(w)):
                if bad[q]:
                    ax.bar(q, top, color="none", edgecolor="0.4", hatch="//", lw=1.5)
                    ax.text(q, top, "unresolved\n(σ ≥ bound)", ha="center", va="bottom", fontsize=15)
                else:
                    ax.bar(q, ls[q], color=FA.MODEL_COL[m])
            ax.set_ylim(min(ls[~bad].min() if (~bad).any() else 0, 0) - 0.5, top + 1.2)
            ax.set_xticks(range(len(w))); ax.set_xticklabels([f"PC{i + 1}" for i in range(len(w))]); ax.minorticks_off()
            ax.set_ylabel(r"$\log_{10}\,\sigma$ along PC"); ax.set_title(f"{FA.MODEL_LAB[m]}\nlens eigen-spectrum (binary marginalised)")
            ax = axs[1, j]; im = ax.imshow(V.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(ln))); ax.set_xticklabels([FA.LABEL.get(n, n) for n in ln], rotation=45); ax.set_yticks(range(len(w)))
            ax.set_yticklabels([f"PC{i + 1}" + (" (unres.)" if bad[i] else "") for i in range(len(w))]); ax.minorticks_off()
            for (i, k), val in np.ndenumerate(V.T):
                if abs(val) > 0.25: ax.text(k, i, f"{val:.2f}", ha="center", va="center", fontsize=15)
            ax.set_title(f"{FA.MODEL_LAB[m]}: eigenvector composition\n(PC1 = best measured)")
        fig.colorbar(im, ax=axs[1, :].tolist(), shrink=0.8)
        fig.suptitle("Which lens-parameter combinations are measured (draw 0, M_Lz=1e6, y=0.1, M200=1e11; kappa known)", y=0.97)
        save(fig, "pca_lens_fisher.png")
    elif a.kind == "corr":
        fig, axs = plt.subplots(1, 3, figsize=(42, 14))
        cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("0.75")
        for ax, m in zip(axs, MODELS):
            r, d = rows[m]; res = FA.analyse(r, m, d["info"], d["m_ref"], ("kappa",)); C, nm = res["_C"], res["_names"]
            s = np.sqrt(np.clip(np.diag(C), 1e-300, None)); R = C / np.outer(s, s)
            unc = np.array([not np.isfinite(res[n][1]) for n in nm]); R[unc, :] = np.nan; R[:, unc] = np.nan   # meaningless rows masked
            im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1); ax.set_xticks(range(len(nm))); ax.set_yticks(range(len(nm))); ax.minorticks_off()
            lab = [FA.LABEL.get(n, n) + (" (unc.)" if u_ else "") for n, u_ in zip(nm, unc)]
            ax.set_xticklabels(lab, rotation=90); ax.set_yticklabels(lab)
            ax.set_title(f"{FA.MODEL_LAB[m]}  (kappa known)")
        fig.colorbar(im, ax=axs.tolist(), shrink=0.7, label="correlation")
        fig.suptitle("Correlation matrices, draw 0.  Grey: unconstrained parameter (row/column masked: its correlations are set by the unresolved direction)", y=1.0)
        save(fig, "correlation_matrices.png")
    else:
        pairs = [("log10_mchirp", "sin_ecl_lat"), ("cos_inc", "psi"), ("ecl_lon", "sin_ecl_lat"), ("log10_mchirp", "log10_mlz")]
        u = FA.load_unlensed(a.fisher, "z1_m1e6_q1_d0"); sets = ([("unlensed", FA.analyse(u[0], "unlensed"))] if u else [])
        sets += [(m, FA.analyse(rows[m][0], m, rows[m][1]["info"], rows[m][1]["m_ref"], ("kappa",))) for m in MODELS if m in rows]
        pairs = [(px, py) for px, py in pairs if sum(px in res["_names"] and py in res["_names"] and np.isfinite(FA.hw90(res, px))
                                                     and np.isfinite(FA.hw90(res, py)) for _, res in sets) >= 2]   # drop empty panels
        fig, axs = plt.subplots(1, len(pairs), figsize=(10 * len(pairs), 9), squeeze=False); axs = axs[0]
        for ax, (px, py) in zip(axs, pairs):
            have = [(m, res) for m, res in sets if px in res["_names"] and py in res["_names"] and np.isfinite(FA.hw90(res, px)) and np.isfinite(FA.hw90(res, py))]
            if not have: ax.axis("off"); continue
            spans = []
            for m, res in have:
                nm = res["_names"]; i, j = nm.index(px), nm.index(py); C2 = res["_C"][np.ix_([i, j], [i, j])]; w, U = np.linalg.eigh(C2)
                r90 = np.sqrt(4.6052); ang = np.degrees(np.arctan2(U[1, 1], U[0, 1]))
                ax.add_patch(Ellipse((res[px][0], res[py][0]), 2 * r90 * np.sqrt(w[1]), 2 * r90 * np.sqrt(w[0]), angle=ang, facecolor=FA.MODEL_COL[m],
                                     alpha=0.35, edgecolor=FA.MODEL_COL[m], lw=2.5, label=FA.MODEL_LAB[m]))
                spans.append((r90 * np.sqrt(C2[0, 0]), r90 * np.sqrt(C2[1, 1])))
            m0 = have[0][1]; sx, sy = max(s[0] for s in spans), max(s[1] for s in spans)
            ax.set_xlim(m0[px][0] - 1.3 * sx, m0[px][0] + 1.3 * sx); ax.set_ylim(m0[py][0] - 1.3 * sy, m0[py][0] + 1.3 * sy)
            ax.axvline(m0[px][0], color="crimson", lw=1); ax.axhline(m0[py][0], color="crimson", lw=1)
            ax.set_xlabel(FA.LABEL.get(px, px)); ax.set_ylabel(FA.LABEL.get(py, py)); ax.legend(loc="upper right")
            tx, ty = min(s[0] for s in spans), min(s[1] for s in spans)
            if sx / tx > 10:                       # zoom inset on the tightest region
                ins = ax.inset_axes([0.06, 0.06, 0.36, 0.36])
                for m, res in have:
                    nm = res["_names"]; i, j = nm.index(px), nm.index(py); C2 = res["_C"][np.ix_([i, j], [i, j])]; w, U = np.linalg.eigh(C2)
                    ins.add_patch(Ellipse((0, 0), 2 * np.sqrt(4.6052 * w[1]), 2 * np.sqrt(4.6052 * w[0]), angle=np.degrees(np.arctan2(U[1, 1], U[0, 1])),
                                          facecolor=FA.MODEL_COL[m], alpha=0.35, edgecolor=FA.MODEL_COL[m], lw=2))
                ins.set_xlim(-1.5 * tx, 1.5 * tx); ins.set_ylim(-1.5 * ty, 1.5 * ty); ins.set_title("zoom", fontsize=14); ins.tick_params(labelsize=11)
        fig.suptitle("90% regions, draw 0 (kappa known); red cross hair: injected"); fig.tight_layout(); save(fig, "ellipses_models.png")

elif a.kind in ("maps", "mbhmaps"):
    pre, xkey, xlab = ("plane_", "y", "$y$") if a.kind == "maps" else ("mbh_", "m_ref", r"$M_{Lz}\ [M_\odot]$")
    rows = FA.load(a.rows, pre)
    # (title, value, contour levels, uninformative(value) -> hatched, what 'uninformative' means)
    Q = [("halo significance  $\\psi_0/\\sigma(\\psi_0)$", lambda res: FA.significance_log(res, "log10_psi0"), [3, 5], lambda z: z < 1, "< 1 sigma"),
         ("BH significance  $M_{Lz}/\\sigma(M_{Lz})$", lambda res: FA.significance_log(res, "log10_mlz"), [3, 5], lambda z: z < 1, "< 1 sigma"),
         ("90% CI of $\\log_{10}(M_{\\rm BH}/M_{200})$", lambda res: FA.hw90(res, "log10_mbh_m200"), [0.1, 0.3], lambda z: z > 1, "> 1 dex"),
         ("$\\sigma(\\log_{10}c_{200})$ / 0.1 dex scatter", lambda res: FA.hw90(res, "log10_c200") / FA.Z90 / 0.1, [1], lambda z: z > 10, "sigma > 1 dex")]
    fig, axs = plt.subplots(2, 4, figsize=(40, 18))
    for i, (kname, known) in enumerate(KNOWN.items()):
        pts = [((r["y"] if xkey == "y" else d["m_ref"]), r["m200"], FA.analyse(r, d["model"], d["info"], d["m_ref"], known)) for r, d in rows]
        xs, ms = sorted({p[0] for p in pts}), sorted({p[1] for p in pts})
        def edges(v):   # cell edges on a log grid
            lv = np.log10(v); e = np.concatenate([[lv[0] - (lv[1] - lv[0]) / 2], (lv[1:] + lv[:-1]) / 2, [lv[-1] + (lv[-1] - lv[-2]) / 2]])
            return 10 ** e
        xe, me = edges(np.array(xs)), edges(np.array(ms))
        for j, (lab, fn, lev, uninf, why) in enumerate(Q):
            Z = np.full((len(ms), len(xs)), np.nan)
            for x, m2, res in pts: Z[ms.index(m2), xs.index(x)] = fn(res)
            ax = axs[i, j]; L = np.log10(Z); title = f"{lab}\n{kname}"
            if np.isfinite(L).any():
                pc = ax.pcolormesh(xs, ms, L, shading="nearest", cmap="viridis"); fig.colorbar(pc, ax=ax, label=r"$\log_{10}$")
                try: cs = ax.contour(xs, ms, Z, levels=lev, colors="w", linewidths=2.5); ax.clabel(cs, fmt="%g", fontsize=16)
                except ValueError: pass
                nh = 0
                for a_, b_ in zip(*np.where(np.isfinite(Z) & uninf(np.nan_to_num(Z, nan=0.0)))):
                    ax.add_patch(plt.Rectangle((xe[b_], me[a_]), xe[b_ + 1] - xe[b_], me[a_ + 1] - me[a_], fill=False, hatch="//", edgecolor="w", lw=0))
                    nh += 1
                if nh: title += f"\nhatched: uninformative ({why})" + ("  — all cells" if nh == np.isfinite(Z).sum() else "")
                ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xlab); ax.set_ylabel(r"$M_{200}\ [M_\odot]$")
            else:
                ax.text(0.5, 0.5, "unconstrained everywhere\n" + ("(mass-sheet degeneracy)" if not known else "(unresolved halo direction)"),
                        ha="center", va="center", transform=ax.transAxes, fontsize=24)
                ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
            ax.set_title(title)
    fixed = "M_Lz=1e6" if a.kind == "maps" else "y=0.1"
    fig.suptitle(f"PML+NFW (draw 0, z=1, M=1e6, q=1, {fixed}, kappa=0.1, |gamma|=0.1).  Significance from the Fisher at the true values; "
                 "white contours: 3 and 5 sigma / CI levels; blank cells: unconstrained", y=1.01)
    fig.tight_layout(); save(fig, f"{a.kind}_pmlnfw.png")

elif a.kind == "shear":
    # kappa known only: |gamma| itself is mass-sheet degenerate (unconstrained when kappa is free), and the sigma ratios of
    # log M_Lz and sin(beta) are identical for kappa free and known (both are mass-sheet invariants).
    # gamma = 0 rows are left out: zero shear is a degenerate point of (gamma1, gamma2) (several extra unresolved directions).
    rows = FA.load(a.rows, "shear_"); fig, axs = plt.subplots(1, 3, figsize=(36, 11)); known = ("kappa",)
    for m in ("pml", "pmlnfw"):
        for y, mk in ((0.1, "o"), (1.0, "s")):
            sel = sorted([(r, d) for r, d in rows if d["model"] == m and abs(r["y"] - y) < 1e-9], key=lambda t: np.hypot(t[0]["vals"]["gamma1"], t[0]["vals"]["gamma2"]))
            g, S, ratios = [], [], {n: [] for n in ("log10_mlz", "sin_ecl_lat")}
            for r, d in sel:
                gg = np.hypot(r["vals"]["gamma1"], r["vals"]["gamma2"])
                if gg <= 0: continue
                res = FA.analyse(r, m, d["info"], d["m_ref"], known); rk = FA.analyse(r, m, d["info"], d["m_ref"], known + ("gamma1", "gamma2"))
                g.append(gg); S.append(gg / res["gamma_abs"][1] if np.isfinite(res["gamma_abs"][1]) else np.nan)
                for n in ratios: ratios[n].append(res[n][1] / rk[n][1] if n in res and n in rk and np.isfinite(res[n][1]) and np.isfinite(rk[n][1]) else np.nan)
            ls = "-" if m == "pml" else "--"; lab = f"{FA.MODEL_LAB[m]}, y={y:g}"
            kw = dict(ls=ls, marker=mk, color=FA.MODEL_COL[m], lw=2.5, ms=13, mfc="none" if y > 0.5 else FA.MODEL_COL[m], mew=2.5, label=lab)
            for ax, v in zip(axs, (S, ratios["log10_mlz"], ratios["sin_ecl_lat"])):
                v = np.array(v, float)
                if np.isfinite(v).any(): ax.plot(g, v, **kw)
                else: ax.plot([], [], **dict(kw, label=lab + " (unconstrained)"))
    for ax, t in zip(axs, (r"shear detection $|\gamma|/\sigma(|\gamma|)$", r"$\sigma(\log M_{Lz})$: shear fitted / shear known",
                           r"$\sigma(\sin\beta)$ (sky): shear fitted / shear known")):
        ax.set_xscale("log"); ax.set_xlabel(r"$|\gamma|$"); ax.set_title(t); ax.grid(alpha=0.3); ax.legend(fontsize=16)
    axs[0].set_yscale("log"); axs[0].axhline(3, color="k", ls=":"); axs[0].text(0.011, 3.4, "3σ", fontsize=18)
    for ax in axs[1:]: ax.axhline(1, color="k", ls=":")
    fig.suptitle("External shear (draw 0, M_Lz=1e6, M200=1e11; kappa known — with kappa free |γ| is mass-sheet degenerate and the ratios are identical).\n"
                 "Ratio > 1: not knowing the shear costs precision; ≈ 1: the shear decouples.  Missing points: unconstrained at that |γ|", y=1.04)
    fig.tight_layout(); save(fig, "shear.png")

elif a.kind == "cumul":
    files = sorted(glob.glob(os.path.join(a.rows, "cumul_*.json"))); fig, axs = plt.subplots(1, len(files), figsize=(13 * len(files), 10), squeeze=False)
    for ax, f in zip(axs[0], files):
        d = json.load(open(f)); m = d["model"]; fm = np.array(d["f_max"])
        unc_ = []
        for nm_key, Fk_key, ls in (("names", "F", "-"), ("names_unlensed", "F_unlensed", ":")):
            names = d[nm_key]; known = [i for i, n in enumerate(names) if n != "kappa"]
            kn = [names[i] for i in known]; sig = []
            Fhs = d.get("F_half") if ls == "-" else None
            for j, Fk in enumerate(d[Fk_key]):     # same covariance treatment as every other plot (unresolved directions -> NaN)
                rr = dict(names=names, F=Fk, F_half=Fhs[j] if Fhs else None, binary=d["binary"], vals=d["vals"] if ls == "-" else {})
                res = FA.analyse(rr, "unlensed", known=("kappa",))   # free parameters only: no derived quantities needed
                sig.append([res[n][1] for n in kn])
            sig = np.array(sig, float)
            show = [n for n in kn if n in d["vals"]] if ls == "-" else ["log10_mchirp", "sin_ecl_lat", "tc"]
            for n in show:
                s = sig[:, kn.index(n)]
                if not np.isfinite(s[-1]): unc_.append(FA.LABEL.get(n, n)); continue
                ax.plot(fm, s / s[-1], ls=ls, lw=2.5, label=FA.LABEL.get(n, n) + (" (unlensed)" if ls == ":" else ""))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.8, 1e3); ax.axhline(1, color="k", lw=1)
        ax.set_xlabel("upper frequency of the analysis [Hz]"); ax.set_ylabel(r"$\sigma(f<f_{\max})\,/\,\sigma_{\rm full}$"); ax.set_title(FA.MODEL_LAB[m])
        ax.legend(fontsize=17, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13))
        if unc_: ax.text(0.03, 0.03, "unconstrained (full band):\n" + ", ".join(unc_), transform=ax.transAxes, ha="left", va="bottom", fontsize=17,
                         bbox=dict(fc="w", ec="0.6"))
    fig.suptitle("Where the information comes from: error using only frequencies below f_max (kappa known; draw 0, M_Lz=1e6, y=0.1)")
    fig.tight_layout(); save(fig, "cumulative_information.png")
else:
    sys.exit(f"unknown kind {a.kind}")
