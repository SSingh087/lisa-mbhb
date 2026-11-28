"""Full-matrix step scan: 90% CI (kappa known) of every lens parameter and of the mass-sheet invariants vs the finite-difference
step, one panel per quantity.  A converged Fisher shows a plateau.
  default     : one line per (stencil, GWLENS solver setting), single point        -> fullscan_<model>.png
  --by-point  : one line per lens point (M_Lz, y, M200), single solver setting     -> fullscan_<model>.png
                plus fullscan_diagnostic.png: 1-|corr| of the NFW derivative pair (psi0, x_s) and sqrt(F_kk) vs step
Writes fullscan_flatness.txt (smallest mean |dlog CI| over two neighbouring step pairs) and, with --by-point,
fullscan_plateau.txt (longest flat run per line: |dlog CI| < 0.05 between neighbours for every quantity).
    python scripts/fisher_study_fullscan_plot.py --scan DIR --out DIR [--by-point]"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from lisa_mbhb import fisher_analysis as FA  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--scan", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--by-point", action="store_true"); ap.add_argument("--absolute", action="store_true", help="x axis = absolute step of kappa, gamma, psi0, x_s (y, M_Lz steps are 10x larger)"); ap.add_argument("--flat-tol", type=float, default=0.05)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); FA.style(1.0)


def cfg_label(d):
    sv = d["solver"]
    return f"{d['stencil'] + 1 if d['stencil'] == 4 else 2}-point, " + ("no interpolation" if sv.get("interpolate") is False else f"interp tol {sv.get('interp_tol', 1e-9):g}")


def point_label(d):
    return rf"$M_{{Lz}}$={d['m_ref']:.0e}, y={d.get('y', 0.1):g}, $M_{{200}}$={d.get('m200', 1e11):.0e}".replace("e+0", "e").replace("e+", "e")


runs = {}
for f in glob.glob(os.path.join(a.scan, "fullscan_*.json")):
    d = json.load(open(f))
    runs.setdefault(d["model"], {}).setdefault(point_label(d) if a.by_point else cfg_label(d), []).append(d)
XLAB = r"absolute step of $\kappa,\gamma,\psi_0,x_s$  ($y$, $M_{Lz}$: 10$\times$)" if a.absolute else "step / stable step"
xof = lambda d: d["lens_steps"]["kappa"] if a.absolute else d["mult"]
summary, plateau, diag, eig = [], [], {}, {}
for model, groups in sorted(runs.items()):
    Q = None; lines = {}
    for key, ds in groups.items():
        ds = sorted(ds, key=lambda d: d["mult"]); pts = []
        for d in ds:
            res = FA.analyse(dict(names=d["names"], F=d["F"], vals=d["vals"], binary=d["binary"]), model, d["info"], d["m_ref"], ("kappa",))
            pts.append((xof(d), res))
            F_ = np.array(d["F"]); k_ = [i for i, x in enumerate(d["names"]) if x != "kappa"]; F_ = F_[np.ix_(k_, k_)]; s_ = np.sqrt(np.diag(F_))
            ev = np.linalg.eigvalsh(0.5 * (F_ + F_.T) / np.outer(s_, s_)); eig.setdefault((model, key), []).append((xof(d), ev / ev[-1]))
            if "log10_psi0" in d["names"]:
                F = np.array(d["F"]); i, j = d["names"].index("log10_psi0"), d["names"].index("log10_xs")
                diag.setdefault((model, key), []).append((xof(d), 1 - abs(F[i, j]) / np.sqrt(F[i, i] * F[j, j]), np.sqrt(F[i, i]), np.sqrt(F[j, j])))
        q = [n for n in pts[0][1] if not n.startswith("_") and n != "kappa" and (n in ds[0]["vals"] or n.endswith("_ms") or n.startswith("log10_m") or n == "log10_c200")]
        Q = q if Q is None else [n for n in Q if n in q] + [n for n in q if n not in Q]
        lines[key] = pts
    nc = 4; nr = -(-len(Q) // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(26, 5.6 * nr), squeeze=False)
    for ax in axs.ravel()[len(Q):]: ax.axis("off")
    for ax, n in zip(axs.ravel(), Q):
        for key, pts in sorted(lines.items()):
            m = np.array([p[0] for p in pts]); hw = np.array([FA.hw90(p[1], n) for p in pts])
            ax.plot(m, hw, "o-", lw=2.2, ms=5, label=key)
            ok = np.isfinite(hw)
            if ok.sum() >= 3:
                r = np.abs(np.diff(np.log10(hw[ok]))); summary.append((model, key, n, float(np.min(np.convolve(r, np.ones(2), "valid")) / 2)))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(FA.LABEL.get(n, n)); ax.set_xlabel(XLAB); ax.set_ylabel("90% CI")
    axs.ravel()[0].legend(fontsize=12)
    for key, pts in sorted(lines.items()):   # longest run of steps over which EVERY quantity is flat
        m = np.array([p[0] for p in pts]); H = np.array([[FA.hw90(p[1], n) for n in Q] for p in pts])
        with np.errstate(invalid="ignore", divide="ignore"):
            flat = np.all(~np.isfinite(H[1:]) | ~np.isfinite(H[:-1]) | (np.abs(np.log10(H[1:] / H[:-1])) < a.flat_tol), axis=1)
        best, cur, s0 = (0, 0), 0, 0
        for k, fl in enumerate(flat):
            if fl:
                if cur == 0: s0 = k
                cur += 1
                if cur > best[1] - best[0]: best = (s0, k + 1)
            else: cur = 0
        plateau.append((model, key, (m[best[0]], m[best[1]]) if best[1] > best[0] else None, best[1] - best[0]))
    what = "one line per lens point (2-point stencil, default solver)" if a.by_point else "draw 0, M_Lz=1e6, y=0.1, M200=1e11"
    fig.suptitle(f"{FA.MODEL_LAB[model]}: full-Fisher step scan ({what}, kappa known).  Converged = flat", y=1.0)
    fig.tight_layout(); fn = os.path.join(a.out, f"fullscan_{model}.png"); fig.savefig(fn, bbox_inches="tight"); plt.close(fig); print("wrote", fn)

if diag:
    models = sorted({m for m, _ in diag})
    fig, axs = plt.subplots(len(models), 3, figsize=(26, 6.5 * len(models)), squeeze=False)
    for r_, model in enumerate(models):
        for (m, key), v in sorted(diag.items()):
            if m != model: continue
            v = np.array(sorted(v))
            axs[r_, 0].plot(v[:, 0], v[:, 1], "o-", lw=2, ms=5, label=key)
            axs[r_, 1].plot(v[:, 0], v[:, 2] / v[-1, 2], "o-", lw=2, ms=5, label=key)
            axs[r_, 2].plot(v[:, 0], v[:, 3] / v[-1, 3], "o-", lw=2, ms=5, label=key)
        for ax, t in zip(axs[r_], [r"$1-|\rho(\partial_{\psi_0}h,\ \partial_{x_s}h)|$", r"$\sqrt{F_{\psi_0\psi_0}}$ / value at largest step",
                                   r"$\sqrt{F_{x_s x_s}}$ / value at largest step"]):
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(XLAB); ax.set_title(f"{FA.MODEL_LAB[model]}: {t}")
        axs[r_, 0].legend(fontsize=12)
    fig.tight_layout(); fn = os.path.join(a.out, "fullscan_diagnostic.png"); fig.savefig(fn, bbox_inches="tight"); plt.close(fig); print("wrote", fn)

if eig:   # eigen-spectrum of the normalised Fisher (kappa known) vs step: a converged direction is flat ("valley" floor)
    models = sorted({m for m, _ in eig}); keys = sorted({k for _, k in eig})
    fig, axs = plt.subplots(len(models), len(keys), figsize=(7 * len(keys), 6 * len(models)), squeeze=False, sharey=True)
    for r_, model in enumerate(models):
        for c_, key in enumerate(keys):
            ax = axs[r_, c_]; v = sorted(eig.get((model, key), []), key=lambda t: t[0])
            if not v: ax.axis("off"); continue
            x = np.array([t[0] for t in v]); E = np.array([t[1] for t in v])
            for j in range(min(6, E.shape[1])):
                ax.plot(x, np.clip(E[:, j], 1e-18, None), "o-", lw=2, ms=4, label=f"eigenvalue {j + 1}")
            ax.axhspan(1e-18, 2.2e-16 * E.shape[1], color="0.85", zorder=0); ax.text(x[0], 3e-17, "machine precision", fontsize=13)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-17, 1e-5); ax.set_xlabel(XLAB)
            ax.set_title(f"{FA.MODEL_LAB[model]}: {key}", fontsize=15)
        axs[r_, 0].set_ylabel(r"eigenvalue / largest (normalised $F$, $\kappa$ known)")
    axs[0, 0].legend(fontsize=12)
    fig.suptitle("Smallest eigenvalues of the Fisher vs step: a resolved direction is flat (valley); one falling as step$^{-2}$ into "
                 "machine precision is unresolved", y=1.0)
    fig.tight_layout(); fn = os.path.join(a.out, "fullscan_eigen.png"); fig.savefig(fn, bbox_inches="tight"); plt.close(fig); print("wrote", fn)

with open(os.path.join(a.out, "fullscan_flatness.txt"), "w") as fh:
    fh.write("model | line | quantity | best mean |dlog10 CI| over two neighbouring step pairs (0 = perfectly flat)\n")
    for s_ in sorted(summary): fh.write(f"{s_[0]:7s} | {s_[1]:45s} | {s_[2]:16s} | {s_[3]:.3g}\n")
with open(os.path.join(a.out, "fullscan_plateau.txt"), "w") as fh:
    fh.write(f"model | line | longest step range where every quantity changes by < {a.flat_tol} dex between neighbouring steps | n intervals\n")
    for p_ in sorted(plateau, key=lambda t: (t[0], t[1])):
        fh.write(f"{p_[0]:7s} | {p_[1]:45s} | {'none' if p_[2] is None else f'{p_[2][0]:g} .. {p_[2][1]:g}'} | {p_[3]}\n")
print("wrote flatness + plateau tables")
