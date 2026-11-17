"""Collect Fisher cluster outputs.  Run as a cluster job (light, but keep it off the headnode).

  steps : python scripts/fisher_plot.py steps --scan DIR --out DIR   -> plateau plots + recommended_steps.json
  grid  : python scripts/fisher_plot.py grid  --shards DIR --out DIR -> fisher_merged.csv + heatmaps / stability table
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["steps", "grid"])
ap.add_argument("--scan", default=None); ap.add_argument("--shards", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--tol", type=float, default=0.02, help="plateau: F_kk within +/-tol of the window median")
ap.add_argument("--cond-max", type=float, default=1e8, help="'stable' Fisher: normalised condition number below this")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def placeholder(fname, msg):
    """Always leave a figure behind, even when there is nothing to plot."""
    fig = plt.figure(figsize=(6, 2)); fig.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9, wrap=True)
    fig.savefig(os.path.join(a.out, fname), dpi=100); plt.close(fig)


def plateau(m, v, tol, min_len=3):
    """Longest run of consecutive multipliers whose F_kk stays within tol of the run's median -> (i0, i1)."""
    m, v = np.asarray(m), np.asarray(v); ok = np.isfinite(v) & (v > 0); best = None
    for i in range(len(m)):
        for j in range(i + min_len, len(m) + 1):
            s = v[i:j]
            if ok[i:j].all() and np.all(np.abs(s / np.median(s) - 1) < tol) and (best is None or j - i >= best[1] - best[0]):
                best = (i, j)
    return best


if a.mode == "steps":
    by = {}
    for f in sorted(glob.glob(os.path.join(a.scan, "stepscan_*.json"))):
        d = json.load(open(f)); by.setdefault(d["param"], []).append(d)
    rec, report = {}, []
    if not by:
        placeholder("step_scan.png", f"no stepscan_*.json found in {a.scan} - all scan jobs failed? see logs")
        json.dump({}, open(os.path.join(a.out, "recommended_steps.json"), "w")); sys.exit("no scan results; wrote empty recommended_steps.json (defaults will be used)")
    ncol = 4; nrow = int(np.ceil(len(by) / ncol)); fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, (name, ds) in zip(axs.flat, sorted(by.items())):
        picks = []
        for d in ds:
            m, v = np.array(d["mults"]), np.array(d["fkk"]); ref = np.nanmedian(v[v > 0]) if np.any(v > 0) else 1.0
            ax.loglog(m * d["base_step"], v / ref, "o-", ms=3, alpha=.8)
            pl = plateau(m, v, a.tol)
            if pl: picks.append(float(np.exp(np.mean(np.log(m[pl[0]:pl[1]]))) * d["base_step"]))
        ax.set_title(name, fontsize=9); ax.set_xlabel("step"); ax.set_ylabel("F_kk / median"); ax.set_ylim(0.1, 10)
        if picks:
            rec[name] = float(np.median(picks)); ax.axvline(rec[name], color="k", ls="--")
            report.append((name, rec[name], len(picks), len(ds), "OK"))
        else:
            report.append((name, ds[0]["base_step"], 0, len(ds), "NO PLATEAU - inspect the plot; default step kept"))
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "step_scan.png"), dpi=130)
    json.dump(rec, open(os.path.join(a.out, "recommended_steps.json"), "w"), indent=1)
    print(f"{'param':14s} {'step':>10s} plateau-fids  status")
    for n, s, k, t, st in report: print(f"{n:14s} {s:10.3g} {k}/{t}  {st}")
    print("wrote", os.path.join(a.out, "recommended_steps.json"))
else:
    rows = []
    for f in sorted(glob.glob(os.path.join(a.shards, "fisher*.json"))):
        d = json.load(open(f)); rows += [r for r in d["rows"] if r.get("ok")]
    if not rows:
        for k in ("sig_log10_mlz", "sig_y", "cond", "snr_diff"): placeholder(f"grid_{k}.png", "no successful Fisher points - see shard logs")
        sys.exit("no rows; wrote placeholder figures")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(os.path.join(a.out, "fisher_merged.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    for r in rows:
        r["stable"] = int(r["cond"] < a.cond_max and r["eig_min"] > 0 and np.isfinite(r.get("sig_y", np.nan)))
    print(f"{len(rows)} Fisher points, {sum(r['stable'] for r in rows)} stable (cond<{a.cond_max:g}, positive-definite)")
    # heatmaps of the lens-parameter errors vs (M_Lz, y), one panel per (z, m_tot); one set of files per lens model
    def ltag(l): return l.replace(":", "").replace("+", "")
    for lens0 in sorted({r["lens"] for r in rows}, key=lambda l: (l != "pm", l)):
        sel = [r for r in rows if r["lens"] == lens0 and r["q"] == rows[0]["q"] and r["inc"] == rows[0]["inc"]]
        if not sel: continue
        srcs = sorted({(r["z"], r["m_tot"]) for r in sel}); Ms = sorted({r["mass_lz"] for r in sel}); ys = sorted({r["y"] for r in sel})
        fx = "" if lens0 == "pm" else "_" + ltag(lens0)
        keys_l = ["sig_log10_mlz"] * (not lens0.startswith("halo")) + ["sig_log10_mhalo"] * (lens0 != "pm") + ["sig_y", "cond", "snr_diff"]
        for key, ttl in [(k, dict(sig_log10_mlz="sigma(log10 M_Lz)", sig_log10_mhalo="sigma(log10 M_halo)", sig_y="sigma(y)", cond="cond (normalised)", snr_diff="SNR of lensing imprint")[k]) for k in keys_l]:
            fig, axs = plt.subplots(1, len(srcs), figsize=(3.6 * len(srcs), 3.2), squeeze=False)
            for ax, (z, mt) in zip(axs[0], srcs):
                Z = np.full((len(ys), len(Ms)), np.nan)
                for r in sel:
                    if (r["z"], r["m_tot"]) == (z, mt) and r.get(key, "") != "": Z[ys.index(r["y"]), Ms.index(r["mass_lz"])] = r[key]
                Zp = np.where(Z > 0, Z, np.nan)
                kw = dict(norm=matplotlib.colors.LogNorm()) if np.isfinite(Zp).any() else {}
                im = ax.imshow(Zp, origin="lower", aspect="auto", cmap="viridis", **kw)
                ax.set_xticks(range(len(Ms)), [f"{m:.0e}" for m in Ms], fontsize=7); ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
                ax.set_xlabel("$M_{Lz}$" + (" (unit only)" if lens0.startswith("halo") else "")); ax.set_ylabel("y"); ax.set_title(f"z={z:g}, M={mt:.0e}", fontsize=8); fig.colorbar(im, ax=ax)
            fig.suptitle(f"{ttl}  ({lens0})"); fig.tight_layout(); fig.savefig(os.path.join(a.out, f"grid_{key}{fx}.png"), dpi=130); plt.close(fig)
    # point mass vs composite / halo-only at identical (z, m_tot, q, inc, M_Lz, y): does the halo help or hurt the lens measurement?
    base = {(r["z"], r["m_tot"], r["q"], r["inc"], r["mass_lz"], r["y"]): r for r in rows if r["lens"] == "pm"}
    others = sorted({r["lens"] for r in rows if r["lens"] != "pm"})
    if base and others:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for l in others:
            m = [(base[k], r) for r in rows if r["lens"] == l for k in [(r["z"], r["m_tot"], r["q"], r["inc"], r["mass_lz"], r["y"])] if k in base]
            for ax, key, ttl in zip(axs, ("sig_y", "sig_log10_mlz", "sig_log10_mchirp"), ("sigma(y)", "sigma(log10 M_Lz)", "sigma(log10 chirp mass)")):
                pts_ = [(b[key], r[key]) for b, r in m if key in b and key in r]
                if pts_: ax.loglog(*zip(*pts_), "o", ms=4, alpha=.7, label=l)
        for ax, ttl in zip(axs, ("sigma(y)", "sigma(log10 M_Lz)", "sigma(log10 chirp mass)")):
            lim = ax.get_xlim() + ax.get_ylim(); lo, hi = min(lim), max(lim); ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
            ax.set_xlabel("point mass"); ax.set_ylabel("other lens"); ax.set_title(ttl, fontsize=9); ax.legend(fontsize=6)
        fig.suptitle("same source and (M_Lz, y); below the dashed line = better measured than with the point-mass lens", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "compare_lenses.png"), dpi=130); plt.close(fig)
    # ranked table: best-measured lens parameters among stable points
    good = sorted([r for r in rows if r["stable"]], key=lambda r: r["sig_y"])[:15]
    print("\nTop 15 stable points by sigma(y):")
    for r in good: print({k: (float(f"{r[k]:.3g}") if isinstance(r[k], float) else r[k]) for k in ("lens", "z", "m_tot", "q", "mass_lz", "y", "snr_lensed", "sig_y", "sig_log10_mlz", "cond") if k in r})
