"""Fisher matrix (MBH + lens parameters) for one source and a list of lenses.  CLUSTER JOB, not headnode.

    python scripts/fisher_point.py --z 1 --m-tot 1e6 --q 0.5 --inc 0.9 --lens pm --mass-lz 1e6 1e7 --y 0.3 1 \
        --steps recommended_steps.json --out DIR --tag _id
Unlensed MBH derivatives are computed once per source (22 waveform evals); each (lens, M_Lz, y) then only costs a
few lens-amplification evaluations.  Writes DIR/fisher<tag>.json (rows + full F matrices) and .csv.
"""
import argparse, csv, itertools, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lisa_mbhb import demo, fisher as FI, waveform as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--approximant", default="IMRPhenomD")
ap.add_argument("--z", type=float, required=True)
ap.add_argument("--m-tot", type=float, required=True, help="source-frame total mass [Msun]")
ap.add_argument("--q", type=float, default=0.5)
ap.add_argument("--inc", type=float, default=0.9)
ap.add_argument("--lens", nargs="+", default=["pm"], help="pm | nfw:<M200> (PM + halo) | halo:<M200> (halo only)")
ap.add_argument("--mass-lz", type=float, nargs="+", default=[1e7])
ap.add_argument("--y", type=float, nargs="+", default=[1.0])
ap.add_argument("--zl-frac", type=float, default=0.5)
ap.add_argument("--steps", default=None, help="JSON {param: step} (from fisher_plot.py steps); default = fisher.STEP_*")
ap.add_argument("--step-mult", type=float, default=1.0, help="multiply every default/recommended step")
ap.add_argument("--max-wall-s", type=float, default=3000)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
a = ap.parse_args()

steps = {**FI.STEP_MBH, **FI.STEP_LENS}
if a.steps and os.path.exists(a.steps):
    steps.update(json.load(open(a.steps))); print("using steps from", a.steps)
steps = {k: v * a.step_mult for k, v in steps.items()}

t0 = time.time()
m1 = a.m_tot / (1 + a.q); m2 = a.q * m1
p = demo.default_params(m1=m1, m2=m2, z=a.z, inclination=a.inc)
S = FI.SourceDerivs(p, W.WaveformConfig(approximant=a.approximant, tc_ref=p["tc"]))
print(f"grid {S.f.size} pts, SNR_unl {np.sqrt(S.ip(S.base, S.base)):.2f}, setup {time.time()-t0:.0f}s", flush=True)

rows, mats, complete = [], [], True
for lens, M, y in itertools.product(a.lens, a.mass_lz, a.y):
    if time.time() - t0 > a.max_wall_s:
        complete = False; print("wall limit reached, stopping early"); break
    try:
        r = S.fisher(lens, M, y, a.z, a.zl_frac, steps)
    except Exception as e:                                   # keep going; flagged row
        print("FAILED", lens, M, y, repr(e), flush=True); rows.append(dict(lens=lens, mass_lz=M, y=y, z=a.z, m_tot=a.m_tot, q=a.q, inc=a.inc, ok=0)); mats.append(None); continue
    row = dict(lens=lens, mass_lz=M, y=y, z=a.z, m_tot=a.m_tot, q=a.q, inc=a.inc, ok=1,
               snr_unlensed=r["snr_unlensed"], snr_lensed=r["snr_lensed"], snr_diff=r["snr_diff"],
               cond=r["cond"], cond_raw=r["cond_raw"], eig_min=r["eig_min"])
    row.update({f"sig_{k}": v for k, v in r["sigma"].items()})
    row.update({f"sigK_{k}": v for k, v in r["sigma_lensknown"].items()})
    row.update({f"sigU_{k}": v for k, v in r["sigma_unlensed"].items()})
    rows.append(row); mats.append(dict(names=r["names"], F=r["F"].tolist()))
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k in
           ("lens", "mass_lz", "y", "snr_lensed", "snr_diff", "cond", "sig_log10_mlz", "sig_y", "sig_log10_mchirp")}, flush=True)

os.makedirs(a.out, exist_ok=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k)) if rows else []
json.dump(dict(args=vars(a), steps=steps, complete=complete, rows=rows, fisher=mats), open(os.path.join(a.out, f"fisher{a.tag}.json.tmp"), "w"))
with open(os.path.join(a.out, f"fisher{a.tag}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
os.replace(os.path.join(a.out, f"fisher{a.tag}.json.tmp"), os.path.join(a.out, f"fisher{a.tag}.json"))   # json present => finished
try:                                                      # per-source plot, always written
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ok = [r for r in rows if r.get("ok")]
    keys_p = [k for k in ("sig_y", "sig_log10_mlz", "sig_log10_mhalo") if ok and k in ok[0]] or ["sig_y"]
    fig, axs = plt.subplots(1, len(keys_p), figsize=(4.5 * len(keys_p), 3.2), squeeze=False)
    for ax, key in zip(axs[0], keys_p):
        for (ln, M), g in itertools.groupby(sorted(ok, key=lambda r: (r["lens"], r["mass_lz"], r["y"])), key=lambda r: (r["lens"], r["mass_lz"])):
            g = list(g); ax.loglog([r["y"] for r in g], [r.get(key, np.nan) for r in g], "o-", label=f"{ln} M={M:.0e}")
        ax.set_xlabel("y"); ax.set_ylabel(key); ax.legend(fontsize=6)
    fig.suptitle(f"z={a.z:g} M={a.m_tot:.0e} q={a.q:g}" + ("" if ok else "  (no successful points)"), fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"fisher{a.tag}.png"), dpi=110); plt.close(fig)
except Exception as e: print("plot failed", repr(e))
if not complete: sys.exit(3)
