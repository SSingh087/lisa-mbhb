"""90% credible intervals for every parameter of every Fisher point, from the stored Fisher matrices (no sampling).
Median and 5%..95% percentiles (the 90% range) from 200k samples of N(truth, F^-1), truncated at the physical bounds; quote as median +plus -minus.
Writes fisher_90pct.csv (long format: one row per point x parameter) and fisher_90pct_lens.csv (lens parameters only, wide).
    python scripts/fisher_table.py --shards DIR --out DIR
"""
import argparse, csv, glob, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lisa_mbhb import demo, fisher as FI, inference as I  # noqa: E402

Z90 = 1.6448536269514722
ap = argparse.ArgumentParser(); ap.add_argument("--shards", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
LENSN = ("log10_mlz", "y", "log10_mhalo")
rng = np.random.default_rng(0)
UP = dict(q=1.0, spin1z=0.99, spin2z=0.99, cos_inc=1.0, sin_ecl_lat=1.0); LO = dict(q=0.02, spin1z=-0.99, spin2z=-0.99, cos_inc=-1.0, sin_ecl_lat=-1.0, y=0.0)
long_rows, wide = [], []
for f in sorted(glob.glob(os.path.join(a.shards, "fisher*.json"))):
    d = json.load(open(f))
    for r, m in zip(d["rows"], d["fisher"]):
        if not (r.get("ok") and m): continue
        names, F = m["names"], np.array(m["F"]); C, cond, _ = FI.covariance(F); sig = np.sqrt(np.clip(np.diag(C), 0, None))
        m1 = r["m_tot"] / (1 + r["q"]); p = demo.default_params(m1=m1, m2=r["q"] * m1, z=r["z"], inclination=r["inc"])
        t = dict(zip(I.SAMPLED, I.params_to_theta(p))); t["log10_mlz"] = np.log10(r["mass_lz"]); t["y"] = r["y"]
        if r["lens"].startswith(("nfw", "halo")): t["log10_mhalo"] = np.log10(float(r["lens"].split(":")[1]))
        x = rng.multivariate_normal([t[n] for n in names], C, size=200000, method="eigh"); ok = np.ones(len(x), bool)   # same bounds as the plots
        for k_, n_ in enumerate(names):
            if n_ in UP: ok &= x[:, k_] <= UP[n_]
            if n_ in LO: ok &= x[:, k_] >= LO[n_]
        pc = np.percentile(x[ok], [5, 50, 95], axis=0)
        base = dict(lens=r["lens"], z=r["z"], m_tot=r["m_tot"], q_src=r["q"], inc=r["inc"], mass_lz=r["mass_lz"], y_src=r["y"],
                    snr_lensed=r["snr_lensed"], snr_diff=r["snr_diff"], cond=cond)
        w = dict(base)
        for k_, (n, s) in enumerate(zip(names, sig)):
            tv = t[n]; p5, med, p95 = pc[:, k_]
            long_rows.append(dict(base, param=n, truth=tv, median=med, p05=p5, p95=p95, plus=p95 - med, minus=med - p5, sigma=s,
                                  rel90_percent=100 * 0.5 * (p95 - p5) / abs(med) if med else np.nan))
            if n in LENSN: w.update({f"{n}_truth": tv, f"{n}_median": med, f"{n}_plus": p95 - med, f"{n}_minus": med - p5})
        wide.append(w)
if not long_rows: sys.exit("no points")
for fn, rows in (("fisher_90pct.csv", long_rows), ("fisher_90pct_lens.csv", wide)):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.out, fn), "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=keys, restval=""); w_.writeheader(); w_.writerows(rows)
    print(fn, len(rows), "rows")
