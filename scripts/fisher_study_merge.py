"""Merge the one-row files rows/fisher_study_<id>_r<k>.json (k = 0..n-1, original row order) into fisher/fisher_study_<id>.json."""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--id", required=True)
ap.add_argument("--n", type=int, required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
ds = [json.load(open(os.path.join(a.rows, f"fisher_study_{a.id}_r{k}.json"))) for k in range(a.n)]
d = ds[0]; d["rows"] = [r for x in ds for r in x["rows"]]; d["seconds"] = sum(x["seconds"] for x in ds); d["n_merged"] = a.n
bad = [r for r in d["rows"] if not r.get("ok")]
out = os.path.join(a.out, f"fisher_study_{a.id}.json"); json.dump(d, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
print("merged", len(d["rows"]), "rows,", len(bad), "failed ->", out)
if bad: raise SystemExit(f"{len(bad)} failed rows: {[r.get('error') for r in bad][:3]}")
