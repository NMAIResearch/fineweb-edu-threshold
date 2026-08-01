#!/usr/bin/env python3
"""Two checks forced by the Gemini red-team review (2026-08-01).

CHECK A (their claim 5b): "excluding the 2024-25 dumps invalidates the longitudinal
truncation trend". Those dumps partition documents across shards by score, so a whole-dump
truncation rate cannot be estimated from one shard. But conditioning on a SINGLE score
grid point removes the composition bias entirely: within one exact score value, a
score-partitioned shard is still a fair sample of documents at that score. So measure
truncation among documents scoring exactly 2.0 (a populated grid point near the mode) for
every dump, 2013 to 2025. If the trend holds within-stratum, the exclusion did not
manufacture it.

CHECK B (their claim 3b): "points either side of any threshold are trivially similar, so
AUC 0.5 at the boundary says nothing about THIS filter". Correct as stated, and testable:
run the identical one-step comparison at several other grid points. If every pair scores
about 0.5, the result is a property of thresholding a continuous score, not a finding about
2.5, and section 4 has to be reframed rather than defended.
"""
import json, os, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

SEED, CONTENT = 20260801, 510
man = json.load(open("random_shard_manifest.json"))
root = os.path.expanduser("~/fw-score2-random")
tok = Tokenizer.from_file(hf_hub_download("HuggingFaceFW/fineweb-edu-classifier", "tokenizer.json"))
tok.no_truncation(); tok.no_padding()
rng = np.random.default_rng(SEED)

# ---------------- CHECK A ----------------------------------------------------
PER, TARGET = 700, np.float32(2.0)
print("CHECK A: truncation among documents scoring EXACTLY 2.0, by dump\n", flush=True)
rows, t0 = [], time.time()
for j, dump in enumerate(sorted(man), 1):
    pf = pq.ParquetFile(os.path.join(root, man[dump]["path"]))
    txt = []
    for b in pf.iter_batches(batch_size=20000, columns=["text", "score"]):
        d = b.to_pydict(); s = np.array(d["score"], dtype=np.float32)
        for i in np.flatnonzero(s == TARGET)[: PER - len(txt)]:
            txt.append(d["text"][i])
        if len(txt) >= PER:
            break
    if len(txt) < 100:
        print(f"  {dump}: only {len(txt)} docs at 2.0, skipped", flush=True)
        continue
    L = np.array([len(e.ids) for e in tok.encode_batch(txt)])
    rows.append({"dump": dump, "year": int(dump.split("-")[2]), "n": len(L),
                 "trunc_pct": (L > CONTENT).mean() * 100, "median_tok": float(np.median(L)),
                 "partitioned": not man[dump]["shard"].startswith("train-")})
    if j % 20 == 0:
        print(f"  ...{j}/110 dumps ({(time.time()-t0)/60:.1f} min)", flush=True)

a = pd.DataFrame(rows)
a.to_csv("redteam_check_a.csv", index=False)
for lab, sub in [("2013-2023 only (94 unpartitioned)", a[~a.partitioned]), ("ALL dumps 2013-2025", a)]:
    r = np.corrcoef(sub.year, sub.trunc_pct)[0, 1]
    print(f"\n  {lab}: n={len(sub)} dumps, {sub.trunc_pct.iloc[0]:.1f}% -> {sub.trunc_pct.iloc[-1]:.1f}%, "
          f"Pearson r with year {r:+.3f}")
print("\n  by year (mean across dumps):")
print(a.groupby("year").agg(dumps=("dump", "size"), trunc=("trunc_pct", "mean"),
                            med=("median_tok", "mean")).to_string(float_format=lambda x: f"{x:.1f}"))

# ---------------- CHECK B ----------------------------------------------------
STEP, CAP = np.float32(0.015625), 8000
POINTS = [np.float32(v) for v in (1.75, 2.0, 2.25, 2.5, 2.75, 3.0)]
print(f"\n\nCHECK B: same one-step AUC test at {len(POINTS)} grid points, {CAP:,}/group\n", flush=True)
clean = [d for d in sorted(man) if man[d]["shard"].startswith("train-")]
need = {v: {"lo": [], "hi": []} for v in POINTS}
for dump in [clean[i] for i in np.linspace(0, len(clean) - 1, 20).astype(int)]:
    pf = pq.ParquetFile(os.path.join(root, man[dump]["path"]))
    for b in pf.iter_batches(batch_size=20000, columns=["text", "score"]):
        d = b.to_pydict(); s = np.array(d["score"], dtype=np.float32)
        for v in POINTS:
            for key, mask in (("lo", s == v), ("hi", s == np.float32(v + STEP))):
                q = need[v][key]
                if len(q) < CAP:
                    for i in np.flatnonzero(mask)[: CAP - len(q)]:
                        q.append(d["text"][i])
    if all(len(q) >= CAP for v in POINTS for q in need[v].values()):
        break

print(f"  {'pair':<28}{'n/group':>9}{'AUC':>9}")
out = []
for v in POINTS:
    lo, hi = need[v]["lo"], need[v]["hi"]
    n = min(len(lo), len(hi))
    if n < 1500:
        print(f"  {float(v):.6f} vs +1 step: only {n}, skipped"); continue
    X = lo[:n] + hi[:n]; y = np.r_[np.zeros(n), np.ones(n)]
    pipe = make_pipeline(TfidfVectorizer(lowercase=True, min_df=5, max_features=60_000,
                                         strip_accents="unicode", sublinear_tf=True),
                         LogisticRegression(max_iter=2000))
    sc = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                         scoring="roc_auc", n_jobs=5)
    flag = "  <-- the admission boundary" if float(v) == 2.5 else ""
    print(f"  {float(v):.6f} vs {float(v)+0.015625:.6f}{n:>9,}{sc.mean():>9.4f}{flag}", flush=True)
    out.append({"point": float(v), "n": n, "auc": sc.mean(), "sd": sc.std()})
pd.DataFrame(out).to_csv("redteam_check_b.csv", index=False)
print("\nwrote redteam_check_a.csv, redteam_check_b.csv")
