#!/usr/bin/env python3
"""Action 4: is the admission boundary distinguishable in content?

The question the article has to answer is not "do high and low scoring documents
differ" (obviously they do) but "do documents on opposite sides of the boundary
differ". Three comparisons, run identically:

  BOUNDARY  score == 2.500000 (CUT, half-to-even sends the tie down)
            vs score == 2.515625 (KEPT, one bfloat16 step higher)
  BAND      rounded-in (2.5 < s < 3.0, kept only by rounding) vs above nominal (s >= 3.0)
  CONTROL   s <= 2.0 vs s >= 4.0

The CONTROL is the point. An AUC near 0.5 on BOUNDARY means nothing on its own —
it could just mean bag-of-words cannot read educational quality. The control has to
come out high on the same features and the same model, or the null is uninformative.
A label-shuffled null is run on BOUNDARY as well.

Reads local shards only, and only the 94 unpartitioned dumps (see HANDOVER 6d).
"""
import glob, json, os, re, sys, time
from collections import Counter
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

SEED = 20260801
CAP = 15_000          # documents per group
N_SHARDS = 14         # local shards to scan, spread across the crawl series
TIE, NEXT_UP = np.float32(2.5), np.float32(2.515625)

GROUPS = {
    "tie_2.500000_CUT":  lambda s: s == TIE,
    "next_2.515625_KEPT": lambda s: s == NEXT_UP,
    "rounded_in":        lambda s: (s > TIE) & (s < 3.0),
    "above_nominal":     lambda s: s >= 3.0,
    "ctrl_low_<=2.0":    lambda s: s <= 2.0,
    "ctrl_high_>=4.0":   lambda s: s >= 4.0,
}

man = json.load(open("random_shard_manifest.json"))
clean = [d for d in sorted(man) if man[d]["shard"].startswith("train-")]
root = os.path.expanduser("~/fw-score2-random")
pick = [clean[i] for i in np.linspace(0, len(clean) - 1, N_SHARDS).astype(int)]
print(f"scanning {len(pick)} of {len(clean)} unpartitioned shards, cap {CAP:,}/group\n", flush=True)

rng = np.random.default_rng(SEED)
buf = {k: [] for k in GROUPS}
t0 = time.time()
for j, dump in enumerate(pick, 1):
    pf = pq.ParquetFile(os.path.join(root, man[dump]["path"]))
    for b in pf.iter_batches(batch_size=20000, columns=["text", "score", "url", "token_count"]):
        d = b.to_pydict()
        s = np.array(d["score"], dtype=np.float32)
        for k, fn in GROUPS.items():
            if len(buf[k]) >= CAP:
                continue
            for i in np.flatnonzero(fn(s))[: CAP - len(buf[k])]:
                buf[k].append((d["text"][i], d["url"][i], d["token_count"][i], float(s[i])))
    print(f"  [{j:2}/{len(pick)}] {dump:<17} " +
          " ".join(f"{k.split('_')[0]}:{len(buf[k]):,}" for k in GROUPS) +
          f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
    if all(len(v) >= CAP for v in buf.values()):
        print("  all groups full, stopping early", flush=True)
        break

print("\nGROUP DESCRIPTIVES")
print(f"  {'group':<22}{'n':>8}{'median tok':>12}{'mean tok':>10}{'distinct hosts':>16}")
host = lambda u: re.sub(r"^www\.", "", (u or "").split("/")[2].lower()) if "//" in (u or "") else "?"
for k, v in buf.items():
    tk = np.array([x[2] for x in v])
    hs = Counter(host(x[1]) for x in v)
    print(f"  {k:<22}{len(v):>8,}{np.median(tk):>12,.0f}{tk.mean():>10,.0f}{len(hs):>16,}")

# top hosts, boundary pair only — a reader will want to see them
for k in ("tie_2.500000_CUT", "next_2.515625_KEPT"):
    hs = Counter(host(x[1]) for x in buf[k]).most_common(8)
    print(f"  top hosts {k}: " + ", ".join(f"{h}({c})" for h, c in hs))


def auc(a, b, label, shuffle=False):
    """Balanced, 5-fold, TF-IDF + logistic regression. Same everything across tests."""
    n = min(len(buf[a]), len(buf[b]), CAP)
    ta = [buf[a][i][0] for i in rng.choice(len(buf[a]), n, replace=False)]
    tb = [buf[b][i][0] for i in rng.choice(len(buf[b]), n, replace=False)]
    X = ta + tb
    y = np.r_[np.zeros(n), np.ones(n)]
    if shuffle:
        y = rng.permutation(y)
    pipe = make_pipeline(
        TfidfVectorizer(lowercase=True, min_df=5, max_features=60_000,
                        strip_accents="unicode", sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    sc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=5)
    print(f"  {label:<46} AUC {sc.mean():.4f} +/- {sc.std():.4f}   (n={n:,}/group)", flush=True)
    return sc.mean()


print("\nCAN A CLASSIFIER TELL THE TWO SIDES APART?  (0.5 = indistinguishable)")
b_auc = auc("tie_2.500000_CUT", "next_2.515625_KEPT",
            "BOUNDARY  2.500000 cut  vs  2.515625 kept")
auc("tie_2.500000_CUT", "next_2.515625_KEPT",
    "  null: same pair, labels shuffled", shuffle=True)
band = auc("rounded_in", "above_nominal", "BAND      rounded-in  vs  above nominal")
ctrl = auc("ctrl_low_<=2.0", "ctrl_high_>=4.0", "CONTROL   score <= 2.0  vs  score >= 4.0")

print(f"\nThe control separates at AUC {ctrl:.3f}, so the features and model can see educational")
print(f"quality when it is there. On the two sides of the actual admission boundary they reach")
print(f"{b_auc:.3f}. Band comparison {band:.3f}.")

pd.DataFrame([{"group": k, "n": len(v),
               "median_tokens": float(np.median([x[2] for x in v])),
               "distinct_hosts": len({host(x[1]) for x in v})} for k, v in buf.items()]
             ).to_csv("marginal_groups.csv", index=False)
print(f"\nwrote marginal_groups.csv   total {(time.time()-t0)/60:.1f} min")
