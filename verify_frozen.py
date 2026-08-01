#!/usr/bin/env python3
"""Re-derive every published figure from the frozen intermediates. No downloads.

analyse_full.py reads 265 GB of shards and writes two small outputs on the way:
full_scores.parquet (one row per document: score, int_score, token_count, dump) and
full_tokenised_sample.parquet (the 188,000-document tokenisation sample). Everything
the article claims is computed from those two, so this script recomputes the lot and
checks it against the published numbers. A reader who does not want to pull the
corpus can run this instead and get a pass or a fail rather than a promise.

What it does NOT verify: the read from the raw shards into these two files, which is
the sampling and tokenisation step. For that, run download_random.py then
analyse_full.py and compare the log.

Run:  .venv/bin/python verify_frozen.py
Exits non-zero on any mismatch.
"""
import sys

import numpy as np
import pandas as pd

CONTENT = 510
FAILED = []


def check(label, got, want, tol=0.0):
    """Compare at the precision the article states, not at float precision."""
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f"  {'ok ' if ok else 'FAIL'}  {label:<52} {got!r:>18}   published {want!r}")
    if not ok:
        FAILED.append(label)


df = pd.read_parquet("full_scores.parquet")
sm = pd.read_parquet("full_tokenised_sample.parquet")
n, tt = len(df), int(df.token_count.sum())
s = df.score.values.astype(np.float32)
isc = df.int_score.values.astype(np.int64)
tok = df.token_count.values

print("SAMPLE")
check("documents", n, 84_005_795)
check("tokens", tt, 77_564_818_929)
check("dumps", df.dump.nunique(), 94)

print("\nM3  quantisation")
check("distinct score values", len(np.unique(s)), 233)
check("lowest score", float(s.min()), 1.5)
check("highest score", float(s.max()), 5.28125)
check("half-to-even matches int_score, %",
      round(float((np.round(np.clip(s, 0, 5)).astype(int) == isc).mean() * 100), 4), 100.0)
check("half-up matches int_score, %",
      round(float((np.floor(np.clip(s, 0, 5) + .5).astype(int) == isc).mean() * 100), 4), 99.4494)
tie = s == 2.5
check("documents exactly on 2.5", int(tie.sum()), 458_461)
check("tie as % of what half-up would retain",
      round(float(tie.sum() / ((isc >= 3).sum() + tie.sum()) * 100), 1), 2.5)

print("\nM1  rounding")
kept = isc >= 3
rin = kept & (s < 3.0)
above = s >= 3.0
for lab, m, d_want, t_want in [("cut", ~kept, 65_908_378, 58_910_929_618),
                               ("rounded-in", rin, 10_352_466, 10_431_542_863),
                               ("above nominal", above, 7_744_951, 8_222_346_448)]:
    check(f"{lab}: documents", int(m.sum()), d_want)
    check(f"{lab}: tokens", int(tok[m].sum()), t_want)
check("of retained, admitted by rounding: docs %",
      round(float(rin.sum() / kept.sum() * 100), 1), 57.2)
check("of retained, admitted by rounding: tokens %",
      round(float(tok[rin].sum() / tok[kept].sum() * 100), 1), 55.9)

print("\nM2  truncation")
check("tokenisation sample size", len(sm), 188_000)
check("BERT/GPT-2 median token ratio",
      round(float((sm.bert_tokens / sm.gpt2_tokens).median()), 3), 0.978)
k2 = sm.int_score >= 3
for lab, m, tr_want, seen_want in [
        ("all documents", pd.Series(True, index=sm.index), 54.1, 45.5),
        ("retained", k2, 58.2, 41.9),
        ("rounded-in band", k2 & (sm.score < 3.0), 56.8, 42.5),
        ("above nominal band", sm.score >= 3.0, 60.0, 41.1)]:
    d_ = sm[m]
    seen = np.minimum(d_.bert_tokens, CONTENT)
    check(f"{lab}: truncated %", round(float((d_.bert_tokens > CONTENT).mean() * 100), 1), tr_want)
    check(f"{lab}: tokens seen %", round(float(seen.sum() / d_.bert_tokens.sum() * 100), 1), seen_want)

print("\nTrend across the crawl series")
by = df.assign(rin=rin).groupby("dump", observed=True).rin.mean() * 100
bt = (sm.assign(tr=sm.bert_tokens > CONTENT).groupby("dump", observed=True).tr.mean() * 100
      ).reindex(by.index)
yr = by.index.str.extract(r"CC-MAIN-(\d{4})")[0].astype(int).values
for lab, v, first, last, r_want in [("rounded-in", by.values, 12.0, 13.3, 0.765),
                                    ("truncation", bt.values, 49.1, 58.0, 0.852)]:
    check(f"{lab}: first dump %", round(float(v[0]), 1), first)
    check(f"{lab}: last dump %", round(float(v[-1]), 1), last)
    check(f"{lab}: Pearson r with year", round(float(np.corrcoef(yr, v)[0, 1]), 3), r_want)

print(f"\n{'=' * 72}")
if FAILED:
    print(f"RESULT: FAIL — {len(FAILED)} figure(s) do not reproduce:")
    for f in FAILED:
        print(f"  {f}")
    sys.exit(1)
print("RESULT: PASS — every published figure reproduces from the frozen intermediates.")
