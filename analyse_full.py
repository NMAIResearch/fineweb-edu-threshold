#!/usr/bin/env python3
"""Every number in the article, computed over the 110-dump random sample.

Supersedes analyse_pilot.py (5 fixed shards) and tokenizer_check.py (50k from those
same 5 shards). Same measurements, correct sample, one pass.

  M1  rounding      the admission boundary is int_score>=3, i.e. raw score STRICTLY > 2.5
  M2  truncation    measured in the classifier's own tokenizer against a 510-token
                    content budget (512 minus [CLS] and [SEP])
  M3  quantisation  bfloat16 puts a grid point exactly on 2.5, where half-to-even cuts

Scores/counts are read for EVERY row (small numeric columns, cheap). Text is read only
for a two-stage sample: random row groups per file, then random rows within them. That
keeps the tokenisation honest without reading 265 GB of text off disk.

Run:  .venv/bin/python analyse_full.py [data_root]
      default data_root = ~/fw-score2-random/data
"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/fw-score2-random/data")
SEED = 20260801
GROUPS_PER_FILE = 100     # random row groups per shard (cluster stage)
ROWS_PER_GROUP = 20       # random rows within each group -> ~2000 docs per dump
MAXLEN, CONTENT = 512, 510
NUM = ["score", "int_score", "token_count"]

files = sorted(glob.glob(os.path.join(ROOT, "*", "*.parquet")))
manifest = json.load(open("random_shard_manifest.json"))
print(f"data root: {ROOT}")
print(f"shards found: {len(files)} of {len(manifest)} in the manifest")
if len(files) != len(manifest):
    missing = sorted(set(manifest) - {f.split("/")[-2] for f in files})
    print(f"⚠️  INCOMPLETE — {len(missing)} dumps missing, e.g. {missing[:5]}")
    print("   Re-run download_random.py. Results below are PARTIAL.\n")
assert files, "no parquet files found"

# The 2024 and 2025 dumps partition documents across shards BY SCORE, so one shard
# from them is a score-stratified slice, not a random sample. They are excluded from
# every headline; the article uses the 94 shuffled dumps spanning 2013 to 2023.
# Set FW_INCLUDE_ALL=1 to measure the (indefensible) all-110 version for comparison.
if not os.environ.get("FW_INCLUDE_ALL"):
    kept_files = [f for f in files if int(f.split("/")[-2].split("-")[2]) < 2024]
    print(f"excluding {len(files) - len(kept_files)} score-partitioned 2024-25 dumps; "
          f"analysing {len(kept_files)}")
    files = kept_files

tok = Tokenizer.from_file(hf_hub_download("HuggingFaceFW/fineweb-edu-classifier", "tokenizer.json"))
tok.no_truncation(); tok.no_padding()
assert tok.encode("x").ids[0] == 101

rng = np.random.default_rng(SEED)
num_frames, samp_rows = [], []
t0 = time.time()

for j, f in enumerate(files, 1):
    dump = f.split("/")[-2]
    pf = pq.ParquetFile(f)

    d = pf.read(columns=NUM).to_pandas()
    # 93M rows across 110 shards: a per-row dump string is the memory risk, and
    # float64 scores are pointless when every value is bfloat16-exact.
    # token_count stays int64 — the corpus total overflows int32.
    d["score"] = d.score.astype(np.float32)
    d["int_score"] = d.int_score.astype(np.int8)
    d["token_count"] = d.token_count.astype(np.int64)
    d["dump"] = pd.Categorical([dump] * len(d), categories=[f.split("/")[-2] for f in files])
    num_frames.append(d)

    ng = pf.num_row_groups
    gidx = rng.choice(ng, size=min(GROUPS_PER_FILE, ng), replace=False)
    texts, meta = [], []
    for g in gidx:
        t = pf.read_row_group(int(g), columns=["text"] + NUM).to_pydict()
        m = len(t["text"])
        ridx = rng.choice(m, size=min(ROWS_PER_GROUP, m), replace=False)
        for i in ridx:
            texts.append(t["text"][i])
            meta.append((t["token_count"][i], t["score"][i], t["int_score"][i]))
    for (tc, sc, isc), e in zip(meta, tok.encode_batch(texts)):
        samp_rows.append((dump, tc, sc, isc, len(e.ids)))

    print(f"  [{j:3}/{len(files)}] {dump:<18} {len(d):>9,} rows, "
          f"{len(texts):>5,} tokenised  ({(time.time()-t0)/60:.1f} min)", flush=True)

df = pd.concat(num_frames, ignore_index=True); del num_frames
sm = pd.DataFrame(samp_rows, columns=["dump", "gpt2_tokens", "score", "int_score", "bert_tokens"])
df.to_parquet("full_scores.parquet"); sm.to_parquet("full_tokenised_sample.parquet")

n, tt = len(df), df.token_count.sum()
s, isc = df.score.values, df.int_score.values.astype(np.int64)  # score already float32, bf16-exact
print(f"\n{'='*72}\nDOCUMENTS {n:,}   TOKENS {tt:,}   DUMPS {df.dump.nunique()}\n{'='*72}")

# ---- M3: quantisation --------------------------------------------------------
bf16 = ((s.view(np.uint32) & 0xFFFF) == 0)
u, c = np.unique(s, return_counts=True)
tie = s == 2.5
print("\nM3  BFLOAT16 QUANTISATION")
print(f"  scores exactly representable in bfloat16: {bf16.mean()*100:.4f}%")
print(f"  distinct score values across {n:,} documents: {len(u):,}")
print(f"  score range {s.min():.6f} to {s.max():.6f}")
print(f"  grid neighbours of the 2.5 tie point:")
for v, k in zip(u[(u >= 2.46) & (u <= 2.55)], c[(u >= 2.46) & (u <= 2.55)]):
    print(f"    {v:<10.6f} {k:>10,}" + ("   <-- TIE, half-to-even -> 2, CUT" if v == 2.5 else ""))
print(f"  half-to-even matches int_score for {(np.round(np.clip(s,0,5)).astype(int)==isc).mean()*100:.4f}% of rows")
print(f"  half-up      matches int_score for {(np.floor(np.clip(s,0,5)+.5).astype(int)==isc).mean()*100:.4f}% of rows")
print(f"  TIE COHORT: {tie.sum():,} docs = {tie.mean()*100:.2f}% of all, "
      f"{tie.sum()/((isc>=3).sum()+tie.sum())*100:.1f}% of what half-up would retain")
for b in (1.5, 3.5, 4.5):
    m = s == b
    if m.sum():
        print(f"    also exactly {b}: {m.sum():,} docs -> int_score {sorted(set(isc[m]))}")

# ---- M1: rounding ------------------------------------------------------------
kept = isc >= 3
print("\nM1  ROUNDING  (boundary = int_score>=3, i.e. score strictly > 2.5)")
print(f"  {'band':<26}{'docs':>13}{'doc %':>8}{'tokens':>17}{'tok %':>8}")
for lab, m in [("cut (int_score<3)", ~kept),
               ("rounded-in (2.5<s<3.0)", kept & (s < 3.0)),
               ("above nominal (s>=3.0)", s >= 3.0)]:
    d_, t_ = m.sum(), df.token_count.values[m].sum()
    print(f"  {lab:<26}{d_:>13,}{d_/n*100:>7.1f}%{t_:>17,}{t_/tt*100:>7.1f}%")
kr = kept & (s < 3.0)
print(f"  OF THE RETAINED CORPUS: {kr.sum()/kept.sum()*100:.1f}% of docs and "
      f"{df.token_count.values[kr].sum()/df.token_count.values[kept].sum()*100:.1f}% of tokens "
      f"were admitted by rounding")

# ---- M2: truncation ----------------------------------------------------------
N = len(sm)
r = sm.bert_tokens / sm.gpt2_tokens
k2 = sm.int_score >= 3
print(f"\nM2  TRUNCATION  (classifier tokenizer, {CONTENT}-token content budget; N={N:,})")
print(f"  BERT/GPT-2 token ratio: median {r.median():.3f}, "
      f"BERT higher for {(r>1).mean()*100:.1f}% of documents")
print(f"  {'population':<28}{'truncated':>11}{'tokens seen':>13}{'median share':>14}")
for lab, m in [("all documents", pd.Series(True, index=sm.index)),
               ("retained (int_score>=3)", k2),
               ("rounded-in band", k2 & (sm.score < 3.0)),
               ("above nominal band", sm.score >= 3.0)]:
    d_ = sm[m]; seen = np.minimum(d_.bert_tokens, CONTENT)
    print(f"  {lab:<28}{(d_.bert_tokens>CONTENT).mean()*100:>10.1f}%"
          f"{seen.sum()/d_.bert_tokens.sum()*100:>12.1f}%{np.median(seen/d_.bert_tokens)*100:>13.1f}%")
p = (sm.bert_tokens > CONTENT).mean()
print(f"  headline {p*100:.1f}% +/- {1.96*np.sqrt(p*(1-p)/N)*100:.2f} pp "
      f"(95% CI, binomial; clustered sampling so treat as a floor on the interval)")

# ---- drift across the crawl series ------------------------------------------
print("\nBY DUMP")
a = df.assign(kept=kept, rin=kr).groupby("dump").agg(
    docs=("score", "size"), med=("score", "median"),
    rounded_in_pct=("rin", lambda x: x.mean()*100))
b = sm.assign(tr=sm.bert_tokens > CONTENT).groupby("dump").agg(
    n_tok=("tr", "size"), trunc_pct=("tr", lambda x: x.mean()*100))
out = a.join(b)
print(out.to_string(float_format=lambda x: f"{x:.1f}"))
out.to_csv("by_dump.csv")

yr = out.index.str.extract(r"CC-MAIN-(\d{4})")[0].astype(int).values
for col in ("rounded_in_pct", "trunc_pct"):
    v = out[col].values
    ok = ~np.isnan(v)
    print(f"\n{col}: {v[ok][0]:.1f}% ({yr[ok][0]}) -> {v[ok][-1]:.1f}% ({yr[ok][-1]}), "
          f"Pearson r with year = {np.corrcoef(yr[ok], v[ok])[0,1]:+.3f}")

print("\nwrote full_scores.parquet, full_tokenised_sample.parquet, by_dump.csv")
print(f"total {(time.time()-t0)/60:.1f} min")
