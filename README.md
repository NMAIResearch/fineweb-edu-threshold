# What actually admits a document to FineWeb-Edu

Three operational details in a published quality filter, measured across 84 million
documents and 94 Common Crawl snapshots.

**Paper (DOI):** [10.5281/zenodo.21740082](https://doi.org/10.5281/zenodo.21740082) · CC BY 4.0

The article is `fineweb_edu_threshold.pdf` (source: `article_draft_v2.md`). This
bundle lets a reader reproduce every figure without the author's cooperation.

## What is measured

1. **Rounding.** The recommended `int_score >= 3` is a rounded integer, so the
   effective admission boundary is a raw score just above 2.5. 57.2% of the retained
   corpus was admitted by rounding.
2. **Truncation.** The classifier reads at most 510 content tokens. 58.2% of retained
   documents are truncated before scoring; token for token the classifier saw 41.9%
   of the retained text, a share that falls as web documents grow longer.
3. **Quantisation.** The classifier ran in bfloat16, giving 233 distinct score values.
   One grid point sits exactly on 2.5, where half-to-even rounding cuts 458,461
   documents.

## Reproduce

```
python download_random.py     # seeded manifest, then the 110 shards by exact path
python analyse_full.py         # sections 1, 2 and 3 (94-dump headline; FW_INCLUDE_ALL=1 for the all-110 diagnostic)
python marginal_test.py        # section 4, the boundary/band separability tests
python redteam_checks.py       # the fixed-score truncation trend to 2025 and the six-point AUC control
```

`random_shard_manifest.json` pins the exact shards (seed 20260801, written before
download so the selection cannot have followed the results). `by_dump.csv` is the
per-snapshot output behind the trend figures.

The quantisation result needs none of the downloads. Read `src/run_edu_bert.py` in
`HuggingFaceFW/fineweb-edu-classifier`, then run one line of NumPy against any
FineWeb-Edu shard:

```python
(np.round(np.clip(df.score, 0, 5)).astype(int) == df.int_score).all()        # True
(np.floor(np.clip(df.score, 0, 5) + 0.5).astype(int) == df.int_score).all()  # False
```

## Sample

One random shard from each of the 110 Common Crawl dumps in
`HuggingFaceFW/fineweb-edu-score-2`. The 16 dumps covering 2024 and 2025 partition
documents across shards by score, so a single shard from them is not a random sample;
they are excluded from the headline, leaving 94 dumps (2013 to 2023). Including them
moves the headline by about 0.1 percentage points.

## Conflict of interest

This analysis was carried out with the assistance of Claude, a model made by Anthropic,
which competes with the organisations that produce and publish these corpora. See the
disclosure note in the article.

## Licence

CC BY 4.0. See `LICENSE`.
