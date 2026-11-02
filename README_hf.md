---
license: cc-by-4.0
language:
  - en
tags:
  - data-curation
  - reproducibility
  - fineweb
pretty_name: What Actually Admits a Document to FineWeb-Edu
---

Purpose: document the repository and explorer correction. Content-only token summaries have been recalculated. The historical fixed-score extension remains unresolved. OpenAI GPT-6 assisted these corrections.

# What actually admits a document to FineWeb-Edu

Three operational details in a published quality filter, measured across 84 million
documents and 94 Common Crawl snapshots.

**Paper (DOI):** [10.5281/zenodo.21740081](https://doi.org/10.5281/zenodo.21740081) · CC BY 4.0

**Interactive tool:** [Threshold Explorer](https://nmairesearch.github.io/fineweb-edu-threshold/)

The manuscript candidate is `article_draft_v2.md`. A corrected PDF has not been produced. This
bundle provides separate analysis scripts; the frozen intermediates are in [Zenodo record 21751942](https://zenodo.org/records/21751942) (version 1.1), and the files named below that this dataset does not hold are in the [GitHub repository](https://github.com/NMAIResearch/fineweb-edu-threshold). Their verification coverage is described below.

## The tool

The explorer uses the corrected content-only series. The fixed-score panel is unavailable pending recalculation; [`redteam_check_a.csv`](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/redteam_check_a.csv) is preserved as historical evidence. See [CORRECTIONS.md](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/CORRECTIONS.md).

[`index.html`](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/index.html) is a threshold explorer over the frozen data: move the raw admission cut
across the 233 observed score values in the sample, switch the rounding rule to watch the 458,461-document
tie cohort at 2.5 appear and disappear, and read the truncation trend across the crawl
series. It is one self-contained file with the data embedded, no dependencies and no
network calls, so it works from `file://` as well as from Pages. [`build_tool.py`](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/build_tool.py)
regenerates the score grid and embeds the corrected per-snapshot series and canonical content metrics:

```
python build_tool.py
```

## What is measured

Under the recorded integer-rounding rule, admission at int_score >= 3 requires a raw score strictly above 2.5; an exact score of 2.5 rounds down. In the 84,005,795-document sample, 10,352,466 of 18,097,417 retained documents scored below 3, or 57.2%, N = 18,097,417 retained documents. The sample contains 458,461 documents exactly at 2.5 and 233 distinct observed score values. These are sample measurements, not a census of the full dataset.

Among the 40,431 retained documents in the 188,000-document tokenisation sample, 23,445 exceeded the 510-token content budget, or 58.0%, N = 40,431 retained documents. The measured prefixes contain 17,087,287 of 40,802,906 content tokens, or 41.9%, across those same documents. The CLS and SEP special tokens are excluded from both token counts. This is a token-weighted sample estimate of prefix exposure, not a word count or a full-corpus census.

## Reproduce the bounded frozen checks

[verify_frozen.py](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/verify_frozen.py) checks the named score-distribution, rounding, tie-count and tokenisation-summary figures against the frozen intermediate files, `full_scores.parquet` and `full_tokenised_sample.parquet` from [Zenodo record 21751942](https://zenodo.org/records/21751942). It does not verify the original shard-to-intermediate transformation, the upstream classifier documentation, the content-classification AUCs or the separate fixed-score trend extension. download_random.py retrieves the manifest-listed shards; analyse_full.py, marginal_test.py and redteam_checks.py perform the corresponding analyses. A complete reproduction must distinguish these steps. Run the two commands below in a clone of the [GitHub repository](https://github.com/NMAIResearch/fineweb-edu-threshold), with the two intermediates beside them; [recompute_content_metrics.py](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/recompute_content_metrics.py) writes [`content_metrics.json`](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/content_metrics.json) and [`corrected_by_dump.csv`](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/corrected_by_dump.csv), which the repository also holds.

```
python recompute_content_metrics.py  # content_metrics.json and corrected_by_dump.csv
python verify_frozen.py             # named checks only; exits non-zero on mismatch
```

That takes about a minute and needs no other download. What it does not cover is the
read from the raw shards into those two files, the sampling and tokenisation step.
For that, the full path:

```
python download_random.py     # seeded manifest, then the 110 shards by exact path
python analyse_full.py         # sections 1, 2 and 3 (94-dump headline; FW_INCLUDE_ALL=1 for the all-110 diagnostic)
python marginal_test.py        # section 4, the boundary/band separability tests
python redteam_checks.py       # the fixed-score truncation trend to 2025 and the six-point AUC control
```

`random_shard_manifest.json` pins the exact shards (seed 20260801, written before
download so the selection cannot have followed the results). `by_dump.csv` preserves the historical aggregate output; [`corrected_by_dump.csv`](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/corrected_by_dump.csv) supplies current truncation figures. [`redteam_check_a.csv`](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/redteam_check_a.csv) is the uncorrected historical fixed-score extension, and [`score_grid.csv`](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/score_grid.csv) the whole 84-million-document score
distribution collapsed onto the 233 distinct values observed in this sample.

The quantisation result needs none of the downloads. Read [`classification/run_edu_bert.py`](https://github.com/huggingface/cosmopedia/blob/main/classification/run_edu_bert.py) in Hugging Face’s Cosmopedia repository, then run one line of NumPy against any
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

The original analysis was carried out with the assistance of Claude, a model made by Anthropic,
which competes with the organisations that produce and publish these corpora. See the
disclosure note in the article.

The article's Verification note identifies its reported upstream source checks. The reproduction instructions distinguish file retrieval, frozen-intermediate checks and the separate analyses. A passing calculation check does not establish semantic support for every statement.

What the author cannot guarantee: text drafted by a language model can carry
errors that are locally plausible, consistent in tone with the surrounding
prose, and therefore survive review. Several methods have been deployed to
mitigate this, including explicit instructions, internal red-teaming and
cross-lab blindspot checks, but the author does not claim the review is
exhaustive. Corrections are logged against the DOI when surfaced.

No warranty is offered beyond the terms of the CC BY 4.0 licence. Independent
analysis and open-science documentation only.

The research question, method, sourcing decisions and analytical judgements are
the author's. The assisting model is not always a neutral party to the subject
matter.

OpenAI GPT-6 assisted this correction. OpenAI competes with model providers discussed in the analysis. Verification scope and the historical publication boundary are stated in [CORRECTIONS.md](https://github.com/NMAIResearch/fineweb-edu-threshold/blob/main/CORRECTIONS.md).

## Licence

CC BY 4.0. See `LICENSE`.