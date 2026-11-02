Purpose: unreleased correction candidate. The score-boundary results and content-only summaries have local verification. The fixed-score extension and historical AUCs have not been rerun, and independent acceptance is pending. The byline below identifies the historical edition. OpenAI GPT-6 assisted this correction candidate; this file does not replace the published edition.

# What actually admits a document to FineWeb-Edu

**Three operational details in a published quality filter, measured across 84 million documents and 94 crawls**
NMAI Research · 1 August 2026 · [doi.org/10.5281/zenodo.21740082](https://doi.org/10.5281/zenodo.21740082)

FineWeb-Edu is a filtered web corpus, built by scoring pages with a small classifier for
"educational value" and keeping the ones that score highly enough. The documented rule is a
threshold of 3 on a scale of 0 to 5. Both the corpus and the classifier are public, and the
corpus stores each document's score alongside its text, which makes the filter unusually easy
to check.

This piece checks it. Nothing below says the filter is wrong, and none of the mechanisms
described is a defect. Each is a reasonable engineering decision. The point is that a phrase as
simple as "a threshold of 3 on a scale of 0 to 5" turns out to describe a selection process whose
operational details can be measured. Their treatment in prior research remains unverified; this audit makes no claim of priority.

Under the recorded integer-rounding rule, admission at int_score >= 3 requires a raw score strictly above 2.5; an exact score of 2.5 rounds down. In the 84,005,795-document sample, 10,352,466 of 18,097,417 retained documents scored below 3, or 57.2%, N = 18,097,417 retained documents. The sample contains 458,461 documents exactly at 2.5 and 233 distinct observed score values. These are sample measurements, not a census of the full dataset.

Among the 40,431 retained documents in the 188,000-document tokenisation sample, 23,445 exceeded the 510-token content budget, or 58.0%, N = 40,431 retained documents. The measured prefixes contain 17,087,287 of 40,802,906 content tokens, or 41.9%, across those same documents. The CLS and SEP special tokens are excluded from both token counts. This is a token-weighted sample estimate of prefix exposure, not a word count or a full-corpus census.

The corrected tokenisation sample also permits a descriptive comparison across the included crawls. The separate fixed-score extension remains unverified after the counting correction. Neither comparison establishes a downstream effect on model quality.

---

## Why this corpus

FineWeb-Edu was selected because its published per-document scores permit a direct check of this admission rule. This does not establish that it is the only auditable corpus. RedPajama-V2 also publishes per-document quality signals, including classifier predictions, as its [dataset card](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) documents.

Publishing scores and the associated transformation makes this analysis possible. A comprehensive comparison with C4, the Pile, Dolma or RedPajama was not performed, and none serves as an empirical control in this study.

## The filter, as documented

The classifier is `HuggingFaceFW/fineweb-edu-classifier`: a BERT-family model
(`Snowflake/snowflake-arctic-embed-m` with a single regression head) trained on 450,000 samples
annotated by Llama-3-70B-Instruct for educational quality on a 0 to 5 scale. Its model card
reports an F1 of 82%, and recommends `int_score >= 3` for curation.

The reported F1 is a binary classification result. Its calculation and underlying confusion counts are not independently established here. The published [training script](https://github.com/huggingface/cosmopedia/blob/main/classification/train_edu_bert.py) rounds predictions and reports multiclass macro-F1; it does not by itself reproduce the binary headline.

The model card separately reports class-3 precision of 0.56, recall of 0.50 and F1 of 0.53, with 3,407 class-3 examples in a held-out set of 46,867. Those are provider-reported metrics, not recomputed results. Classes 4 and 5 have lower reported F1 scores, so class 3 is not the weakest class. The distinction between agreement with Llama-generated labels and independent educational-quality validation remains material.

The scoring code is published too, in the upstream [Cosmopedia inference script](https://github.com/huggingface/cosmopedia/blob/main/classification/run_edu_bert.py). Three
lines from it carry the rest of this piece:

```python
model = AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                           torch_dtype=torch.bfloat16)
inputs = tokenizer(batch[args.text_column], return_tensors="pt",
                   padding="longest", truncation=True).to(device)
batch["int_score"] = [int(round(max(0, min(score, 5)))) for score in logits]
```

## 1. The threshold is applied to a rounded integer

The recommendation is `int_score >= 3`, and `int_score` is the raw regression output rounded to
an integer. A document scoring 2.6 therefore has an `int_score` of 3 and is kept. The effective
admission boundary is a raw score strictly greater than 2.5; an exact score of 2.5 is excluded.

It is worth being clear that this is the normal thing to do, and that there is a reading on which
it is simply correct. The classifier is an ordinal regression trained on integer labels, so class
3 occupies the open interval above 2.5 and below 3.5 under half-to-even rounding, and rounding a continuous prediction to the nearest
integer is the standard way to assign it to a class. On that reading 2.5 is not a leak. It is the
definition of the lower edge of class 3, and the classifier's own card uses exactly that
formulation: `we recommend using int_score >= 3 as a threshold for data curation`.

The two dataset cards, which are what someone using the corpus actually reads, do not use that
formulation. They describe "setting a threshold of 3 (on a scale of 0 to 5)", "using a score of 3
as a threshold for keeping and removing files", and "a less strict threshold of 2 instead of 3".
That is the language of a cutoff on a continuous scale. So the gap here is not between the
documentation and the code. It is between what the phrase "a threshold of 3 out of 5" leads a
reader to picture and what the pipeline delivers, and this audit measures that difference in the selected sample. Whether earlier work measured it remains unverified. Across the sample of 84,005,795 documents:

| Band | Documents | Share | Tokens | Share |
|---|---|---|---|---|
| Cut (`int_score` below 3) | 65,908,378 | 78.5% | 58,910,929,618 | 76.0% |
| Kept only by rounding (score above 2.5, below 3.0) | 10,352,466 | 12.3% | 10,431,542,863 | 13.4% |
| Above the nominal threshold (score 3.0 or more) | 7,744,951 | 9.2% | 8,222,346,448 | 10.6% |

Within the sampled rows, 10,352,466 of 18,097,417 retained documents scored below 3, or 57.2%, N = 18,097,417 retained documents. They supply 10,431,542,863 of 18,653,889,311 retained GPT-2 tokens, or 55.9%, across those retained documents. These are measurements of the selected sample.

The share is stable. Across the 94 crawls it moves only between 11.7% and 13.3% of all documents,
drifting gently upward with the crawl year (Pearson r of +0.77 against year).

## 2. The classifier reads at most 510 tokens

`run_edu_bert.py` calls the tokenizer with `truncation=True` and no explicit `max_length`, so the
limit falls back to `model_max_length`, which the tokenizer configuration sets to 512. The
model's `config.json` sets `max_position_embeddings` to 512 as well, so this is architectural
rather than a tuning choice. Since 512 includes the `[CLS]` and `[SEP]` markers, the real budget
for document text is 510 tokens.

Measuring this properly requires the classifier's own tokenizer. The `token_count` stored in the
corpus is a GPT-2 count, which is a different tokenizer and gives a different answer. Re-tokenised
with the classifier's own WordPiece vocabulary, on a sample of 188,000 documents:

| Population | Truncated documents / N | Truncated share | Visible / total content tokens | Visible share |
|---|---:|---:|---:|---:|
| All documents | 101,357 / 188,000 | 53.9% | 77,482,545 / 170,394,619 | 45.5% |
| Retained documents | 23,445 / 40,431 | 58.0% | 17,087,287 / 40,802,906 | 41.9% |
| Kept only by rounding | 13,064 / 23,081 | 56.6% | 9,649,650 / 22,711,107 | 42.5% |
| Above the nominal threshold | 10,381 / 17,350 | 59.8% | 7,437,637 / 18,091,799 | 41.1% |

Among the 40,431 retained documents in the 188,000-document tokenisation sample, 23,445 exceeded the 510-token content budget, or 58.0%, N = 40,431 retained documents. The measured prefixes contain 17,087,287 of 40,802,906 content tokens, or 41.9%, across those same documents. The CLS and SEP special tokens are excluded from both token counts. This is a token-weighted sample estimate of prefix exposure, not a word count or a full-corpus census.

Two details are worth drawing out. The first is that truncation is highest in the top band, so the
documents the filter rates most highly are the ones it has seen the smallest proportion of. The
tempting explanation is that the score is partly a proxy for document length, but the retained correlation results do not establish that explanation: across all 84 million documents the correlation between score and length is
negligible (Pearson r of +0.04, Spearman +0.07). These small correlations do not exclude nonlinear associations or confounding. The observed description is narrower. Median length rises
gently across the score bands, from 544 tokens at the bottom to 661 at the top, and because the
510-token budget sits right in the middle of that range, a small shift in the median moves a lot
of documents across it.

The second question is whether truncation exposure differs across crawl dates. In the corrected sample, the fraction is 980 of 2,000 documents in the first included crawl and 1,157 of 2,000 in the last. This is a descriptive sample comparison. It is not a population-weighted trend or an estimate of a causal effect of time.

The earlier fixed-score extension used the same special-token counting error. Its per-dump aggregates do not retain the lengths needed to recompute the corrected truncation rates. Its reported truncation percentages and correlations are therefore unverified pending replay from the selected documents. Conditioning on a fixed score would still leave the capped-pool sampling limitation described below.

## 3. bfloat16 puts a grid point exactly on the tie

The scoring script loads the model in bfloat16. That is a sensible choice for a large batch job
and it has a consequence that survives into the published data.

bfloat16 has eight bits of significand precision, including the implicit leading bit, so the model's outputs land on a coarse grid. In the
region around the threshold the spacing is 0.015625. This is visible in the corpus: 100.0000%
of the 84 million scores are exactly representable in bfloat16, and there are only 233 distinct
score values in the entire sample. A quality score presented as continuous is operationally a
score taking 233 distinct observed values in this sample. This is not a limit on the values bfloat16 can represent.

One of those grid points falls exactly on 2.5, and Python's `round` is half-to-even. A tie is
resolved towards whichever neighbouring integer is even, so 2.5 becomes 2, and the document is
cut. Half-to-even reproduces the published `int_score` for 100.0000% of the 84 million rows;
ordinary round-half-up reproduces 99.4494% of them.

| Score | Documents |
|---|---|
| 2.484375 | 467,968 |
| 2.500000 | 458,461 |
| 2.515625 | 449,176 |

The tie point is not a spike. It holds an unremarkable number of documents for its position on the
grid, and every one of them is cut. Under half-up rounding, 18,555,878 sampled documents would be retained; the 458,461-document tie cohort is 2.5% of that counterfactual set, N = 18,555,878. This is an admission-rule comparison, not a quality judgement.

Two objections are worth meeting head on. Any threshold on a discrete scale has to assign exact
ties to one side, and bfloat16 is an unremarkable dtype for a job of this size. Neither is a
defect. What is worth reporting is that the two combine. Under a continuous score the set of
documents landing exactly on a tie point would be vanishingly small, and the choice of tie rule
would be immaterial. Quantisation turns that near-empty set into 458,461 documents.

The direction of the tie also depends on the parity of the threshold, which is visible because
HuggingFace published two corpora at two thresholds. FineWeb-Edu keeps `int_score >= 3`. Three is
odd, so the tie at 2.5 rounds to the even neighbour 2, away from admission, and those documents
are cut. FineWeb-Edu-score-2 keeps `int_score >= 2`. Two is even, so the tie at 1.5 rounds to 2,
toward admission, and the 975,821 documents sitting exactly on 1.5 are kept. The same rule is
conservative at one threshold and permissive at the other, and nothing about educational quality
decides which.

## 4. Are the two sides of the boundary actually different?

The mechanisms above describe how the line is drawn. The remaining question is whether the line
separates anything.

The test compares documents scoring exactly 2.500000, which are cut, with documents scoring
exactly 2.515625, which are kept. These are adjacent points on the bfloat16 grid, one step apart,
and the rounding rule puts them on opposite sides of the corpus. This probes differences detectable by one representation and classifier; a negative result cannot decide whether the boundary is meaningful.

The method is deliberately plain: TF-IDF over word unigrams, logistic regression, five-fold
cross-validation, 15,000 documents per group, reporting area under the ROC curve. An AUC of 0.5
is the chance-discrimination baseline for this test; a value near it does not establish that the groups are equivalent.

| Comparison | AUC |
|---|---|
| Boundary: score 2.500000 (cut) against 2.515625 (kept) | 0.505 |
| The same pair with the labels shuffled | 0.498 |
| Document length alone, on the same pair | 0.503 |
| Band: kept-only-by-rounding against above the nominal threshold | 0.767 |
| Control: score 2.0 or below against score 4.0 or above | 0.997 |

The TF-IDF and logistic-regression test distinguished the widely separated score groups in the stored experiment. Those groups were labelled using the original classifier's scores, so this control does not independently validate educational quality. The adjacent-score comparisons produced AUCs near the chance baseline under this method. This limits what the tested representation detected; it does not establish that the documents are equivalent, that no other method could distinguish them, or that an individual admission decision is unjustified.

The stored boundary result is 0.5054 against a shuffled-label result of 0.4982, a difference of 0.0072, with 15,000 documents per group, N = 30,000. This difference is descriptive; no significance or equivalence finding is established here. The stored length-only result is 0.503.

It would be easy to over-read that. One objection to that reading is that for any
continuous score thresholded anywhere, two points one step apart will look alike, so an AUC near
0.5 at the boundary may say nothing about this filter in particular. That is correct, and it is
testable, so here is the same comparison run at six points on the grid:

| Pair, one bfloat16 step apart | AUC |
|---|---|
| 1.750000 against 1.765625 | 0.4996 |
| 2.000000 against 2.015625 | 0.4911 |
| 2.250000 against 2.265625 | 0.4994 |
| 2.500000 against 2.515625, the admission boundary | 0.5068 |
| 2.750000 against 2.765625 | 0.5011 |
| 3.000000 against 3.015625 | 0.5022 |

The six tested adjacent-score pairs gave AUCs near the chance baseline with the specified features and model. The admission-boundary pair was not distinctive within that comparison. This does not establish the absence of signal elsewhere on the score range, the educational equivalence of the groups or the absence of particular document types at the tie point. The result remains conditional on the tested method and sampling procedure.

The row that carries real weight is the fourth: the band kept only by rounding is distinguishable
from the band above the nominal threshold, at an AUC of 0.767.
That comparison shows that the sampled score bands are distinguishable under the tested representation. It does not establish their relative educational quality or justify a conclusion about every individual decision. The broader band changes membership by 10,352,466 documents within the measured 84,005,795-document sample, while the content comparison has the narrower capped-pool scope described below.

The historical boundary-tokenisation statement cannot be reproduced from a separately retained capped-pool tokenisation output and remains unverified. It is distinct from the frozen tokenisation sample used above. Neither the TF-IDF AUC nor a GPT-2 length summary establishes how much of those documents the classifier read.

## What this does and does not show

**It does not show that FineWeb-Edu is a poor corpus.** Nothing here tests downstream model quality. Downstream effectiveness is outside this audit. Every mechanism
described above is a reasonable engineering decision taken in isolation. Rounding a regression
output to an integer is normal. This implementation truncates at the context limit. Chunking or another document-processing policy would require separate evaluation. bfloat16 is the sensible dtype for a job of this size.

**What it shows is how much a one-line description leaves out.** "Keep documents scoring 3 or more
out of 5" is a phrase most readers would take to describe a corpus in which the 9.2% of candidates
scoring 3 or better survive. The pipeline it describes retains 21.5%, judges each document on its
first 510 tokens, which for most documents is not the whole document, and resolves the exact
boundary case by the parity of an integer.

Filter thresholds are governance decisions. They determine what a model is trained on, and there
is good evidence that these choices affect what the resulting models can do. That makes the
distance between a stated threshold and its operational meaning worth measuring rather than
assuming, and it is measurable here only because the scores were published. The general lesson is uncomfortable for
the many pipelines that publish neither: withholding per-document scores prevents this direct reconstruction from those published rows alone; it does not rule out every other form of audit. [1]

Related mechanisms can occur elsewhere, but their prevalence is unverified. Rounding a low-precision score creates a tie cohort only where a boundary is representable and occupied. A classifier that truncates long documents can omit content; its affected share depends on the length distribution, tokenizer and processing policy. This study does not establish that the share increases in every corpus or over every period.

## Method

**Sample.** One shard chosen at random from each of the 110 Common Crawl dumps in
`HuggingFaceFW/fineweb-edu-score-2`, using a fixed seed, with the manifest recording the selected shard identities. The downloader writes a new manifest before fetching, but code order alone does not establish the historical selection chronology. 233 GB.

**Exclusion.** 16 of the 110 dumps, covering 2024 and 2025, were dropped. Those dumps partition
documents across shards by score, so a single shard from them is a score-stratified slice rather
than a sample: some shards are 100% `int_score` 2, others are 86% `int_score` 3 and contain no 2s
at all. Scores are shuffled within a shard, so this cannot be corrected by sampling rows
differently inside a file. The remaining 94 dumps, covering 2013 to 2023, are shuffled and are
used throughout. Including the excluded dumps would have moved the headline from 57.2% to 57.1%
in the historical score calculation. The older truncation comparison counted special tokens inconsistently and is not retained as a corrected estimate. The exclusion still does not establish uniform document-level sampling across the full corpus.

**Scale.** 84,005,795 documents and 77,564,818,929 tokens for the score-based results, which use
every row. Truncation uses a two-stage sample of 188,000 documents (random row groups, then random
rows within them), re-tokenised with the classifier's own tokenizer. The boundary tests use 15,000
documents per group at 2.5, and 8,000 per group for the six-point comparison.

The content-separability tests use capped pools filled from the first eligible rows encountered in the selected shards. The stored main run filled all groups after four shards, and some groups filled in the first shard. These pools are not a random sample of all eligible documents across the crawl series. Random cross-validation within each pool does not remove selection, duplication or shared-host dependence. The AUC results are conditional on this collection procedure.

**The trend to 2025.** The historical fixed-score extension is unverified after the special-token correction and requires replay. Holding score fixed controls observed score composition, not sampling within that score. The implementation takes the first eligible rows up to a cap from each selected shard; it does not establish a representative within-score sample.

**Uncertainty.** The corrected all-document truncation count is 101,357 of 188,000, or 53.9%. A binomial calculation treating documents as independent gives an approximate 95% interval of plus or minus 0.23 percentage points. Clustered sampling violates that independence assumption, and this is not a validated population confidence interval or a guaranteed lower bound on its width. AUCs are means
across five folds with the standard deviation shown in the underlying output.

**Caveat.** `fineweb-edu-score-2` retains `int_score >= 2` only, so this sample does not describe documents scoring below a raw 1.5. The count of documents exactly at 1.5 describes the selected shards, not the entire corpus.

## Reproduce

verify_frozen.py checks the named score-distribution, rounding, tie-count and corrected content-only tokenisation-summary figures against the frozen intermediate files. It does not verify the original shard-to-intermediate transformation, the upstream classifier documentation, the content-classification AUCs or the separate fixed-score trend extension. download_random.py retrieves the manifest-listed shards; analyse_full.py, marginal_test.py and redteam_checks.py perform the corresponding analyses. A complete reproduction must distinguish these steps.

```
download_random.py     seeded manifest, then the 110 shards by exact path
analyse_full.py        sections 1, 2 and 3
marginal_test.py       section 4
redteam_checks.py      fixed-score extension and adjacent-score controls, separate rerun required
```

The rounding and quantisation results in section 3 need none of this. Read
[`classification/run_edu_bert.py`](https://github.com/huggingface/cosmopedia/blob/main/classification/run_edu_bert.py) in Hugging Face’s Cosmopedia repository, then run one line of NumPy against any
FineWeb-Edu shard:

```python
(np.round(np.clip(df.score, 0, 5)).astype(int) == df.int_score).all()   # True
(np.floor(np.clip(df.score, 0, 5) + 0.5).astype(int) == df.int_score).all()   # False
```

---

*Verification: frozen score and tokenisation calculations were rerun for this correction candidate. The stored encoded lengths include CLS and SEP; content-only summaries subtract them before applying the content budget. The model card and published Cosmopedia training and inference scripts were checked as primary sources. The training script reports multiclass macro-F1 and does not independently reproduce the card's binary headline. Upstream metrics remain attributed reports. The fixed-score truncation extension requires replay; the AUCs are retained historical outputs, not newly reproduced results. No downstream model was trained or evaluated, and no independent acceptance is claimed.*

*Conflict of interest: this analysis was carried out with the assistance of Claude, a model made
by Anthropic. Anthropic competes with the organisations that produce and publish the corpora
discussed here, and has an interest in how the auditability of open training data is characterised.
The line marked [1] above states a general conclusion about open data practice from which that
position benefits, and should be read with that in mind.*

OpenAI GPT-6 assisted the correction review and calculations. OpenAI competes with model providers discussed in the analysis and has an interest in how open training-data auditability is described. This assistance is not independent acceptance of the correction.

## Conflict of interest and scope

The Verification note identifies the reported upstream source checks. The reproduction section separates retrieval, frozen-intermediate checks and the analyses required for the remaining outputs. Source relevance, semantic support and sample representativeness require assessment beyond a passing calculation script.

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
