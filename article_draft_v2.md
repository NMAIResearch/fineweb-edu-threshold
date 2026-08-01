# What actually admits a document to FineWeb-Edu

**Three operational details in a published quality filter, measured across 84 million documents and 94 crawls**
NM AI Research · 1 August 2026 · [doi.org/10.5281/zenodo.21740082](https://doi.org/10.5281/zenodo.21740082)

FineWeb-Edu is a filtered web corpus, built by scoring pages with a small classifier for
"educational value" and keeping the ones that score highly enough. The documented rule is a
threshold of 3 on a scale of 0 to 5. Both the corpus and the classifier are public, and the
corpus stores each document's score alongside its text, which makes the filter unusually easy
to check.

This piece checks it. Nothing below says the filter is wrong, and none of the mechanisms
described is a defect. Each is a reasonable engineering decision. The point is that a phrase as
simple as "a threshold of 3 on a scale of 0 to 5" turns out to describe a selection process whose
operational details are large, measurable, and as far as I can find unmeasured.

**In short.** The threshold is applied to a rounded integer, so admission begins at a raw score of
2.5, and 57.2% of the retained documents sit in the half-point band below the nominal figure. The
classifier reads at most 510 tokens, so it never saw 58.2% of the documents it admitted in full,
and token for token it saw 41.9% of the text it was judging. That share is falling: truncation has
risen steadily from 2013 to 2025. The classifier ran in bfloat16, which quantises its output onto
a grid of 233 distinct values, and one grid point lands exactly on 2.5, where the rounding rule
sends 458,461 documents down rather than up.

---

## Why this corpus

FineWeb-Edu is not the largest corpus, nor necessarily the best. It was chosen because it is
the only widely used one that publishes its filter's per-document score inline. Each row carries
`score`, `int_score`, `language_score` and `token_count` next to the text.

That matters more than it may sound. RedPajama, C4, the Pile and Dolma all publish the documents
that survived filtering, but not the number that admitted each one. Without the score you can
inspect a filter's output, but you cannot audit its decision. Publishing the score is what makes
this analysis possible at all, and it is to the corpus authors' credit that it is there. Other
corpora function here as a control rather than as alternatives: a property that also shows up in
C4 and RedPajama would be a property of web text, not of this filter.

## The filter, as documented

The classifier is `HuggingFaceFW/fineweb-edu-classifier`: a BERT-family model
(`Snowflake/snowflake-arctic-embed-m` with a single regression head) trained on 450,000 samples
annotated by Llama-3-70B-Instruct for educational quality on a 0 to 5 scale. Its model card
reports an F1 of 82%, and recommends `int_score >= 3` for curation.

Two things about that 82% are worth holding onto. It is a binary metric, computed by
rounding the regression output to an integer and
splitting at 3, the same `int_score >= 3` rule that builds the corpus, so it measures whether the
model sorts documents above or below the line, not whether it recovers a score. And the same model card publishes a per-class
report in which class 3, the decision band itself, scores a precision of 0.56, a recall of 0.50
and an F1 of 0.53. Overall accuracy on the held-out set is 0.71. The model is at its weakest
exactly where the threshold falls, which is a reasonable thing for a filter to be, and a useful
thing to know before reading anything below.

The scoring code is published too, in the classifier repository as `src/run_edu_bert.py`. Three
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
admission boundary is a raw score of 2.5, not 3.

It is worth being clear that this is the normal thing to do, and that there is a reading on which
it is simply correct. The classifier is an ordinal regression trained on integer labels, so class
3 occupies the interval from 2.5 to 3.5, and rounding a continuous prediction to the nearest
integer is the standard way to assign it to a class. On that reading 2.5 is not a leak. It is the
definition of the lower edge of class 3, and the classifier's own card uses exactly that
formulation: `we recommend using int_score >= 3 as a threshold for data curation`.

The two dataset cards, which are what someone using the corpus actually reads, do not use that
formulation. They describe "setting a threshold of 3 (on a scale of 0 to 5)", "using a score of 3
as a threshold for keeping and removing files", and "a less strict threshold of 2 instead of 3".
That is the language of a cutoff on a continuous scale. So the gap here is not between the
documentation and the code. It is between what the phrase "a threshold of 3 out of 5" leads a
reader to picture and what the pipeline delivers, and the size of that gap does not appear to have
been measured. Across the sample of 84,005,795 documents:

| Band | Documents | Share | Tokens | Share |
|---|---|---|---|---|
| Cut (`int_score` below 3) | 65,908,378 | 78.5% | 58,910,929,618 | 76.0% |
| Kept only by rounding (score above 2.5, below 3.0) | 10,352,466 | 12.3% | 10,431,542,863 | 13.4% |
| Above the nominal threshold (score 3.0 or more) | 7,744,951 | 9.2% | 8,222,346,448 | 10.6% |

Of the documents the filter retained, 57.2% scored below 3, and they supply 55.9% of the retained
tokens. A majority of a corpus described as filtered at 3 out of 5 is made of documents the model
scored between 2.5 and 3.

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

| Population | Truncated | Share of tokens the classifier saw |
|---|---|---|
| All documents | 54.1% | 45.5% |
| Retained documents | 58.2% | 41.9% |
| Kept only by rounding | 56.8% | 42.5% |
| Above the nominal threshold | 60.0% | 41.1% |

Token for token, the classifier saw 41.9% of the text in the corpus it was admitting. More
than half the words in FineWeb-Edu were never read by the filter that selected them.

Two details are worth drawing out. The first is that truncation is highest in the top band, so the
documents the filter rates most highly are the ones it has seen the smallest proportion of. The
tempting explanation is that the score is partly a proxy for document length, but that turns out
to be wrong: across all 84 million documents the correlation between score and length is
negligible (Pearson r of +0.04, Spearman +0.07). What is true is narrower. Median length rises
gently across the score bands, from 544 tokens at the bottom to 661 at the top, and because the
510-token budget sits right in the middle of that range, a small shift in the median moves a lot
of documents across it.

The second is that the exposure is growing, and this holds under a stricter test than a simple
year-on-year average. Because document length itself changes over time, the fairer comparison
fixes the score and asks whether truncation still rises. Among documents scoring exactly 2.0, a
single point on the grid, truncation runs from 52.7% in 2013 to 62.3% in 2025, with a Pearson r
against year of +0.84 across 107 crawls. Mean median length for that same fixed score rises from
530 tokens to 632.

Web documents are getting longer while the window stays at 512. The filter's blind spot widens
every year, and a threshold calibrated on 2013 text is not doing the same job on 2025 text.

## 3. bfloat16 puts a grid point exactly on the tie

The scoring script loads the model in bfloat16. That is a sensible choice for a large batch job
and it has a consequence that survives into the published data.

bfloat16 carries eight bits of mantissa, so the model's outputs land on a coarse grid. In the
region around the threshold the spacing is 0.015625. This is visible in the corpus: 100.0000%
of the 84 million scores are exactly representable in bfloat16, and there are only 233 distinct
score values in the entire sample. A quality score presented as continuous is operationally a
233-level ordinal.

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
grid, and every one of them is cut. That is 458,461 documents, 2.5% of everything that
round-half-up would have retained, removed by the parity of an integer.

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
and the rounding rule puts them on opposite sides of the corpus. If the boundary is meaningful,
some difference should be detectable.

The method is deliberately plain: TF-IDF over word unigrams, logistic regression, five-fold
cross-validation, 15,000 documents per group, reporting area under the ROC curve. An AUC of 0.5
means the two groups cannot be told apart.

| Comparison | AUC |
|---|---|
| Boundary: score 2.500000 (cut) against 2.515625 (kept) | 0.505 |
| The same pair with the labels shuffled | 0.498 |
| Document length alone, on the same pair | 0.503 |
| Band: kept-only-by-rounding against above the nominal threshold | 0.767 |
| Control: score 2.0 or below against score 4.0 or above | 0.997 |

The control is the important row. The same features and the same model separate clearly poor from
clearly strong documents at an AUC of 0.997, so the method can see educational quality when it is
present. On the two sides of the actual admission boundary it reaches 0.505, against a
label-shuffled null of 0.498. Length alone does no better, at 0.503.

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

The boundary is not special. Every pair sits at chance, so the correct conclusion is the weaker
one: the score carries no locally detectable signal anywhere on its range. Read alongside the
finding that it takes only 233 distinct values, that says the score's effective resolution is much
coarser than its apparent precision, and a single grid step is not a meaningful quantity of
educational quality. What the test does establish about the boundary specifically is narrow but
worth having: the documents sitting on the tie point are ordinary. They are not some pathological
class of empty or boilerplate pages that the rule is quietly catching.

The row that carries real weight is the fourth: the band kept only by rounding is distinguishable
from the band above the nominal threshold, at an AUC of 0.767.
No single marginal decision can be justified on the content, but admitting a 0.5-wide band of
scores brings in around ten million documents that are recognisably different in character from
the ones that cleared 3. Where the band edge sits is not a detail, even though which side of it
any individual document falls is close to a coin toss.

A small detail from the same sample: the median document at the boundary runs to 535 tokens in
the classifier's own tokenizer, against a budget of 510, and 52.3% of documents at the tie point
are truncated. The median marginal document is itself cut short, so the score that decided its
fate was formed without the end of it.

## What this does and does not show

**It does not show that FineWeb-Edu is a poor corpus.** Nothing here tests downstream model
quality, and the corpus has demonstrably worked well for the people using it. Every mechanism
described above is a reasonable engineering decision taken in isolation. Rounding a regression
output to an integer is normal. Truncating at the model's context length is unavoidable without
retraining. bfloat16 is the sensible dtype for a job of this size.

**What it shows is how much a one-line description leaves out.** "Keep documents scoring 3 or more
out of 5" is a phrase most readers would take to describe a corpus in which the 9.2% of candidates
scoring 3 or better survive. The pipeline it describes retains 21.5%, judges each document on its
first 510 tokens, which for most documents is not the whole document, and resolves the exact
boundary case by the parity of an integer.

Filter thresholds are governance decisions. They determine what a model is trained on, and there
is good evidence that these choices affect what the resulting models can do. That makes the
distance between a stated threshold and its operational meaning worth measuring rather than
assuming, and it is measurable here only because the scores were published. The general lesson is uncomfortable for
the many pipelines that publish neither: a filter that does not publish its scores cannot be
audited, only its output inspected. [1]

Two of the three mechanisms generalise beyond this corpus. Any pipeline that rounds a
low-precision score at an integer threshold has a tie cohort, and I am not aware of one that
reports it. Any classifier-based filter applied to documents longer than its context window is
judging a prefix, and the share affected will rise as long as web documents keep getting longer.

## Method

**Sample.** One shard chosen at random from each of the 110 Common Crawl dumps in
`HuggingFaceFW/fineweb-edu-score-2`, using a fixed seed, with the manifest written to disk before
downloading so the selection cannot have followed the results. 233 GB.

**Exclusion.** 16 of the 110 dumps, covering 2024 and 2025, were dropped. Those dumps partition
documents across shards by score, so a single shard from them is a score-stratified slice rather
than a sample: some shards are 100% `int_score` 2, others are 86% `int_score` 3 and contain no 2s
at all. Scores are shuffled within a shard, so this cannot be corrected by sampling rows
differently inside a file. The remaining 94 dumps, covering 2013 to 2023, are shuffled and are
used throughout. Including the excluded dumps would have moved the headline from 57.2% to 57.1%
and truncation from 54.1% to 55.3%, so the numbers barely change, but they would have been
reached by a route that does not hold up.

**Scale.** 84,005,795 documents and 77,564,818,929 tokens for the score-based results, which use
every row. Truncation uses a two-stage sample of 188,000 documents (random row groups, then random
rows within them), re-tokenised with the classifier's own tokenizer. The boundary tests use 15,000
documents per group at 2.5, and 8,000 per group for the six-point comparison.

**The trend to 2025.** The 16 excluded dumps can still be used for one purpose. They are
partitioned by score, so a whole-dump rate cannot be recovered from a single shard, but
conditioning on one exact score value removes that bias: among documents scoring precisely 2.0, a
score-partitioned shard is still a fair sample of documents at that score. The truncation series
in section 2 is measured that way across 107 dumps, which both extends it to 2025 and controls for
the corpus getting easier or harder to score over time. On the 94 unpartitioned dumps alone the
same series gives a Pearson r of +0.79.

**Uncertainty.** The truncation headline of 54.1% carries a binomial 95% interval of ±0.23
percentage points, which should be read as a floor because the sample is clustered. AUCs are means
across five folds with the standard deviation shown in the underlying output.

**Caveat.** `fineweb-edu-score-2` retains `int_score >= 2` only, so nothing here speaks to
documents scoring below a raw 1.5. The 975,821 documents at exactly 1.5 are therefore a floor,
not a full count.

## Reproduce

```
download_random.py     seeded manifest, then the 110 shards by exact path
analyse_full.py        sections 1, 2 and 3
marginal_test.py       section 4
```

The rounding and quantisation results in section 3 need none of this. Read
`src/run_edu_bert.py` in the classifier repository, then run one line of NumPy against any
FineWeb-Edu shard:

```python
(np.round(np.clip(df.score, 0, 5)).astype(int) == df.int_score).all()   # True
(np.floor(np.clip(df.score, 0, 5) + 0.5).astype(int) == df.int_score).all()   # False
```

---

*Verification: the classifier's architecture, dtype, rounding expression, tokenizer limit and
training details were read from the primary artefacts in `HuggingFaceFW/fineweb-edu-classifier`
(`src/run_edu_bert.py`, `config.json`, `tokenizer_config.json`, `README.md`, accessed 1 August
2026), not from secondary description. That the reported 82% F1 is computed on the rounded integer
prediction, the same discretisation that defines `int_score`, rather than on the raw score, was
read from the training script `train_edu_bert.py` in `huggingface/cosmopedia`, whose
`compute_metrics` rounds both predictions and labels before scoring, accessed the same day. The
quoted threshold wording is taken verbatim from the
dataset cards of `HuggingFaceFW/fineweb-edu` and `HuggingFaceFW/fineweb-edu-score-2` and from the
classifier card, all accessed the same day. The F1 of 82%, the per-class report and the 450,000
annotation count are the model card's own published figures and have not been independently
recomputed. All document counts, score distributions, truncation shares and AUCs were computed
from `HuggingFaceFW/fineweb-edu-score-2` by the scripts named above. An earlier draft asserted that
the score partly proxies for document length; that was tested against all 84 million documents,
found to be wrong (Pearson r of +0.04), and has been withdrawn. No downstream model was trained or
evaluated, so no claim is made here about the effect of any of this on model quality.*

*Conflict of interest: this analysis was carried out with the assistance of Claude, a model made
by Anthropic. Anthropic competes with the organisations that produce and publish the corpora
discussed here, and has an interest in how the auditability of open training data is characterised.
The line marked [1] above states a general conclusion about open data practice from which that
position benefits, and should be read with that in mind.*
