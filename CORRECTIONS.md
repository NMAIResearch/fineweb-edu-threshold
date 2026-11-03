# Content-token correction

Purpose: identify corrected measurements and the limits of this source and explorer update. The linked Zenodo record and `fineweb_edu_threshold.pdf` remain the historical release. This update does not replace their bytes or claim a new DOI version.

The stored `bert_tokens` lengths include CLS and SEP. Comparing those lengths directly with a content budget counted special tokens as source text. `token_lengths.py` subtracts those tokens; `recompute_content_metrics.py` creates the canonical metrics and per-snapshot series from the unchanged frozen intermediates. `sync_public_summaries.py` generates the main token-summary paragraph and dataset card. `build_tool.py` embeds the same metrics in the explorer.

`by_dump.csv` and `redteam_check_a.csv` preserve historical aggregates. The explorer takes truncation from `corrected_by_dump.csv`, retaining the score counts from `by_dump.csv`. It does not plot the historical fixed-score extension: those aggregates lack the individual lengths needed for an exact correction. The selected raw documents are required before that panel can return. The fixed-score extension used capped, first-eligible rows within shards, which also limits representativeness.

The manuscript distinguishes provider-reported metrics from reproduced calculations. The published training script reports multiclass macro-F1 and does not reproduce the model card's binary headline. Historical content-classifier AUCs remain attributed to stored outputs; they are not evidence of equivalence or educational quality. The earlier capped-pool boundary-tokenisation statement remains unverified. No downstream model was evaluated.

`verify_frozen.py` checks its named score and corrected token-summary quantities. It does not verify raw-shard sampling, the fixed-score extension, historical AUCs or every upstream claim. `test_explorer.py` checks embedded numerical parity and rejects the historical series as an active chart. These checks do not constitute independent acceptance.

OpenAI GPT-6 assisted this correction. OpenAI competes with model providers discussed in the research. The original analysis received assistance from Claude, made by Anthropic. Independent review and release approval remain separate from these local checks.

## Adjacent-score control

Found 3 November 2026. The fixed-score control in `redteam_checks.py` added 0.015625 at every point. At 1.75 that skips 1.7578125, which `score_grid.csv` shows is an occupied score, so that pair spans two bfloat16 steps, not one; the other five are adjacent. `fineweb_edu_threshold.pdf` labels all six "one bfloat16 step apart" and remains the historical release. The script now selects the next bfloat16 value and `article_draft_v2.md`, the unreleased candidate, labels the first pair as two steps. The corrected control has not been rerun, so the table keeps the historical AUCs.

Found by GPT-6.1 Sol (OpenAI) in a review of this repository and checked against `score_grid.csv` by Claude Opus 5.5 (Anthropic). OpenAI and Anthropic compete with model providers discussed in the research.
