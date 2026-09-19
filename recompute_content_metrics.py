"""Recompute content-only summaries without altering frozen intermediates.

Writes content_metrics.json and corrected_by_dump.csv. The fixed-score extension
and AUC experiments are outside this command's scope.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from token_lengths import content_lengths, CONTENT_BUDGET


def main():
    sample_path = Path("full_tokenised_sample.parquet")
    sm = pd.read_parquet(sample_path)
    sm["content_tokens"] = content_lengths(sm.bert_tokens)
    masks = {"all": np.ones(len(sm), dtype=bool), "retained": sm.int_score >= 3,
             "rounded_in": (sm.int_score >= 3) & (sm.score < 3),
             "above": sm.score >= 3, "tie": sm.score == 2.5}
    result = {"purpose": "Authoritative content-only metrics derived from unchanged frozen encoded lengths",
              "input_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
              "counting": "bert_tokens includes CLS and SEP; subtract two before applying the content budget",
              "content_budget": CONTENT_BUDGET, "populations": {}}
    for name, mask in masks.items():
        d = sm[mask]
        lengths = d.content_tokens
        visible = np.minimum(lengths, CONTENT_BUDGET)
        result["populations"][name] = {
            "n": len(d), "truncated": int((lengths > CONTENT_BUDGET).sum()),
            "old_truncated": int((d.bert_tokens > CONTENT_BUDGET).sum()),
            "content_tokens": int(lengths.sum()), "visible_content_tokens": int(visible.sum()),
            "truncated_pct": float((lengths > CONTENT_BUDGET).mean() * 100),
            "visible_pct": float(visible.sum() / lengths.sum() * 100),
            "median_content_tokens": float(lengths.median())}
    by = sm.assign(tr=sm.content_tokens > CONTENT_BUDGET).groupby("dump", observed=True).agg(
        n_tok=("tr", "size"), truncated=("tr", "sum"), trunc_pct=("tr", lambda x: x.mean() * 100))
    years = by.index.str.extract(r"CC-MAIN-(\d{4})")[0].astype(int).values
    result["trend"] = {"n_dumps": len(by), "first_pct": float(by.trunc_pct.iloc[0]),
                       "last_pct": float(by.trunc_pct.iloc[-1]),
                       "pearson_year": float(np.corrcoef(years, by.trunc_pct)[0, 1])}
    result["content_gpt2_median_ratio"] = float((sm.content_tokens / sm.gpt2_tokens).median())
    Path("content_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    by.to_csv("corrected_by_dump.csv")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
