#!/usr/bin/env python3
"""Regenerate the data blocks embedded in index.html, the threshold explorer.

The page carries its own corrected data and works offline.
Three small tables go in, between the markers in the <script> block:

  GRID    score_grid.csv        the 233-point bfloat16 grid with document and token
                                counts and the shipped int_score for each value
  DUMPS   corrected_by_dump.csv content-only truncation, joined to by_dump.csv scores
  METRICS content_metrics.json  corrected retained-document numerators and denominators

The historical fixed-score extension is not plotted until raw-text recalculation.

score_grid.csv is itself derived from full_scores.parquet (written by analyse_full.py).
It is regenerated here when the parquet is present, and read from disk otherwise, so a
reader without the 254 MB file can still rebuild the page.

Run:  .venv/bin/python build_tool.py
"""
import os
import re
import sys
import json
from pathlib import Path

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
p = lambda *a: os.path.join(HERE, *a)

# ---- the score grid: regenerate from the parquet if it is here ---------------
if os.path.exists(p("full_scores.parquet")):
    df = pd.read_parquet(p("full_scores.parquet"), columns=["score", "int_score", "token_count"])
    g = (df.groupby("score", observed=True)
           .agg(docs=("score", "size"), tokens=("token_count", "sum"),
                int_score=("int_score", "max"))
           .reset_index())
    assert (df.groupby("score", observed=True).int_score.nunique() == 1).all(), \
        "a score maps to more than one int_score; the grid is not a function"
    g.to_csv(p("score_grid.csv"), index=False)
    print(f"regenerated score_grid.csv from full_scores.parquet: {len(g)} grid points")
else:
    g = pd.read_csv(p("score_grid.csv"))
    if "int_scores" in g:                      # tolerate the earlier column name
        g = g.rename(columns={"int_scores": "int_score"})
    print(f"read score_grid.csv: {len(g)} grid points (parquet absent)")

# The page's arithmetic is only as good as these two identities.
assert len(g) == g.score.nunique(), "duplicate grid points"
assert (g.docs.sum(), g.tokens.sum()) == (84_005_795, 77_564_818_929), \
    f"grid totals moved: {g.docs.sum():,} docs, {g.tokens.sum():,} tokens"

# repr, not %g: every score is a dyadic rational and prints exactly, and rounding the
# grid to six significant figures in a page about rounding would be its own joke.
grid = "score,docs,tokens,int_score\n" + "\n".join(
    f"{float(r.score)!r},{r.docs},{r.tokens},{int(r.int_score)}" for r in g.itertuples())

# ---- per-snapshot series ------------------------------------------------------
d = pd.read_csv(p("by_dump.csv"))[["dump", "docs", "rounded_in_pct"]]
corrected = pd.read_csv(p("corrected_by_dump.csv"))
assert set(d.dump) == set(corrected.dump), "corrected snapshot coverage differs"
d = d.merge(corrected, on="dump", validate="one_to_one").sort_values("dump")
assert ((d.truncated / d.n_tok * 100 - d.trunc_pct).abs() < 1e-9).all()
dumps = "dump,docs,rounded_in_pct,n_tok,truncated,trunc_pct\n" + "\n".join(
    f"{r.dump},{r.docs},{r.rounded_in_pct:.3f},{r.n_tok},{r.truncated},{r.trunc_pct:.2f}"
    for r in d.itertuples())
metrics = json.loads(Path(p("content_metrics.json")).read_text())
assert metrics["counting"].startswith("bert_tokens includes CLS and SEP")
assert metrics["populations"]["all"]["n"] == int(d.n_tok.sum())
assert metrics["populations"]["all"]["truncated"] == int(d.truncated.sum())

# ---- inject -------------------------------------------------------------------
html = Path(p("index.html")).read_text()
blocks = {"GRID": ("GRID_CSV", grid), "DUMPS": ("DUMP_CSV", dumps)}
for tag, (var, body) in blocks.items():
    pat = re.compile(rf"(/\* {tag}-START \*/\n).*?(\n/\* {tag}-END \*/)", re.S)
    assert len(pat.findall(html)) == 1, f"unique marker pair {tag} not found in index.html"
    html = pat.sub(lambda m: m.group(1) + f"const {var} = `{body}`;" + m.group(2), html)

pat = re.compile(r"(/\* METRICS-START \*/\n).*?(\n/\* METRICS-END \*/)", re.S)
assert len(pat.findall(html)) == 1, "unique METRICS marker pair not found"
html = pat.sub(lambda m: m.group(1) + "const METRICS = " + json.dumps(metrics) + ";" + m.group(2), html)
Path(p("index.html")).write_text(html)
print(f"wrote index.html: {len(g)} grid points, {len(d)} corrected sampled dumps, "
      f"{len(html)/1024:.0f} KB")
