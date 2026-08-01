#!/usr/bin/env python3
"""Regenerate the data blocks embedded in index.html, the threshold explorer.

The page carries its own data so it works offline and cannot drift from the record.
Three small tables go in, between the markers in the <script> block:

  GRID    score_grid.csv        the 233-point bfloat16 grid with document and token
                                counts and the shipped int_score for each value
  DUMPS   by_dump.csv           per-snapshot rounded-in and truncation shares, 94 dumps
  FIXED   redteam_check_a.csv   truncation among documents scoring exactly 2.0, 107 dumps

score_grid.csv is itself derived from full_scores.parquet (written by analyse_full.py).
It is regenerated here when the parquet is present, and read from disk otherwise, so a
reader without the 254 MB file can still rebuild the page.

Run:  .venv/bin/python build_tool.py
"""
import os
import re
import sys

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
d = pd.read_csv(p("by_dump.csv"))
dumps = "dump,docs,rounded_in_pct,trunc_pct\n" + "\n".join(
    f"{r.dump},{r.docs},{r.rounded_in_pct:.3f},{r.trunc_pct:.2f}" for r in d.itertuples())

a = pd.read_csv(p("redteam_check_a.csv"))
fixed = "dump,year,trunc_pct,median_tok,partitioned\n" + "\n".join(
    f"{r.dump},{r.year},{r.trunc_pct:.4f},{r.median_tok:g},{r.partitioned}"
    for r in a.itertuples())

# ---- inject -------------------------------------------------------------------
html = open(p("index.html")).read()
blocks = {"GRID": ("GRID_CSV", grid), "DUMPS": ("DUMP_CSV", dumps), "FIXED": ("FIXED_CSV", fixed)}
for tag, (var, body) in blocks.items():
    pat = re.compile(rf"(/\* {tag}-START \*/\n).*?(\n/\* {tag}-END \*/)", re.S)
    assert pat.search(html), f"marker pair {tag} not found in index.html"
    html = pat.sub(lambda m: m.group(1) + f"const {var} = `{body}`;" + m.group(2), html)

open(p("index.html"), "w").write(html)
print(f"wrote index.html: {len(g)} grid points, {len(d)} sampled dumps, {len(a)} fixed-score dumps, "
      f"{len(html)/1024:.0f} KB")
