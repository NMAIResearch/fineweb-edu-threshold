#!/usr/bin/env python3
"""Next action 2: kill the fixed-shard caveat.

One SEEDED RANDOM shard from each of the 110 CC dumps in fineweb-edu-score-2.
~265 GB. The manifest is written to disk first so the sample is reproducible and
so a reader can check we did not pick shards after seeing results.

Two traps this avoids, both recorded in HANDOVER.md section 6c:
  - filename conventions differ between dumps (train-NNNNN-of-NNNNN vs NNN_NNNNN),
    so this downloads EXACT paths from the repo listing, never an --include glob
    that can match nothing and still exit 0;
  - this machine's wifi drops roughly hourly, so every file gets its own retry
    loop and the script is safe to re-run: completed files are skipped.

Run:  nohup .venv/bin/python download_random.py > download_random.log 2>&1 &
Re-run after any interruption. It resumes.
"""
import json, os, sys, time
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPO = "HuggingFaceFW/fineweb-edu-score-2"
SEED = 20260801
DEST = os.path.expanduser("~/fw-score2-random")
MANIFEST = "random_shard_manifest.json"
MAX_TRIES = 200          # wifi drops hourly; give up only on something real
INDEX = "repo_shard_index.json"

# ---- manifest: seeded, written once, never regenerated -----------------------
if os.path.exists(MANIFEST):
    manifest = json.load(open(MANIFEST))
    print(f"reusing existing {MANIFEST} ({len(manifest)} dumps) — sample is fixed")
else:
    index = json.load(open(INDEX))
    rng = np.random.default_rng(SEED)
    manifest = {}
    for dump in sorted(index):
        shards = sorted(index[dump])
        pick = shards[int(rng.integers(len(shards)))]
        manifest[dump] = {"shard": pick,
                          "path": f"data/{dump}/{pick}.parquet",
                          "n_shards": len(shards)}
    json.dump(manifest, open(MANIFEST, "w"), indent=1)
    print(f"wrote {MANIFEST}: 1 random shard from each of {len(manifest)} dumps, seed {SEED}")

os.makedirs(DEST, exist_ok=True)
todo = sorted(manifest)
print(f"target {DEST}\n{len(todo)} files to fetch\n", flush=True)

done = failed = 0
t0 = time.time()
for k, dump in enumerate(todo, 1):
    path = manifest[dump]["path"]
    local = os.path.join(DEST, path)
    if os.path.exists(local) and os.path.getsize(local) > 1_000_000:
        print(f"[{k:3}/{len(todo)}] {dump:<18} skip, have {os.path.getsize(local)/2**30:.2f} GB", flush=True)
        done += 1
        continue

    for attempt in range(1, MAX_TRIES + 1):
        try:
            p = hf_hub_download(REPO, path, repo_type="dataset", local_dir=DEST)
            sz = os.path.getsize(p) / 2**30
            el = (time.time() - t0) / 60
            print(f"[{k:3}/{len(todo)}] {dump:<18} OK  {sz:5.2f} GB  "
                  f"(attempt {attempt}, {el:.0f} min elapsed)", flush=True)
            done += 1
            break
        except KeyboardInterrupt:
            sys.exit(1)
        except Exception as e:
            if isinstance(e, HfHubHTTPError) and getattr(e.response, "status_code", 0) == 404:
                print(f"[{k:3}/{len(todo)}] {dump:<18} FATAL 404 {path}", flush=True)
                failed += 1
                break
            wait = min(60, 2 ** min(attempt, 6))
            print(f"[{k:3}/{len(todo)}] {dump:<18} retry {attempt} in {wait}s: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(wait)
    else:
        print(f"[{k:3}/{len(todo)}] {dump:<18} GAVE UP after {MAX_TRIES}", flush=True)
        failed += 1

print(f"\ncomplete: {done}/{len(todo)} downloaded, {failed} failed, "
      f"{(time.time()-t0)/3600:.1f} h")
print("VERIFY BY FILE COUNT, NOT EXIT STATUS:")
print(f"  find {DEST} -name '*.parquet' | wc -l   # expect {len(todo)}")
