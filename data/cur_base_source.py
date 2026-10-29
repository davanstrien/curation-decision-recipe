# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "polars", "optuna"]
# ///
"""Register the base component of the curation mix as frozen sources.

base-r1g: the combined arm's control mix, built once with a fixed seed instead of inside every job: r1g pool ->
  Edge-scored subset (s3_edge_subset.py rule) -> C1 drop list -> s2_sweep.build_mix (w_v2 0.5, w_ts 0.4, w_logic 0.1,
  no tasksource cap, max 4 repeats, MIX_ROWS rows, seed). Gold targets (teacher_blend 1.0, as arm_run.sh).
lanes: the combined arm's gated lanes for the main recipe, copied unchanged: procpile, tool, fh_arxiv, nls.
  JADE and chess stay out of the main recipe.
Each source dir gets SOURCE.json (origin, build command, licence class, teacher, row count, file sha256).
usage (CPU Job, dm-recipe mounted at /b, dm-curation at /c, scripts at /s):
  python /s/cur_base_source.py --out /c/sources --version v1 [--mix-rows 200000 --seed 0]
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_sweep import build_mix, index_pool  # noqa: E402

B = Path("/b")
LANES = {
    "procpile": ("data/stage4/g/procpile_gl.jsonl", "program-computed answers (procedural pile)"),
    "tool": ("data/stage4/tool/tool_gl.jsonl", "tool oracle (program-computed)"),
    "fh-arxiv": ("data/stage4/fh/fh_arxiv_gl.jsonl", "arXiv author categories (source metadata)"),
    "nls": ("data/stage4/cat/nls_gl.jsonl", "NLS catalogue subject headings (human)"),
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def write_source(d, files, meta):
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["files"] = {}
    for f in files:
        n = sum(1 for _ in open(d / f))
        meta["files"][f] = {"rows": n, "sha256": sha_file(d / f)}
    meta["built"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["command"] = sys.argv
    (d / "SOURCE.json").write_text(json.dumps(meta, indent=1))
    print(d, json.dumps(meta["files"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--mix-rows", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default="/tmp/base")
    args = ap.parse_args()
    out, work = Path(args.out), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    pool = B / "data/stage2/r1/pool_train_r1g.jsonl"
    scores = B / "data/stage2/scores_r0.parquet"
    drop_file = B / "data/stage4/c1/c1_drop_sources.txt"
    scored = set(pl.read_parquet(scores, columns=["id", "edge_logits"]).drop_nulls()["id"].to_list())
    drop = {x.strip() for x in open(drop_file) if x.strip()}
    sub = work / "pool_edge_c1.jsonl"
    n = kept = 0
    with open(pool) as fin, open(sub, "w") as fout:
        for line in fin:
            n += 1
            r = json.loads(line)
            if r["id"] in scored and r["src"] not in drop:
                fout.write(line)
                kept += 1
    print(f"pool {n} rows -> Edge-scored, C1-dropped {kept}", flush=True)
    params = {"w_v2": 0.5, "w_ts": 0.4, "w_logic": 0.1, "ts_cap": "none"}
    d = out / "base-r1g" / args.version
    d.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    stats = build_mix(str(sub), index_pool(str(sub)), params, args.mix_rows, 4, args.seed, str(d / "base.jsonl"))
    write_source(d, ["base.jsonl"], {
        "origin": "r1g pool (tasksource + knowledgator gliclass-v2.0 + v3-logic), Edge-scored subset, C1 drop list",
        "inputs": {str(pool): sha_file(pool), str(scores): sha_file(scores), str(drop_file): sha_file(drop_file)},
        "params": params, "mix_rows": args.mix_rows, "seed": args.seed, "mix_stats": stats,
        "licence_class": "permissive",
        "licence": "per-source; NC excluded by the phase-1 source screen (verify per source before release)",
        "teacher": "source labels (tasksource gold; gliclass-v2.0 / v3-logic: knowledgator, LLM-generated)",
    })
    for name, (rel, teacher) in LANES.items():
        ld = out / name / args.version
        ld.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(B / rel, ld / Path(rel).name)
        write_source(ld, [Path(rel).name], {
            "origin": f"combined-arm lane, copied from hf://buckets/YOUR_NAME/dm-recipe/{rel}",
            "licence_class": "permissive", "licence": "see reports/I (lane report)", "teacher": teacher,
        })


if __name__ == "__main__":
    main()
