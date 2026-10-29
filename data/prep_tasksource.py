# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///
"""Convert a tasksource-jev-typed-decisions parquet shard to the dm_train.py JSONL format.

Keeps commercial-use rows, drops sources that touch our eval tasks, caps rows per source, and writes a
random sample of N questions.
"""
import argparse
import json
import random

import polars as pl

EXCLUDE = ("github", "kubernetes", "issue")


def convert(row: dict) -> dict | None:
    kind, opts, target = row["kind"], list(row["options"] or []), list(row["target"] or [])
    if kind == "noul":
        if len(target) != 1:
            return None
        crit, tgt = {}, {"true": target[0], "false": 1 - target[0]}
    elif kind == "score":
        if len(opts) != len(target) or len(opts) < 2:
            return None
        crit, tgt = opts, {str(i): p for i, p in enumerate(target)}
    else:
        if len(opts) != len(target) or not 2 <= len(opts) <= 16 or len(set(opts)) != len(opts):
            return None
        crit, tgt = opts, dict(zip(opts, target))
    return {"id": row["id"], "state": row["state"], "source": row["source"],
            "question": {"type": kind, "instructions": row["question"], "criteria": crit}, "target": tgt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--per-source-cap", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = pl.read_parquet(args.parquet).filter(pl.col("license_use") == "commercial")
    df = df.filter(~pl.col("source").str.to_lowercase().str.contains("|".join(EXCLUDE)))
    df = df.filter(pl.col("state").str.len_chars() <= args.max_chars)
    rows = df.sample(fraction=1.0, shuffle=True, seed=args.seed).to_dicts()
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    per, out = {}, []
    for r in rows:
        if per.get(r["source"], 0) >= args.per_source_cap:
            continue
        c = convert(r)
        if c is None:
            continue
        per[r["source"]] = per.get(r["source"], 0) + 1
        out.append(c)
        if len(out) >= args.n:
            break
    with open(args.out, "w") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    kinds = {}
    for c in out:
        kinds[c["question"]["type"]] = kinds.get(c["question"]["type"], 0) + 1
    print(f"wrote {len(out)} from {len(per)} sources; kinds {kinds}")


if __name__ == "__main__":
    main()
