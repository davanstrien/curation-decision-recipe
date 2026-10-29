# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "huggingface_hub>=1.32"]
# ///
"""Convert the released dataset (davanstrien/curation-decision-mix) to the trainer's JSONL input.

The release stores nested fields (state when it is a record, question, target, true_labels) as JSON strings and adds
provenance columns. The trainer (train/gliclass_ub_train.py) reads one JSON object per line, in one of two shapes:
  decision rows  {id, src, state, question, target, multi_label?}
  GLiClass rows  {id, src, text, all_labels, true_labels, prompt, multi_label?}

`web` rows are skipped unless their `state` has been rebuilt (the release carries labels only; see the dataset card).

    uv run data/release_to_jsonl.py --out mix.jsonl                          # all released configs
    uv run data/release_to_jsonl.py --out mix.jsonl --licence-class permissive --configs procpile nls
"""

import argparse
import json
from collections import Counter

import polars as pl
from huggingface_hub import hf_hub_download

REPO = "davanstrien/curation-decision-mix"
CONFIGS = ["base", "decisionbank", "prompt", "web", "nls", "tool", "harvard", "arxiv", "procpile"]


def decode_state(state, record):
    """`record` rows had a dict or list state in the training mix (stored here as a JSON string); others had text."""
    return json.loads(state) if record else state


def to_trainer_row(r):
    """One release row as a trainer row, or None when it cannot be trained on (a web row without its text)."""
    row = {"id": r["id"], "src": r["src"]}
    if r["row_type"] == "decision":
        if not r["state"]:
            return None
        row["state"] = decode_state(r["state"], r["state_is_record"])
        row["question"] = json.loads(r["question"])
        row["target"] = json.loads(r["target"])
    else:
        row["text"] = r["text"]
        row["all_labels"] = list(r["all_labels"])
        row["true_labels"] = json.loads(r["true_labels"])
        row["prompt"] = r["prompt"] or ""
    if r["multi_label"]:
        row["multi_label"] = True
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="JSONL file to write")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=None, help="dataset revision (branch, tag or commit; default: main)")
    ap.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    ap.add_argument("--licence-class", nargs="+", default=None,
                    help="keep only these licence classes, e.g. permissive sa (default: all)")
    args = ap.parse_args()

    counts = Counter()
    with open(args.out, "w") as f:
        for config in args.configs:
            path = hf_hub_download(args.repo, f"data/{config}/train.parquet", repo_type="dataset",
                                   revision=args.revision)
            df = pl.read_parquet(path)
            if args.licence_class:
                df = df.filter(pl.col("licence_class").is_in(args.licence_class))
            for r in df.iter_rows(named=True):
                row = to_trainer_row(r)
                if row is None:
                    counts[f"{config}: skipped (no text)"] += 1
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[config] += 1
    for k, v in counts.items():
        print(f"{k}: {v:,}")
    print(f"wrote {sum(v for k, v in counts.items() if 'skipped' not in k):,} rows to {args.out}")


if __name__ == "__main__":
    main()
