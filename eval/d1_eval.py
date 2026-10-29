# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers>=5.19", "torch", "torchvision", "pillow", "numpy"]
# ///
"""LiquidAI/d1-2B (or another d1 model) zero-shot on our curation eval files (decision rows), written in the
trainer's prediction format ({id, ids, logits}) so cur_score.py / cur_gate.py score it like our runs and Edge.

Mapping (2026-10-27): noul -> d1 noul (P(yes)); a choice whose keys are all level_<k> -> d1 score with the level
descriptions lowest first; any other choice -> d1 choice with criteria {key: description}. Logits = log of d1's
(calibrated) probabilities, clipped at 1e-6. The state is passed as-is (string or JSON value); d1 renders it.
usage: uv run scripts/d1_eval.py --evals /c/evals/v3/eval_*.jsonl --out /c/runs/d1-2b-v3/predictions
         [--model LiquidAI/d1-2B] [--chunk 64] [--max-rows N]
"""

import argparse
import json
import math
import re
from pathlib import Path

import torch
from transformers import AutoModel

LEVEL = re.compile(r"level_(\d+)")


def to_question(q):
    """-> (d1 question, our keys in output order)"""
    if q["type"] == "noul":
        return {"type": "noul", "instructions": q["instructions"]}, ["true", "false"]
    crit = q["criteria"]
    if isinstance(crit, list):
        crit = {c: c for c in crit}
    keys = list(crit)
    desc = {k: (v if v else k) for k, v in crit.items()}
    if len(keys) >= 2 and all(LEVEL.fullmatch(k) for k in keys):
        order = sorted(keys, key=lambda k: int(LEVEL.fullmatch(k).group(1)))
        return {"type": "score", "instructions": q["instructions"], "criteria": [desc[k] for k in order]}, order
    return {"type": "choice", "instructions": q["instructions"], "criteria": desc}, keys


def logits_from(answer, keys):
    def lg(p):
        return math.log(min(max(float(p), 1e-6), 1 - 1e-6))

    if answer["type"] == "noul":
        p = answer["noul"]
        return [lg(p), lg(1 - p)]
    probs = answer["probabilities"]
    if answer["type"] == "score":
        return [lg(probs[str(i)]) for i in range(len(keys))]
    return [lg(probs[k]) for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="LiquidAI/d1-2B")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    if args.chunk < 1:
        raise SystemExit("--chunk must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).to(device)
    stats = {"model": args.model, "sets": {}}
    for path in sorted(set(args.evals)):
        name = Path(path).stem.removeprefix("eval_")
        rows = [json.loads(x) for x in open(path) if x.strip()][: args.max_rows]
        n_err = 0
        with open(out / f"{name}__d1.jsonl", "w") as f:
            for i in range(0, len(rows), args.chunk):
                chunk = rows[i : i + args.chunk]
                qs = [to_question(r["question"]) for r in chunk]
                res = model.system_one_batch([(r["state"], {"q": q}) for r, (q, _) in zip(chunk, qs, strict=True)])
                for r, (_, keys), o in zip(chunk, qs, res, strict=True):  # d1 must answer every request
                    try:
                        lgts = logits_from(o["answers"]["q"], keys)
                    except (KeyError, TypeError) as e:  # never score a missing answer as a prediction (review)
                        raise SystemExit(f"{name}: no readable answer for {r['id']}: {e!r}") from e
                    f.write(json.dumps({"id": r["id"], "ids": keys, "logits": lgts}) + "\n")
        stats["sets"][name] = {"rows": len(rows), "readout_errors": n_err}
        print(f"{name}: {len(rows)} rows; readout errors {n_err}", flush=True)
    (out / "d1_eval_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
