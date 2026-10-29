# /// script
# requires-python = ">=3.10"
# dependencies = ["gliclass>=0.1.21", "flashdeberta", "transformers>=5", "torch", "numpy"]
# ///
"""Edge zero-shot on our curation eval files (decision rows), with the same inputs our models get: text = state,
prompt = instructions, labels = option descriptions (choice) or ["yes", "no"] (noul). Single-label softmax over
the label logits, as in the 10-23 bench. Writes predictions in the trainer's format ({id, ids, logits}) so
cur_score.py scores Edge and our runs the same way.
usage: python cur_edge_eval.py --evals /b/data/stage4/prompt/eval_pc_*.jsonl --out /b/runs/edge-cureval/predictions
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("USE_FLASHDEBERTA", "1")

import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline

MODEL_ID = "knowledgator/gliclass-v4-edge"


def as_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def render_state(state):
    """structured states rendered as the trainer renders them (gliclass_ub_train.render_state), 2026-10-27"""
    if state in (None, "", {}, []):
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "\n".join(f"- {as_text(item)}" for item in state)
    lines = []
    for key, value in state.items():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            lines += [f"{key}:"] + [f"- {item}" for item in value]
        else:
            lines.append(f"{key}: {as_text(value)}")
    return "\n".join(lines)
MAX_LEN = 1024
BATCH = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=MAX_LEN, help="token limit (1024 = the 10-23 equal-input runs)")
    args = ap.parse_args()
    files = sorted({f for pat in args.evals for f in glob.glob(pat)})
    model = (
        GLiClassModel.from_pretrained(MODEL_ID, dtype=torch.float16)
        .to("cuda", dtype=torch.float16)
        .eval()
    )
    pipe = ZeroShotClassificationPipeline(
        model,
        MODEL_ID,
        max_length=args.max_len,
        classification_type="single-label",
        device="cuda:0",
        progress_bar=False,
    ).pipe
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"model": MODEL_ID, "max_len": args.max_len, "sets": {}}
    for path in files:
        name = "cur_" + Path(path).stem.removeprefix("eval_")
        rows = [json.loads(x) for x in open(path)]
        groups = defaultdict(list)  # same label set + prompt -> batch together
        for r in rows:
            q = r["question"]
            if q["type"] == "noul":
                keys, labels = ["true", "false"], ["yes", "no"]
            else:
                keys = list(q["criteria"])
                labels = [
                    str(v) if v is not None else k for k, v in q["criteria"].items()
                ]
            groups[(tuple(keys), tuple(labels), q.get("instructions") or "")].append(r)
        n_trunc = n_cut = 0
        with open(out / f"{name}__edge.jsonl", "w") as f:
            for (keys, labels, prompt), rs in groups.items():
                for i in range(0, len(rs), BATCH):
                    chunk = rs[i : i + BATCH]
                    strings = [
                        pipe.prepare_input(render_state(r["state"]), list(labels), None, prompt)
                        for r in chunk
                    ]
                    enc = pipe.tokenizer(
                        strings,
                        truncation=True,
                        max_length=args.max_len,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")
                    n_trunc += int((enc["attention_mask"].sum(-1) >= args.max_len).sum())
                    n_cls = (enc["input_ids"] == model.config.class_token_index).sum(-1)
                    n_cut += int((n_cls < len(labels)).sum())
                    with torch.inference_mode():
                        logits = (
                            model(**enc, max_num_classes=len(labels))
                            .logits[:, : len(labels)]
                            .float()
                            .cpu()
                            .numpy()
                        )
                    for r, lg in zip(chunk, logits):
                        f.write(
                            json.dumps(
                                {
                                    "id": r["id"],
                                    "ids": list(keys),
                                    "logits": [float(x) for x in lg],
                                }
                            )
                            + "\n"
                        )
        stats["sets"][name] = {"rows": len(rows), "inputs_at_max_len": n_trunc, "labels_cut": n_cut}
        print(f"{name}: {len(rows)} rows; at max_len {n_trunc}; labels cut {n_cut}", flush=True)
    (out / "edge_eval_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
