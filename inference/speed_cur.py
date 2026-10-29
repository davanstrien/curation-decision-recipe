# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers>=5.19", "torch", "torchvision", "pillow", "numpy"]
# ///
"""Throughput of the release model, GLiClass Edge and d1-2B on the same curation eval rows, one GPU per run.

Rows: the first --max-rows rows of each eval file, length-sorted, timed end to end (tokenisation included) after a
warm-up pass on 64 rows. Our model and Edge use the trainer's eval path (gliclass_ub_train.predict: engine mapping,
bf16 autocast, 2,048 tokens); ours with flash-attn2, Edge with FlashDeBERTa (fp16), as in their accuracy runs.
d1 uses system_one_batch in chunks (d1_eval.py). The llm arm (vLLM, separate image) times one generated answer letter
per row; it measures speed only.
usage: python speed_cur.py --model ours|edge|d1|llm --evals /c/evals/v3/eval_fwe.jsonl ... --out /c/runs/speed-cur
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OURS = "/c/runs/cur5-cand-encsoup3-head/model"
EDGE = "knowledgator/gliclass-v4-edge"
D1 = "LiquidAI/d1-2B"


def load_rows(path, n):
    with open(path) as f:
        rows = [json.loads(x) for x in f if x.strip()]
    return rows[:n]


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def encoder_runner(which, batch_size):
    if which == "edge":
        os.environ["USE_FLASHDEBERTA"] = "1"
    sys.path.insert(0, str(ROOT))
    import gliclass_ub_train as T
    import torch
    from gliclass import GLiClassModel

    if which == "ours":
        model, tokenizer = T.load_model(OURS)
        model = model.to(T.DEVICE).eval()
        T.set_attention(model, "kernels-community/flash-attn2")
    else:
        model = (
            GLiClassModel.from_pretrained(EDGE, dtype=torch.float16)
            .to(T.DEVICE, dtype=torch.float16)
            .eval()
        )
        tokenizer = EDGE
    pipe = T.make_pipe(model, tokenizer, 2048)
    autocast = (
        which == "ours"
    )  # ours: fp32 weights + bf16 autocast (trainer eval); Edge: fp16 weights, no autocast

    @torch.inference_mode()
    def run(rows):
        inputs = [T.decision_inputs(r, "engine") for r in rows]
        inputs.sort(key=lambda x: len(x[1]))
        sync()
        t0 = time.perf_counter()
        for i in range(0, len(inputs), batch_size):
            chunk = inputs[i : i + batch_size]
            labels = [c[2] for c in chunk]
            tok = pipe.prepare_inputs(
                [c[1] for c in chunk],
                labels,
                same_labels=False,
                prompt=[c[3] for c in chunk],
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                logits = model(
                    **tok, max_num_classes=max(len(x) for x in labels)
                ).logits
            logits.float().cpu()
        sync()
        return time.perf_counter() - t0

    return run


def d1_runner(batch_size):
    import torch
    from d1_eval import to_question
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        D1, trust_remote_code=True, dtype=torch.bfloat16
    ).to("cuda")

    def run(rows):
        rows = sorted(rows, key=lambda r: len(state_text(r["state"])))
        sync()
        t0 = time.perf_counter()
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            res = model.system_one_batch(
                [(r["state"], {"q": to_question(r["question"])[0]}) for r in chunk]
            )
            assert len(res) == len(chunk)
        sync()
        return time.perf_counter() - t0

    return run


def llm_prompt(row):
    q = row["question"]
    if q["type"] == "noul":
        opts = {"A": "yes", "B": "no"}
    else:
        crit = q["criteria"]
        vals = list(crit.values()) if isinstance(crit, dict) else list(crit)
        opts = {chr(65 + i): (v or k) for i, (k, v) in enumerate(zip(list(crit), vals))}
    lines = "\n".join(f"{k}. {v}" for k, v in opts.items())
    return (
        f"{state_text(row['state'])}\n\n{q.get('instructions') or ''}\nOptions:\n{lines}\n"
        "Answer with the letter of one option only."
    )


def llm_runner(model_id, max_len):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, max_model_len=max_len + 512, gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    params = SamplingParams(max_tokens=2, temperature=0)

    def run(rows):
        prompts = []
        for r in rows:
            ids = tok(llm_prompt(r))["input_ids"][
                -max_len:
            ]  # keep the question and options at the end
            msg = [{"role": "user", "content": tok.decode(ids)}]
            prompts.append(
                tok.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        t0 = time.perf_counter()
        llm.generate(prompts, params, use_tqdm=False)
        return time.perf_counter() - t0

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ours", "edge", "d1", "llm"], required=True)
    ap.add_argument("--llm", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--max-rows", type=int, default=1000)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = {
        Path(p).stem.removeprefix("eval_"): load_rows(p, args.max_rows)
        for p in args.evals
    }
    warm = next(iter(sets.values()))[:64]
    result = {"model": args.model, "max_rows": args.max_rows, "runs": {}}
    if args.model == "llm":
        result["llm"] = args.llm
        batch_sizes = [0]  # vLLM schedules its own batches
    else:
        batch_sizes = args.batch_sizes
    for bs in batch_sizes:
        if args.model in ("ours", "edge"):
            run = encoder_runner(args.model, bs)
        elif args.model == "d1":
            run = d1_runner(bs)
        else:
            run = llm_runner(args.llm, 2048)
        run(warm)
        per_set, total_rows, total_s = {}, 0, 0.0
        for name, rows in sets.items():
            secs = run(rows)
            per_set[name] = {
                "rows": len(rows),
                "seconds": secs,
                "rows_per_s": len(rows) / secs,
            }
            total_rows += len(rows)
            total_s += secs
            print(args.model, bs, name, json.dumps(per_set[name]), flush=True)
        result["runs"][str(bs)] = {
            "sets": per_set,
            "rows": total_rows,
            "seconds": total_s,
            "rows_per_s": total_rows / total_s,
        }
        print(args.model, bs, "TOTAL", total_rows / total_s, flush=True)
        if args.model == "llm":
            break
    (out / f"{args.model}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
