# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.0",
#   "transformers>=5.20",
#   "flash-linear-attention==0.5.2",
#   "huggingface_hub>=1.21",
# ]
# ///
"""Serving throughput on real k8s test documents, one harness for every student.

All models: bf16, length-sorted batches under a padded-token budget, real inputs (no fixed length), one forward pass
per item, timing includes tokenisation. LLMs read the last-position logits (letter readout); encoders run the
sequence-classification forward at max_length 2048 (as in the accuracy runs). Base weights only: a merged LoRA and
the letter-row head do not change the cost.
"""
import argparse
import json
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Prompt rendering copied from dm_train.py (single-file Jobs script).
SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
          "Respond with only its uppercase letter, with no explanation or reasoning.")
LETTERS = "ABCDEFGHIJKLMNOP"


def option_list(question):
    qtype, criteria = question["type"], question["criteria"]
    if qtype == "noul":
        criteria = criteria if isinstance(criteria, dict) else {}
        return [(key, criteria.get(key) or f"The proposition is {key}.") for key in ("true", "false")]
    if qtype == "choice" and isinstance(criteria, dict):
        return [(key, text or key) for key, text in criteria.items()]
    if qtype == "choice":
        return [(str(key), str(key)) for key in criteria]
    return [(str(i), text) for i, text in enumerate(criteria)]


def render(item):
    opts = option_list(item["question"])
    state = item["state"] if isinstance(item["state"], str) else json.dumps(item["state"], ensure_ascii=False)
    payload = {"evidence": state, "criterion": item["question"]["instructions"],
               "options": [{"letter": LETTERS[i], "description": f"{key}: {text}"} for i, (key, text) in enumerate(opts)]}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    return msgs, [key for key, _ in opts]

LLMS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"]
ENCODERS = ["dmis-lab/UltraBERT-150M", "dmis-lab/UltraBERT-380M"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def batches(seqs, token_budget):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out, cur = [], []
    for i in order:
        longest = max([len(seqs[j]) for j in cur] + [len(seqs[i])])
        if cur and longest * (len(cur) + 1) > token_budget:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def pad(seqs, idx, pad_id, device):
    longest = max(len(seqs[i]) for i in idx)
    ids = torch.full((len(idx), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(idx), longest), dtype=torch.long)
    for row, i in enumerate(idx):
        ids[row, : len(seqs[i])] = torch.tensor(seqs[i])
        mask[row, : len(seqs[i])] = 1
    return ids.to(device), mask.to(device)


def tokenize_llm(tok, items):
    seqs = []
    for it in items:
        msgs, _ = render(it)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        seqs.append(tok.encode(prompt, add_special_tokens=False))
    return seqs


def tokenize_encoder(tok, items, max_length):
    return [tok(it["state"], truncation=True, max_length=max_length)["input_ids"] for it in items]


@torch.no_grad()
def run_llm(model, seqs, token_budget, pad_id):
    for idx in batches(seqs, token_budget):
        ids, mask = pad(seqs, idx, pad_id, "cuda")
        hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        last = mask.sum(1) - 1
        model.lm_head(hidden[torch.arange(len(idx), device="cuda"), last])
    torch.cuda.synchronize()


@torch.no_grad()
def run_encoder(model, seqs, token_budget, pad_id):
    for idx in batches(seqs, token_budget):
        ids, mask = pad(seqs, idx, pad_id, "cuda")
        _ = model(input_ids=ids, attention_mask=mask).logits
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="/b/data/prepared/k8s_test.jsonl")
    ap.add_argument("--only", choices=["all", "llm", "enc"], default="all")
    ap.add_argument("--repeat", type=int, default=4, help="test set copies per timed pass")
    ap.add_argument("--out", default="/tmp/serve-bench/bench.jsonl")
    args = ap.parse_args()
    items = load_jsonl(args.test) * args.repeat
    hw = torch.cuda.get_device_name(0)
    rows = []
    names = {"all": LLMS + ENCODERS, "llm": LLMS, "enc": ENCODERS}[args.only]
    for name in names:
        is_llm = name in LLMS
        tok = AutoTokenizer.from_pretrained(name)
        if is_llm:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
            budgets = [32768, 65536]
        else:
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=8, dtype=torch.bfloat16).cuda().eval()
            budgets = [16384, 65536]
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        for budget in budgets:
            t0 = time.perf_counter()
            seqs = tokenize_llm(tok, items) if is_llm else tokenize_encoder(tok, items, 2048)
            t_tok = time.perf_counter() - t0
            runner = run_llm if is_llm else run_encoder
            runner(model, seqs[:256], budget, pad_id)  # warm-up
            t1 = time.perf_counter()
            runner(model, seqs, budget, pad_id)
            t_fwd = time.perf_counter() - t1
            n = len(seqs)
            row = {"model": name, "hardware": hw, "token_budget": budget, "docs": n,
                   "mean_tokens": sum(map(len, seqs)) / n, "tokenize_s": round(t_tok, 2), "forward_s": round(t_fwd, 2),
                   "docs_per_s": round(n / (t_tok + t_fwd), 1), "docs_per_s_forward_only": round(n / t_fwd, 1),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            print(json.dumps(row), flush=True)
            rows.append(row)
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    from huggingface_hub import sync_bucket
    sync_bucket(os.path.dirname(args.out), f"hf://buckets/YOUR_NAME/dm-recipe/runs/serve-bench-{args.only}")


if __name__ == "__main__":
    main()
