# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.11.0", "numpy>=2", "gliclass==0.1.21", "transformers==5.20.2", "kernels==0.17.2",
#                 "huggingface_hub>=1.32", "polars", "trackio==0.43.0"]
#
# # Defaults for `hf jobs uv run`; command-line flags override them.
# [tool.hf-jobs]
# flavor = "rtx-pro-6000"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///
"""Train a GLiClass uni-encoder on an UltraBERT backbone (gliclass 0.1.21 model classes and data processing).

Recipe (knowledgator/gliclass-v4-edge config, runs/gliclass_v4_recipe_research.md):
  sequence  <bos><<LABEL>>l1<<LABEL>>l2...<<SEP>>{prompt}{text}   (labels shuffled per example, prompt_first)
  model     pooling first (the <bos> token), class_token_pooling first, embed_class_token, MLP scorer
            (2h -> 1024 -> 512 -> 1), dropout 0.3
  loss      multi_label_classification, alpha-weighted BCE (focal alpha 0.75, gamma -1) over the valid label slots;
            single-label rows are one-hot, soft targets ({label: score}) allowed
Changes against gliclass train.py / Trainer:
  - own training loop: any exception (OOM included) stops the run; a step with loss exactly 0 or non-finite fails
    (gliclass Trainer.training_step swallows exceptions and returns loss 0)
  - problem_type multi_label_classification only (single-label CE is broken for encoder models in 0.1.21)
  - UltraBERT attention from --attn (default kernels-community/flash-attn2), bf16 autocast, fp32 master weights,
    loss in fp32, optional gradient checkpointing
  - no option-length cap; --max-length 4096 by default (up to 32768); token-budget batches of similar length
  - <<LABEL>>/<<SEP>>/<<EXAMPLE>> appended after the 49,152-row vocab (ids 49152-49154), seeded from words;
    the tokenizer adds <bos>=0 in front; pad = 0 = <bos>, masked by attention_mask
The saved model loads with gliclass GLiClassModel.from_pretrained + ZeroShotClassificationPipeline (so with
di-stage0/engines/gliclass_engine.py). A plain reload uses sdpa; set the attention after loading (--eval-attn here).

Input rows: gliclass JSON/JSONL {text, all_labels, true_labels (list or {label: score}), prompt?} or decision JSONL
{id, state, question{type, instructions, criteria}, target}, converted with --mapping:
  engine   the gliclass_engine.py (card jev_api.py) mapping: render_state(state) as text, instructions as prompt,
           one label per option description; noul -> [criteria.true or "yes", criteria.false or "no"].
           Extension: score questions (list criteria) -> one label per criterion (the engine refuses them).
  enceval  the enc_eval.py mapping (encmp-zs-gliclass-* runs): evidence_text(state), labels "key: description",
           noul default "The proposition is true/false."
Eval (--eval name=path[@N]): decision JSONL through the pipeline's prepare_inputs, softmax over the label logits
(raw logits; --readout calibrated uses calibrated logits when the model has a calibrator). Writes
predictions/<name>__desc.jsonl ({id, ids, qtype, logits, gold}) for cur_score.py.

usage (HF Jobs):
  hf jobs run --flavor rtx-pro-6000 --secrets HF_TOKEN -v hf://buckets/YOUR_NAME/dm-recipe:/b -v scripts:/s \\
    hf.co/spaces/YOUR_NAME/ub-train-image -- python /s/gliclass_ub_train.py --run-name X \\
    --train /b/data/gliclass/v2_6k.jsonl:5000 --eval k8s=/b/data/prepared/k8s_test.jsonl@200 --max-steps 200
  convert only: python gliclass_ub_train.py --convert in.jsonl out.jsonl [--mapping engine]
  eval only an existing model: --base-model knowledgator/gliclass-v4-edge --max-steps 0 --eval ...
"""

import argparse
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPECIAL_TOKENS = ["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"]
SPECIAL_TOKEN_INIT_WORDS = {
    "<<LABEL>>": "label",
    "<<SEP>>": "text",
    "<<EXAMPLE>>": "example",
}


def load_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        if path.endswith(".json"):
            rows = json.load(f)
            return rows[:limit] if limit else rows
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def split_spec(spec):
    """path[:N] -> (path, N or None)"""
    path, _, n = (
        spec.rpartition(":") if spec.rpartition(":")[2].isdigit() else (spec, "", "")
    )
    return path, int(n) if n else None


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------ trackio (never fatal)
TRACKIO = None


def tracker_init(args):
    global TRACKIO
    if args.no_trackio:
        return
    try:
        import trackio

        trackio.init(
            project=args.trackio_project,
            name=args.run_name,
            space_id=args.trackio_space,
            bucket_id=args.trackio_bucket,
            private=True,
            config=vars(args),
        )
        TRACKIO = trackio
        print(
            "trackio: logging to", args.trackio_space, args.trackio_project, flush=True
        )
    except Exception as e:
        print(f"WARNING: trackio init failed, continuing without it: {e!r}", flush=True)


def tracker_log(metrics, step=None):
    if TRACKIO is None:
        return
    try:
        TRACKIO.log(
            {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and v is not None
            },
            step=step,
        )
    except Exception as e:
        print(f"WARNING: trackio log failed: {e!r}", flush=True)


def tracker_finish():
    if TRACKIO is None:
        return
    try:
        TRACKIO.finish()
    except Exception as e:
        print(f"WARNING: trackio finish failed: {e!r}", flush=True)


# ------------------------------------------------------------------ engine mapping (di-stage0/engines/gliclass_engine.py)
def as_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def render_state(state):
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


def engine_options(question):
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        if not criteria or len(criteria) < 2:
            raise ValueError("a choice question needs at least two criteria")
        if isinstance(criteria, dict):
            return list(criteria), [
                as_text(v) if v is not None else k for k, v in criteria.items()
            ]
        return [str(c) for c in criteria], [str(c) for c in criteria]
    if kind == "noul":
        c = criteria or {}
        return ["true", "false"], [
            as_text(c.get("true") or "yes"),
            as_text(c.get("false") or "no"),
        ]
    if kind == "score":  # extension: the engine refuses score questions
        return [str(i) for i in range(len(criteria))], [as_text(c) for c in criteria]
    raise ValueError(f"unsupported question type {kind!r}")


# ------------------------------------------------------------------ enc_eval mapping (scripts/enc_eval.py)
def enceval_option_list(question):
    qtype, criteria = question["type"], question["criteria"]
    if qtype == "noul":
        criteria = criteria if isinstance(criteria, dict) else {}
        return [
            (key, criteria.get(key) or f"The proposition is {key}.")
            for key in ("true", "false")
        ]
    if qtype == "choice" and isinstance(criteria, dict):
        return [(key, text or key) for key, text in criteria.items()]
    if qtype == "choice":
        return [(str(key), str(key)) for key in criteria]
    return [(str(i), text) for i, text in enumerate(criteria)]


def evidence_text(state):
    if isinstance(state, str):
        return state
    keys = list(state)
    if set(keys) == {"task", "context", "response"}:
        keys = ["task", "response", "context"]
    lines = []
    for k in keys:
        v = state[k]
        if v in (None, "", [], {}):
            continue
        lines.append(
            f"{k.replace('_', ' ')}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def decision_inputs(item, mapping):
    """-> (keys, text, labels, prompt or None), the inference-time input of one decision item."""
    q = item["question"]
    if mapping == "engine":
        keys, labels = engine_options(q)
        return (
            keys,
            render_state(item["state"]),
            labels,
            as_text(q.get("instructions") or "") or None,
        )
    opts = enceval_option_list(q)
    return (
        [k for k, _ in opts],
        evidence_text(item["state"]),
        [f"{k}: {v}" for k, v in opts],
        q["instructions"],
    )


def target_vector(item, keys):
    """Soft target per key; noul with one side given -> the other side is 1 - p. None when no target."""
    target = item.get("target") or {}
    if not target:
        return None
    if item["question"]["type"] == "noul" and len(target) == 1:
        if "true" in target:
            return [float(target["true"]), 1.0 - float(target["true"])]
        if "false" in target:
            return [1.0 - float(target["false"]), float(target["false"])]
    vec = [float(target.get(k, 0.0)) for k in keys]
    if sum(vec) > 0 or (
        item.get("multi_label") and len(vec) >= 2
    ):  # multi-label: an all-negative row is a target
        return vec
    return None


def gold_index(item, keys):  # as enc_eval.py
    target = item.get("target") or {}
    if not target:
        return None
    vec = [float(target.get(k, 0.0)) for k in keys]
    return int(np.argmax(vec)) if sum(vec) > 0 else None


def decision_to_gliclass(item, mapping):
    """Decision row -> gliclass row, or (None, reason)."""
    try:
        keys, text, labels, prompt = decision_inputs(item, mapping)
    except ValueError as e:
        return None, str(e)
    if len(set(labels)) < len(labels):
        return None, "duplicate label strings"
    vec = target_vector(item, keys)
    if vec is None:
        return None, "no target"
    row = {
        "text": text,
        "all_labels": labels,
        "true_labels": dict(zip(labels, vec)),
        "prompt": prompt or "",
        "id": item.get("id"),
    }
    for k in ("src", "teacher", "licence_class", "revision"):  # per-row provenance survives conversion
        if item.get(k) is not None:
            row[k] = item[k]
    if item["question"]["type"] == "score":  # ordered levels, low -> high
        row["ordinal_order"] = list(labels)
    elif item.get("ordinal") or (
        len(keys) >= 2 and all(re.fullmatch(r"level_\d+", k) for k in keys)
    ):  # graded choice rows: level_<k> keys (or an explicit flag) give the order, whatever the label wording
        ranked = sorted(zip(keys, labels), key=lambda kl: int(kl[0].split("_")[1]) if kl[0].startswith("level_") else 0)
        row["ordinal_order"] = [lab for _, lab in ranked] if not item.get("ordinal") else list(labels)
    if item.get("multi_label"):  # independent yes/no per label: never exclusive, even with one positive
        row["multi_label"] = True
    else:  # choice / noul / score questions pick one option: exclusive whatever the soft targets sum to
        row["single_label"] = True
    return row, None


def reservoir_lines(specs, k, seed):
    """Uniform sample of k lines over all files, streaming (keeps only k raw lines in memory)."""
    rng, sample, seen = random.Random(seed), [], 0
    for spec in specs:
        path, n = split_spec(spec)
        with open(path) as f:
            for j, line in enumerate(f):
                if n is not None and j >= n:
                    break
                if not line.strip():
                    continue
                if len(sample) < k:
                    sample.append(line)
                else:
                    r = rng.randrange(seen + 1)
                    if r < k:
                        sample[r] = line
                seen += 1
    print(f"train: sampled {len(sample)} of {seen} rows (seed {seed})", flush=True)
    rng.shuffle(sample)
    return [json.loads(line) for line in sample]


def load_train_rows(specs, mapping, sample=None, seed=0):
    rows, skipped, srcs = [], Counter(), Counter()
    parts = (
        [(f"sample of {len(specs)} file(s)", reservoir_lines(specs, sample, seed))]
        if sample
        else [(split_spec(spec)[0], load_jsonl(*split_spec(spec))) for spec in specs]
    )
    for path, part in parts:
        for r in part:
            srcs[str(r.get("src", "?")).split(":")[0]] += 1
            if "question" in r and "state" in r:
                g, why = decision_to_gliclass(r, mapping)
                if g is None:
                    skipped[why] += 1
                    continue
                rows.append(g)
            else:
                if not r.get("all_labels"):
                    skipped["no labels"] += 1
                    continue
                row = {
                    "text": str(r["text"]),
                    "all_labels": list(r["all_labels"]),
                    "true_labels": r["true_labels"],
                    "prompt": r.get("prompt") or "",
                    "id": r.get("id"),
                }
                if (
                    "teacher_blend" in r
                ):  # per-row gold weight (filt_r1_apply.py), used with --use-row-blend
                    row["teacher_blend"] = float(r["teacher_blend"])
                if r.get(
                    "ordinal_order"
                ):  # ordered levels (low -> high); used by --ordinal-sigma / --rps-coef
                    row["ordinal_order"] = list(r["ordinal_order"])
                if r.get("multi_label"):
                    row["multi_label"] = True
                if r.get("single_label"):
                    row["single_label"] = True
                rows.append(row)
        print(f"train: {len(part)} rows from {path}", flush=True)
    if skipped:
        print("train: skipped", dict(skipped), flush=True)
    print("train: sources", json.dumps(dict(srcs.most_common(30))), flush=True)
    return rows


# ------------------------------------------------------------------ targets: exclusivity, soft targets, teacher blend
def target_list(row):
    """Targets in all_labels order (list true_labels -> 0/1; dict -> scores, soft kept)."""
    tl = row["true_labels"]
    if isinstance(tl, dict):
        return [float(tl.get(lab, 0.0)) for lab in row["all_labels"]]
    gold = set(tl)
    return [1.0 if lab in gold else 0.0 for lab in row["all_labels"]]


def is_exclusive(targets):
    """Mutually exclusive options: 2+ labels and targets summing to 1 (one gold label, or a distribution)."""
    return len(targets) >= 2 and abs(sum(targets) - 1.0) < 1e-3


def ordinal_soften(row, targets, sigma):
    """Hard one-hot target on an ordinal row -> discretised Gaussian over the levels (sigma in levels): a graded label
    from one rater means "about level k". Soft (e.g. vote-split) targets are kept as they are."""
    order = row.get("ordinal_order")
    if not order or sigma <= 0 or sorted(targets) != [0.0] * (len(targets) - 1) + [1.0]:
        return targets
    rank = {lab: i for i, lab in enumerate(order)}
    g = rank.get(row["all_labels"][targets.index(1.0)])
    if g is None:
        return targets
    w = [
        math.exp(-((rank.get(lab, g) - g) ** 2) / (2 * sigma**2))
        if lab in rank
        else 0.0
        for lab in row["all_labels"]
    ]
    z = sum(w)
    return [x / z for x in w]


def check_ordinal(r, t):
    """None if the row's ordinal_order is usable, else the reason. Usable: unique levels, every level among the row's
    labels (so ranks are 0..K-1, contiguous), K >= 2, an exclusive target, and the gold mass on the levels (a row
    whose gold is a non-level option such as "Indeterminate" keeps CE but gets no RPS)."""
    order = r["ordinal_order"]
    if len(order) < 2 or len(set(order)) < len(order):
        return "ordinal_bad_levels"
    if not set(order) <= set(r["all_labels"]):
        return "ordinal_level_missing"
    if r.get("multi_label") or not (is_exclusive(t) or r.get("single_label")):
        return "ordinal_not_exclusive"
    lvl = set(order)
    if sum(v for lab, v in zip(r["all_labels"], t) if lab in lvl) <= 0:
        return "ordinal_gold_not_level"
    return None


def mark_targets(rows, ordinal_sigma=0.0):
    """Freeze targets as {label: score} (the dataset shuffles all_labels in place), set `exclusive`, report soft rows.
    Rows flagged multi_label are never exclusive (BCE path, also with one positive). Ordinal rows that fail
    check_ordinal lose their ordinal_order (counted), so RPS / softening never apply to a malformed row."""
    stats = Counter()
    for r in rows:
        t = target_list(r)
        if r.get("single_label") and not r.get("multi_label") and len(t) >= 2 and sum(t) > 0:
            if abs(sum(t) - 1.0) >= 1e-6:  # e.g. WebOrganizer soft targets summing to 0.998: renormalise
                stats["single_label_renormalised"] += 1
                z = sum(t)
                t = [v / z for v in t]
        if r.get("ordinal_order"):
            why = check_ordinal(r, t)
            if why:
                stats[why] += 1
                del r["ordinal_order"]
            else:
                stats["ordinal"] += 1
                t = ordinal_soften(r, t, ordinal_sigma)
        r["exclusive"] = 0 if r.get("multi_label") else int(bool(is_exclusive(t) or (r.get("single_label") and sum(t) > 0)))
        stats["multi_label"] += bool(r.get("multi_label"))
        soft = any(0.0 < v < 1.0 for v in t)
        stats["soft" if soft else "hard"] += 1
        stats["exclusive"] += r["exclusive"]
        stats["exclusive_soft"] += r["exclusive"] and soft
        if len(set(r["all_labels"])) < len(r["all_labels"]):
            stats["duplicate_labels"] += 1
        r["true_labels"] = dict(zip(r["all_labels"], t))
    print("targets", json.dumps(dict(stats)), flush=True)
    return dict(stats)


def teacher_distribution(logits, exclusive):
    z = torch.tensor(logits, dtype=torch.float64)
    return (torch.softmax(z, -1) if exclusive else torch.sigmoid(z)).tolist()


def apply_teacher(rows, paths, weights, alpha, use_row_blend=False):
    """target = alpha * gold + (1 - alpha) * teacher; teacher = weighted mean over the score files (filt_score.py
    parquet: id + per-label logits in the row's original all_labels order), softmax for exclusive rows, sigmoid
    otherwise. Rows without scores in every file keep their gold targets. With use_row_blend a row's own
    `teacher_blend` field replaces alpha for that row; rows with alpha >= 1 keep their gold targets."""
    import polars as pl

    tables = []
    for spec in paths:  # path or path#column (e.g. scores_r0.parquet#pilot_logits)
        path, _, col = spec.partition("#")
        col = col or "logits"
        df = pl.read_parquet(path, columns=["id", col]).drop_nulls()
        tables.append(dict(zip(df["id"].to_list(), df[col].to_list())))
    weights = weights or [1.0 / len(paths)] * len(paths)
    assert len(weights) == len(paths) and abs(sum(weights) - 1) < 1e-6, weights
    stats = Counter()
    for r in rows:
        a = r.get("teacher_blend", alpha) if use_row_blend else alpha
        if a >= 1.0:
            stats["gold_only"] += 1
            continue
        logit_sets = [t.get(r.get("id")) for t in tables]
        if any(lg is None or len(lg) != len(r["all_labels"]) for lg in logit_sets):
            stats["no_teacher"] += 1
            continue
        gold = [r["true_labels"][lab] for lab in r["all_labels"]]
        teacher = [0.0] * len(gold)
        for w, lg in zip(weights, logit_sets):
            for k, v in enumerate(teacher_distribution(lg, r["exclusive"])):
                teacher[k] += w * v
        r["true_labels"] = {
            lab: a * g + (1 - a) * t
            for lab, g, t in zip(r["all_labels"], gold, teacher)
        }
        stats["blended"] += 1
        stats["row_blend"] += use_row_blend and "teacher_blend" in r
    print("teacher", json.dumps(dict(stats)), flush=True)
    return dict(stats)


def augment_label_sets(
    rows, distractors, distractor_max_labels, neg_subset, neg_subset_min_labels, seed
):
    """Opt-in label-set augmentation against label-name priors (after targets are final; one pass, seeded).
    neg_subset P: with probability P, a multi-label (not exclusive) row with >= neg_subset_min_labels labels keeps a
      random subset of its negatives (size uniform in 1..n_neg; at least 2 labels in all); every label with target
      > 0 is kept. Exclusive rows are skipped: their options can depend on each other ("choose none if ...").
    distractors K: a row with < distractor_max_labels labels gets k ~ U[1, K] extra labels with target 0, drawn
      uniformly from the distinct label names of the training rows (no name already in the row, case-insensitive).
    Gold labels and targets are unchanged and `exclusive` keeps its value (a single-label row stays single-label)."""
    rng = random.Random(seed)
    vocab = sorted({lab for r in rows for lab in r["all_labels"]})
    stats = Counter()
    for r in rows:
        labels, targets = r["all_labels"], r["true_labels"]
        if (
            neg_subset > 0
            and not r["exclusive"]
            and len(labels) >= neg_subset_min_labels
            and rng.random() < neg_subset
        ):
            pos = [lab for lab in labels if targets[lab] > 0]
            neg = [lab for lab in labels if targets[lab] <= 0]
            if neg:
                keep = set(rng.sample(neg, max(rng.randint(1, len(neg)), 2 - len(pos))))
                labels = [lab for lab in labels if targets[lab] > 0 or lab in keep]
                stats["neg_subset_rows"] += 1
                stats["neg_subset_dropped"] += len(r["all_labels"]) - len(labels)
        if distractors > 0 and len(labels) < distractor_max_labels:
            taken = {lab.strip().lower() for lab in labels}
            extra = []
            for _ in range(rng.randint(1, distractors)):
                for _try in range(10):
                    cand = vocab[rng.randrange(len(vocab))]
                    if cand.strip().lower() not in taken:
                        taken.add(cand.strip().lower())
                        extra.append(cand)
                        break
            labels = labels + extra
            stats["distractor_rows"] += 1
            stats["distractors_added"] += len(extra)
        r["all_labels"] = labels
        r["true_labels"] = {lab: targets.get(lab, 0.0) for lab in labels}
    n_labels = Counter(min(len(r["all_labels"]), 10) for r in rows)
    print(
        "label_aug",
        json.dumps(
            {
                **stats,
                "vocab": len(vocab),
                "n_labels(10=10+)": dict(sorted(n_labels.items())),
            }
        ),
        flush=True,
    )
    return {**stats, "vocab": len(vocab)}


# ------------------------------------------------------------------ model
def init_special_token_embeddings(embeddings, tokenizer, new_ids, num_pretrained_rows):
    """As gliclass train.py: seed each new token row from a word embedding, scaled to the median row norm."""
    weight = embeddings.weight
    with torch.no_grad():
        target_norm = weight[:num_pretrained_rows].float().norm(dim=-1).median()
        for token, token_id in new_ids.items():
            word = SPECIAL_TOKEN_INIT_WORDS[token]
            word_ids = [
                i
                for i in tokenizer(word, add_special_tokens=False)["input_ids"]
                if i < num_pretrained_rows
            ]
            vector = weight[word_ids].float().mean(dim=0)
            weight[token_id] = (vector * target_norm / vector.norm()).to(weight.dtype)
            print(
                f"init {token} (id {token_id}) from {word!r}, norm {target_norm:.3f}",
                flush=True,
            )


def build_model(args):
    from gliclass import GLiClassModel, GLiClassModelConfig
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    num_pretrained_rows = len(tokenizer)
    encoder_config = AutoConfig.from_pretrained(args.encoder)
    # UltraBERT: tokenizer and embedding rows match. DeBERTa-v3 pads its embedding (128,100 rows for 128,001 tokens);
    # gliclass puts the new tokens at len(tokenizer) and resizes to the tokenizer, as here.
    if encoder_config.model_type == "ultrabert":
        assert num_pretrained_rows == encoder_config.vocab_size, (
            num_pretrained_rows,
            encoder_config.vocab_size,
        )
    assert num_pretrained_rows <= encoder_config.vocab_size, (
        num_pretrained_rows,
        encoder_config.vocab_size,
    )
    config = GLiClassModelConfig(
        encoder_config=encoder_config,
        encoder_model=args.encoder,
        label_model_name=None,
        class_token_index=num_pretrained_rows,
        text_token_index=num_pretrained_rows + 1,
        example_token_index=num_pretrained_rows + 2,
        pooling_strategy="first",
        class_token_pooling="first",
        embed_class_token=True,
        scorer_type="mlp",
        scorer_mlp_hidden_size=1024,
        dropout=args.dropout,
        focal_loss_alpha=args.focal_alpha,
        focal_loss_gamma=args.focal_gamma,
        focal_loss_reduction="mean",
        architecture_type="uni-encoder",
        prompt_first=True,
        shuffle_labels=True,
        problem_type="multi_label_classification",
        normalize_features=False,
        contrastive_loss_coef=0,
    )
    model = GLiClassModel(config, from_pretrained=True).to(dtype=torch.float32)
    # post_init must not re-initialise the pretrained encoder
    ref = __import__("transformers").AutoModel.from_pretrained(
        args.encoder, dtype=torch.float32
    )
    w0, w1 = ref.get_input_embeddings().weight, model.get_input_embeddings().weight
    assert torch.equal(w0, w1[: w0.shape[0]]), (
        "encoder weights changed after GLiClassModel init"
    )
    del ref
    tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    new_ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
    assert list(new_ids.values()) == [
        num_pretrained_rows,
        num_pretrained_rows + 1,
        num_pretrained_rows + 2,
    ], new_ids
    embeddings = model.resize_token_embeddings(len(tokenizer))
    init_special_token_embeddings(embeddings, tokenizer, new_ids, num_pretrained_rows)
    return model, tokenizer


def save_model(model, tokenizer, save_dir):
    """save_pretrained, then drop the encoder sub-config dtype: gliclass rebuilds the encoder with
    AutoModel.from_config(encoder_config), which keeps a saved float32 dtype even under
    GLiClassModel.from_pretrained(torch_dtype=bfloat16) and gives a mixed fp32/bf16 model (the engine's bf16 load
    then fails in the projector)."""
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    cfg_path = Path(save_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    for key in ("dtype", "torch_dtype"):
        cfg.get("encoder_config", {}).pop(key, None)
    cfg_path.write_text(json.dumps(cfg, indent=2))


def load_model(path):
    from gliclass import GLiClassModel
    from transformers import AutoTokenizer

    model = GLiClassModel.from_pretrained(path, torch_dtype=torch.float32)
    return model, AutoTokenizer.from_pretrained(path)


def check_tokenizer(model, tokenizer):
    """bos first, pad = 0 masked, special tokens = the config indices, labels survive as single tokens."""
    cfg = model.config
    ids = tokenizer("<<LABEL>>a cat<<LABEL>>dog<<SEP>>Is it?some text")["input_ids"]
    info = {
        "bos": tokenizer.bos_token_id,
        "pad": tokenizer.pad_token_id,
        "sep": tokenizer.sep_token_id,
        "mask": tokenizer.mask_token_id,
        "len": len(tokenizer),
        "padding_side": tokenizer.padding_side,
        "class_token_index": cfg.class_token_index,
        "text_token_index": cfg.text_token_index,
        "example_token_index": cfg.example_token_index,
        "embedding_rows": model.get_input_embeddings().num_embeddings,
        "sample_ids": ids,
    }
    print("tokenizer", json.dumps(info), flush=True)
    if cfg.encoder_config.model_type == "ultrabert":
        assert ids[0] == tokenizer.bos_token_id == 0 and tokenizer.pad_token_id == 0, (
            info
        )
        assert tokenizer.sep_token_id == 15 and tokenizer.mask_token_id == 16, info
        assert (
            ids.count(cfg.class_token_index) == 2
            and ids.count(cfg.text_token_index) == 1
        ), info
        assert ids[1] == cfg.class_token_index, info
        assert tokenizer.padding_side == "right", info
        assert info["embedding_rows"] == len(tokenizer) == cfg.vocab_size, info
    return info


def set_attention(model, impl):
    enc = model.model.encoder_model
    if enc.config.model_type not in (
        "ultrabert",
        "modernbert",
    ):  # modernbert: Ettin rerankers (B1)
        return getattr(enc.config, "_attn_implementation", None)
    enc.set_attn_implementation(impl)
    got = enc.config._attn_implementation
    assert got == impl, (got, impl)
    return got


# ------------------------------------------------------------------ batching
def build_string(row):
    """The uni-encoder prompt_first string of GLiClassDataset (labels in stored order)."""
    return (
        "".join(f"<<LABEL>>{label}" for label in row["all_labels"])
        + "<<SEP>>"
        + (row.get("prompt") or "")
        + str(row["text"])
    )


def drop_label_overflow(rows, tokenizer, max_length, margin=16):
    """Rows whose label block (<<LABEL>>l1...<<SEP>>) alone reaches max_length - margin lose label slots to
    truncation; CE would then renormalise the target onto the visible labels. Drop them and count."""
    keep, dropped = [], 0
    for s in range(0, len(rows), 2000):
        part = rows[s : s + 2000]
        enc = tokenizer(["".join(f"<<LABEL>>{lab}" for lab in r["all_labels"]) + "<<SEP>>" for r in part])
        for r, ids in zip(part, enc["input_ids"]):
            if len(ids) >= max_length - margin:
                dropped += 1
            else:
                keep.append(r)
    print(f"label overflow: dropped {dropped} of {len(rows)} rows (label block >= {max_length - margin} tokens)",
          flush=True)
    return keep, dropped


def token_lengths(rows, tokenizer, max_length, chunk=2000):
    lengths = []
    for s in range(0, len(rows), chunk):
        enc = tokenizer(
            [build_string(r) for r in rows[s : s + chunk]],
            truncation=True,
            max_length=max_length,
        )
        lengths += [len(x) for x in enc["input_ids"]]
    return lengths


def make_batches(lengths, token_budget, max_batch, rng, pool=100):
    """Shuffle, sort by length inside pools of pool*max_batch items, cut batches with n * max_len <= token_budget."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    batches = []
    size = pool * max_batch
    for s in range(0, len(idx), size):
        chunk = sorted(idx[s : s + size], key=lambda i: lengths[i])
        cur, cur_max = [], 0
        for i in chunk:
            new_max = max(cur_max, lengths[i])
            if cur and (
                new_max * (len(cur) + 1) > token_budget or len(cur) >= max_batch
            ):
                batches.append(cur)
                cur, new_max = [], lengths[i]
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def to_device(batch):
    return {
        "input_ids": batch["input_ids"].to(DEVICE, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(DEVICE, non_blocking=True),
        "labels": batch["labels"].to(DEVICE, non_blocking=True).float(),
        "exclusive": batch["exclusive"].to(DEVICE, non_blocking=True).bool()
        if "exclusive" in batch
        else None,
        "ranks": batch["ranks"].to(DEVICE, non_blocking=True)
        if "ranks" in batch
        else None,
        "max_num_classes": batch["max_num_classes"],
    }


def rps_per_row(logits, target, ranks):
    """Conditional ranked probability score per row: sum over thresholds k of (P(level <= k) - T(level <= k))^2 /
    (K - 1), levels ordered by `ranks` (slot -> level rank, -1 = not a level). P = softmax over the level logits;
    T renormalised over the level slots (mass on non-level options such as "Indeterminate" does not move the level CDF). Rows without
    levels, or whose target has no mass on the levels, get 0. check_ordinal guarantees ranks 0..K-1."""
    valid = ranks >= 0
    n_levels = valid.sum(-1)
    width = int(ranks.max().item()) + 1 if valid.any() else 0
    if width < 2:
        return torch.zeros(logits.shape[0], device=logits.device)
    onehot = torch.nn.functional.one_hot(
        ranks.clamp_min(0), width
    ).float() * valid.unsqueeze(-1)  # [B, S, L]
    # softmax over the LEVEL logits only (stable when a non-level logit dominates; review 9 H5)
    lvl_logits = logits.masked_fill(~valid, float("-inf"))
    has = valid.any(-1, keepdim=True)
    probs = torch.softmax(torch.where(has, lvl_logits, torch.zeros_like(lvl_logits)), -1) * valid
    p_lvl = torch.einsum("bs,bsl->bl", probs, onehot).cumsum(-1)
    t_lvl = torch.einsum("bs,bsl->bl", target, onehot)
    t_mass = t_lvl.sum(-1, keepdim=True)
    t_lvl = (t_lvl / t_mass.clamp_min(1e-8)).cumsum(-1)
    n_levels = torch.where(t_mass.squeeze(-1) > 0, n_levels, torch.zeros_like(n_levels))
    lvl_mask = torch.arange(width, device=probs.device).unsqueeze(0) < (
        n_levels - 1
    ).unsqueeze(-1)  # K-1 thresholds
    rps = (((p_lvl - t_lvl) ** 2) * lvl_mask).sum(-1) / (n_levels - 1).clamp_min(1)
    return torch.where(n_levels >= 2, rps, torch.zeros_like(rps))


def compute_loss(model, batch, single_label_loss="bce", brier_coef=0.0, rps_coef=0.0):
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_num_classes=batch["max_num_classes"],
            return_dict=True,
        )
    logits, labels, mask = out.logits.float(), batch["labels"], out.class_mask.bool()
    excl = batch.get("exclusive")
    if single_label_loss == "bce" or excl is None or not excl.any():
        # gliclass loss (alpha-weighted BCE over valid label slots), in fp32
        return model.model.get_loss(logits, labels, None, out.class_mask)
    # softmax_ce: exclusive rows -> CE against the (soft) target over that row's labels; other rows -> the gliclass
    # alpha-weighted BCE averaged over their valid slots; the batch loss is the mean over rows.
    # brier_coef > 0 adds brier_coef * sum_k (p_k - y_k)^2 (p = softmax over the row's labels) to exclusive rows
    from gliclass.loss_functions import focal_loss_with_logits

    cfg = model.config
    masked = logits.masked_fill(~mask, float("-inf"))
    target = (labels * mask).clamp_min(0)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    ce = -(target * torch.log_softmax(masked, -1).masked_fill(~mask, 0.0)).sum(-1)
    if brier_coef > 0:
        ce = ce + brier_coef * ((torch.softmax(masked, -1) - target) ** 2 * mask).sum(
            -1
        )
    if (
        rps_coef > 0 and batch.get("ranks") is not None
    ):  # order-aware term for ordinal rows (rank >= 0 slots)
        ce = ce + rps_coef * rps_per_row(masked, target, batch["ranks"])
    bce = focal_loss_with_logits(
        logits, labels, cfg.focal_loss_alpha, cfg.focal_loss_gamma, "none"
    )
    bce_row = (bce * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return torch.where(excl, ce, bce_row).mean()


def param_groups(model, encoder_lr, others_lr, wd_encoder, wd_others):
    groups = {(e, d): [] for e in (True, False) for d in (True, False)}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_encoder = name.startswith("model.encoder_model.")
        decay = p.ndim >= 2  # no decay on biases and norm weights
        groups[(is_encoder, decay)].append(p)
    out = []
    for (is_encoder, decay), params in groups.items():
        if params:
            out.append(
                {
                    "params": params,
                    "lr": encoder_lr if is_encoder else others_lr,
                    "weight_decay": (wd_encoder if is_encoder else wd_others)
                    if decay
                    else 0.0,
                    "name": f"{'encoder' if is_encoder else 'others'}{'' if decay else '_nodecay'}",
                }
            )
    return out


# ------------------------------------------------------------------ throughput bench
def bench(model, args, lengths):
    """Forward + backward (no optimizer step) on synthetic gliclass inputs of fixed length."""
    cfg = model.config
    results = []
    model.train()
    for L in lengths:
        bs = max(1, args.token_budget // L)
        g = torch.Generator().manual_seed(0)
        head = [0]
        for _ in range(8):
            head += [cfg.class_token_index] + torch.randint(
                1000, 40000, (3,), generator=g
            ).tolist()
        head += [cfg.text_token_index]
        ids = torch.randint(1000, 40000, (bs, L), generator=g)
        ids[:, : len(head)] = torch.tensor(head)
        batch = {
            "input_ids": ids.to(DEVICE),
            "attention_mask": torch.ones_like(ids).to(DEVICE),
            "labels": (torch.rand(bs, 8, generator=g) < 0.2).float().to(DEVICE),
            "max_num_classes": 8,
        }
        try:
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for step in range(3 + args.bench_steps):
                if step == 3:
                    sync()
                    t0 = time.perf_counter()
                loss = compute_loss(
                    model, batch, args.single_label_loss, args.brier_coef, args.rps_coef
                )
                loss.backward()
                model.zero_grad(set_to_none=True)
            sync()
            secs = (time.perf_counter() - t0) / args.bench_steps
            r = {
                "length": L,
                "batch": bs,
                "tokens_per_s": bs * L / secs,
                "s_per_step": secs,
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9
                if DEVICE == "cuda"
                else None,
            }
        except (
            torch.OutOfMemoryError
        ) as e:  # report and stop the bench; training would fail the same way
            r = {"length": L, "batch": bs, "error": f"OOM: {str(e)[:200]}"}
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print("bench", json.dumps(r), flush=True)
        results.append(r)
    return results


# ------------------------------------------------------------------ training
def train(model, tokenizer, rows, args, out):
    from gliclass.data_processing import (
        AugmentationConfig,
        DataCollatorWithPadding,
        GLiClassDataset,
    )

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    class Dataset(GLiClassDataset):
        def _prepare(self, example):
            out = super()._prepare(example)
            out["exclusive"] = int(example.get("exclusive", 0))
            order = {lab: i for i, lab in enumerate(example.get("ordinal_order") or [])}
            out["ranks"] = (
                [order.get(lab, -1) for lab in out["labels_text"]] if order else []
            )
            return out

    ds = Dataset(
        rows,
        tokenizer,
        AugmentationConfig(enabled=False),
        {},
        max_length=args.max_length,
        problem_type="multi_label_classification",
        architecture_type="uni-encoder",
        prompt_first=True,
        shuffle_labels=True,
    )
    lengths = token_lengths(rows, tokenizer, args.max_length)
    n_trunc = sum(1 for n in lengths if n >= args.max_length)
    print(
        f"train rows {len(rows)}; tokens median {statistics.median(lengths)} p95 "
        f"{sorted(lengths)[int(0.95 * len(lengths)) - 1]} max {max(lengths)}; at max_length {n_trunc}",
        flush=True,
    )
    collator = DataCollatorWithPadding(device="cpu")

    def collate(items):
        assert all(len(it["labels_text"]) == len(it["labels"]) for it in items)
        batch = collator(
            [
                {
                    "input_ids": it["input_ids"],
                    "attention_mask": it["attention_mask"],
                    "labels": it["labels"],
                    "labels_text": it["labels_text"],
                    "exclusive": it["exclusive"],
                }
                for it in items
            ]
        )
        ranks = torch.full(
            batch["labels"].shape, -1, dtype=torch.long
        )  # level rank per label slot; -1 = not ordinal
        for b, it in enumerate(items):
            if it["ranks"]:
                ranks[b, : len(it["ranks"])] = torch.tensor(it["ranks"])
        batch["ranks"] = ranks
        return batch

    # batch plans for every epoch up front, so the schedule knows the total step count
    plans, total = [], 0
    target_steps = args.max_steps if args.max_steps is not None else None
    epoch = 0
    while True:
        batches = make_batches(
            lengths,
            args.token_budget,
            args.batch_size,
            random.Random(args.seed * 1000 + epoch),
        )
        plans.append(batches)
        total += len(batches)
        epoch += 1
        if (target_steps is not None and total >= target_steps) or (
            target_steps is None and epoch >= args.epochs
        ):
            break
    total_steps = target_steps if target_steps is not None else total
    steps_per_epoch = len(plans[0])
    warmup = max(1, int(args.warmup * total_steps))
    print(
        f"steps per epoch {steps_per_epoch}, total steps {total_steps}, warmup {warmup}",
        flush=True,
    )
    if steps_per_epoch < 50:
        print(f"WARNING: only {steps_per_epoch} optimizer steps per epoch", flush=True)

    # dev rows (--dev-holdout): held out of training in main(); scored with the training loss every --dev-every
    # steps (no grad, eval mode) so the release run can pick its step on rows the gate never sees
    dev_rows = getattr(args, "dev_rows", None) or []
    dev_ds = Dataset(
        dev_rows,
        tokenizer,
        AugmentationConfig(enabled=False),
        {},
        max_length=args.max_length,
        problem_type="multi_label_classification",
        architecture_type="uni-encoder",
        prompt_first=True,
        shuffle_labels=False,  # fixed label order: dev readings comparable across steps (review P1)
    ) if dev_rows else None
    dev_batches = (
        make_batches(token_lengths(dev_rows, tokenizer, args.max_length), args.token_budget, args.batch_size,
                     random.Random(0))
        if dev_rows else []
    )
    dev_curve = []

    @torch.no_grad()
    def dev_loss():
        """mean training loss over the fixed dev batches, weighted by rows per batch (BCE rows average over their
        label slots, so this is not a global slot mean; with fixed batches and label order it is comparable across
        steps, which is all step selection needs). Python / torch / CUDA RNG state is restored afterwards, so dev
        scoring does not change the training run (review P1)."""
        py_state, torch_state = random.getstate(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if DEVICE == "cuda" else None
        model.eval()
        loader = torch.utils.data.DataLoader(dev_ds, batch_sampler=dev_batches, collate_fn=collate, num_workers=0,
                                             generator=torch.Generator().manual_seed(0))
        tot, n = 0.0, 0
        for b in loader:
            b = to_device(b)
            k = b["input_ids"].shape[0]
            tot += compute_loss(model, b, args.single_label_loss, args.brier_coef, args.rps_coef).item() * k
            n += k
        model.train()
        random.setstate(py_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        return tot / max(n, 1)

    def save_checkpoint(step):
        d = out / "ckpt" / f"s{step}"
        save_model(model, tokenizer, d)
        print("checkpoint", d, flush=True)

    groups = param_groups(
        model, args.encoder_lr, args.others_lr, args.weight_decay, args.weight_decay
    )
    for g in groups:
        print(
            f"param group {g['name']}: {sum(p.numel() for p in g['params']) / 1e6:.1f}M lr {g['lr']} wd {g['weight_decay']}"
        )
    opt = torch.optim.AdamW(
        groups, betas=(0.9, 0.999), eps=1e-8, fused=DEVICE == "cuda"
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(
            (s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))
        ),
    )
    if args.grad_ckpt:
        model.model.encoder_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log = open(out / "train_log.jsonl", "w")
    step, examples, t_start = 0, 0, time.perf_counter()
    win = {"loss": [], "real": 0, "padded": 0, "t": time.perf_counter(), "items": 0}
    first_losses, last_losses = [], []
    done = False
    for ep, batches in enumerate(plans):
        loader = torch.utils.data.DataLoader(
            ds,
            batch_sampler=batches,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=DEVICE == "cuda",
        )
        for batch in loader:
            batch = to_device(batch)
            loss = compute_loss(
                model, batch, args.single_label_loss, args.brier_coef, args.rps_coef
            )
            lv = loss.item()
            if not math.isfinite(lv) or lv == 0.0:
                raise RuntimeError(f"step {step}: loss {lv} (zero or non-finite)")
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            ).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            n = batch["input_ids"].shape[0]
            examples += n
            win["loss"].append(lv)
            win["real"] += int(batch["attention_mask"].sum().item())
            win["padded"] += batch["input_ids"].numel()
            win["items"] += n
            (first_losses if step <= 20 else last_losses).append(lv)
            last_losses[:] = last_losses[-20:]
            if step % args.log_every == 0 or step == total_steps:
                sync()
                dt = time.perf_counter() - win["t"]
                rec = {
                    "step": step,
                    "epoch": ep + (step - ep * steps_per_epoch) / steps_per_epoch,
                    "loss": statistics.mean(win["loss"]),
                    "grad_norm": gnorm,
                    "lr_encoder": next(
                        g["lr"]
                        for g in opt.param_groups
                        if g["name"].startswith("encoder")
                    ),
                    "lr_others": next(
                        g["lr"]
                        for g in opt.param_groups
                        if g["name"].startswith("others")
                    ),
                    "tokens_per_s": win["real"] / dt,
                    "padded_tokens_per_s": win["padded"] / dt,
                    "pad_frac": 1 - win["real"] / win["padded"],
                    "items_per_s": win["items"] / dt,
                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9
                    if DEVICE == "cuda"
                    else None,
                }
                print("train", json.dumps(rec), flush=True)
                tracker_log(
                    {f"train/{k}": v for k, v in rec.items() if k != "step"}, step=step
                )
                log.write(json.dumps(rec) + "\n")
                log.flush()
                win = {
                    "loss": [],
                    "real": 0,
                    "padded": 0,
                    "t": time.perf_counter(),
                    "items": 0,
                }
            if step in args.eval_steps_set and step < total_steps and args.eval:
                cur_attn = model.model.encoder_model.config._attn_implementation
                set_attention(model, args.eval_attn)
                model.eval()
                res, _ = run_evals(
                    model, tokenizer, args, out, step=step, subdir=f"s{step}"
                )
                args.evals_by_step[step] = res
                if not args.no_sync:  # partial results survive a timeout; never fatal
                    try:
                        from huggingface_hub import sync_bucket

                        sync_bucket(str(out), f"{args.out_bucket}/{args.run_name}")
                    except Exception as e:
                        print(f"WARNING: mid-run sync failed: {e!r}", flush=True)
                set_attention(model, cur_attn)
                model.train()
            dev_due = dev_rows and args.dev_every and (step % args.dev_every == 0 or step == total_steps)
            if dev_due:
                rec = {"step": step, "dev_loss": dev_loss()}
                dev_curve.append(rec)
                print("dev", json.dumps(rec), flush=True)
                tracker_log({"dev/loss": rec["dev_loss"]}, step=step)
                with open(out / "dev_log.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if args.save_every and (
                (step % args.save_every == 0 and step < total_steps) or (step == total_steps and not args.save_model)
            ):
                save_checkpoint(step)  # with --save-model, main() saves the final model to <out>/model
            if (dev_due or (args.save_every and step % args.save_every == 0)) and step < total_steps \
                    and not args.no_sync:
                try:  # partial results and checkpoints survive a timeout; never fatal
                    from huggingface_hub import sync_bucket

                    sync_bucket(str(out), f"{args.out_bucket}/{args.run_name}")
                except Exception as e:
                    print(f"WARNING: mid-run sync failed: {e!r}", flush=True)
            if step >= total_steps:
                done = True
                break
        if done:
            break
    secs = time.perf_counter() - t_start
    log.close()
    if args.grad_ckpt:
        model.model.encoder_model.gradient_checkpointing_disable()
    model.eval()
    return {
        "steps": step,
        "examples": examples,
        "epochs_run": examples / len(rows),
        "steps_per_epoch": steps_per_epoch,
        "train_seconds": secs,
        "loss_first20": statistics.mean(first_losses),
        "loss_last20": statistics.mean(last_losses) if last_losses else None,
        "n_rows": len(rows),
        "rows_at_max_length": n_trunc,
        **({"dev_rows": len(dev_rows), "dev_curve": dev_curve,
            "dev_best_step": min(dev_curve, key=lambda r: r["dev_loss"])["step"]} if dev_curve else {}),
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9
        if DEVICE == "cuda"
        else None,
    }


# ------------------------------------------------------------------ eval
def make_pipe(model, tokenizer, max_length):
    from gliclass import ZeroShotClassificationPipeline

    pipe = ZeroShotClassificationPipeline(
        model,
        tokenizer,
        classification_type="single-label",
        max_length=max_length,
        device=DEVICE,
        progress_bar=False,
    )
    return getattr(pipe, "pipe", pipe)


@torch.inference_mode()
def forward_logits(model, pipe, batch_inputs, readout):
    texts = [b[1] for b in batch_inputs]
    labels = [b[2] for b in batch_inputs]
    prompts = [b[3] for b in batch_inputs]
    tok = pipe.prepare_inputs(texts, labels, same_labels=False, prompt=prompts)
    n_cls = (tok["input_ids"] == model.config.class_token_index).sum(-1).tolist()
    n = max(len(x) for x in labels)
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        out = model(**tok, max_num_classes=n)
    raw = out.logits
    if readout == "raw" and getattr(out, "uncalibrated_logits", None) is not None:
        raw = out.uncalibrated_logits
    raw = raw.float().cpu()
    lens = tok["attention_mask"].sum(-1).tolist()
    return [raw[i, : len(labels[i])].tolist() for i in range(len(labels))], n_cls, lens


def predict(model, pipe, items, args):
    inputs = [decision_inputs(it, args.mapping) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(inputs[i][1]))
    logits, cut, lengths = [None] * len(items), 0, [0] * len(items)
    sync()
    t0 = time.perf_counter()
    for s in range(0, len(order), args.eval_batch_size):
        idx = order[s : s + args.eval_batch_size]
        lg, n_cls, lens = forward_logits(
            model, pipe, [inputs[i] for i in idx], args.readout
        )
        for i, row_logits, c, ln in zip(idx, lg, n_cls, lens):
            logits[i], lengths[i] = row_logits, ln
            cut += c != len(inputs[i][2])
    sync()
    secs = time.perf_counter() - t0
    preds = []
    for it, inp, lg in zip(items, inputs, logits):
        keys = inp[0]
        preds.append(
            {
                "id": it["id"],
                "ids": keys,
                "qtype": it["question"]["type"],
                "logits": lg,
                "gold": gold_index(it, keys),
            }
        )
    return preds, {
        "seconds": secs,
        "items_per_s": len(items) / secs,
        "labels_cut_by_truncation": cut,
        "inputs_at_max_length": sum(1 for n in lengths if n >= args.eval_max_length),
        "eval_max_length": args.eval_max_length,
    }


def softmax(x):
    z = np.asarray(x, dtype=np.float64)
    z = np.exp(z - z.max())
    return z / z.sum()


def score(preds):
    rows = [p for p in preds if p["gold"] is not None]
    res = {"n": len(rows)}
    if not rows:
        return res
    hits = [int(np.argmax(p["logits"]) == p["gold"]) for p in rows]
    res["acc"] = float(np.mean(hits))
    res["nll"] = float(
        np.mean([-math.log(max(softmax(p["logits"])[p["gold"]], 1e-12)) for p in rows])
    )
    by_type = {}
    for qt in sorted({p["qtype"] for p in rows}):
        sub = [p for p in rows if p["qtype"] == qt]
        top = Counter(p["ids"][int(np.argmax(p["logits"]))] for p in sub)
        by_type[qt] = {
            "n": len(sub),
            "acc": float(
                np.mean([int(np.argmax(p["logits"]) == p["gold"]) for p in sub])
            ),
            "top_class": top.most_common(1)[0][0],
            "top_class_share": top.most_common(1)[0][1] / len(sub),
        }
    res["by_type"] = by_type
    # macro-F1 over (question type, option key) classes that occur as gold
    gold = [(p["qtype"], p["ids"][p["gold"]]) for p in rows]
    pred = [(p["qtype"], p["ids"][int(np.argmax(p["logits"]))]) for p in rows]
    f1s = []
    for c in set(gold):
        tp = sum(1 for g, q in zip(gold, pred) if g == c and q == c)
        fp = sum(1 for g, q in zip(gold, pred) if g != c and q == c)
        fn = sum(1 for g, q in zip(gold, pred) if g == c and q != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    res["macro_f1"] = float(np.mean(f1s))
    res["collapse"] = any(
        v["top_class_share"] > 0.95 and v["n"] >= 20 for v in by_type.values()
    )
    return res


def run_evals(model, tokenizer, args, out, step=None, subdir=""):
    step = args.final_step if step is None else step
    (out / "predictions" / subdir).mkdir(parents=True, exist_ok=True)
    pipe = make_pipe(model, tokenizer, args.eval_max_length)
    results, cache = {}, {}
    for spec in args.eval:
        name, _, rest = spec.partition("=")
        path, _, n = rest.partition("@")
        items = load_jsonl(path, int(n) if n else None)
        preds, timing = predict(model, pipe, items, args)
        with open(out / "predictions" / subdir / f"{name}__desc.jsonl", "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in preds)
        res = {"file": path, **score(preds), **timing}
        results[name] = res
        cache[name] = (items, preds)
        print("eval", name, f"step={step}", json.dumps(res), flush=True)
        flat = {
            f"eval/{name}/acc": res.get("acc"),
            f"eval/{name}/macro_f1": res.get("macro_f1"),
            f"eval/{name}/nll": res.get("nll"),
            f"eval/{name}/items_per_s": res["items_per_s"],
        }
        for qt, v in res.get("by_type", {}).items():
            flat[f"eval/{name}/{qt}/acc"] = v["acc"]
            flat[f"eval/{name}/{qt}/top_class_share"] = v["top_class_share"]
        tracker_log(flat, step=step)
    return results, cache


def reload_check(save_dir, args, cache):
    """Reload with gliclass (default attention), rerun the first eval rows, compare logits with the in-memory ones;
    also check that pipeline __call__ scores equal softmax(logits)."""
    model, tokenizer = load_model(save_dir)
    model.to(DEVICE).eval()
    attn = set_attention(model, args.eval_attn)
    pipe = make_pipe(model, tokenizer, args.eval_max_length)
    name = next(iter(cache))
    items, preds = cache[name]
    items, preds = items[: args.reload_check_n], preds[: args.reload_check_n]
    new, _ = predict(model, pipe, items, args)
    diffs = [
        float(np.max(np.abs(np.array(a["logits"]) - np.array(b["logits"]))))
        for a, b in zip(preds, new)
    ]
    agree = float(
        np.mean(
            [
                np.argmax(a["logits"]) == np.argmax(b["logits"])
                for a, b in zip(preds, new)
            ]
        )
    )
    # high-level pipeline __call__ on a few items: same top label, score ~ softmax(our logits)
    # (bf16 autocast: flash attention has no fp32 path; the engine loads the model in bf16)
    call_diffs, call_agree = [], []
    with (
        torch.inference_mode(),
        torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"),
    ):
        for it, p in list(zip(items, new))[:8]:
            _, text, labels, prompt = decision_inputs(it, args.mapping)
            best = pipe(text, labels, prompt=prompt)[0][0]
            ref = softmax(p["logits"])
            call_agree.append(best["label"] == labels[int(np.argmax(ref))])
            call_diffs.append(abs(best["score"] - float(ref.max())))
    # the engine's load: bf16 weights, no autocast; every parameter must be bf16 and the top label must hold
    from gliclass import GLiClassModel

    m16 = (
        GLiClassModel.from_pretrained(save_dir, torch_dtype=torch.bfloat16)
        .to(DEVICE)
        .eval()
    )
    set_attention(m16, args.eval_attn)
    dtypes = sorted({str(p.dtype) for p in m16.parameters()})
    pipe16 = make_pipe(m16, tokenizer, args.eval_max_length)
    bf16_agree = []
    with torch.inference_mode():
        for it, p in list(zip(items, new))[:16]:
            _, text, labels, prompt = decision_inputs(it, args.mapping)
            best = pipe16(text, labels, prompt=prompt)[0][0]
            bf16_agree.append(best["label"] == labels[int(np.argmax(p["logits"]))])
    res = {
        "eval": name,
        "n": len(items),
        "reload_attn": attn,
        "max_abs_logit_diff": max(diffs),
        "bf16_load_dtypes": dtypes,
        "bf16_load_top1_agreement": float(np.mean(bf16_agree)),
        "mean_abs_logit_diff": float(np.mean(diffs)),
        "top1_agreement": agree,
        "pipeline_call_top1_agreement": float(np.mean(call_agree)),
        "pipeline_call_max_prob_diff": max(call_diffs),
    }
    print("reload_check", json.dumps(res), flush=True)
    return res


# ------------------------------------------------------------------ provenance
PROV_ENV = ("JOB_ID", "ACCELERATOR", "CPU_CORES", "HF_HUB_OFFLINE", "MAX_STEPS", "IMAGE", "IMAGE_TAG", "SPACE_ID",
            "SPACE_REPO_SHA", "ARTIFACT_URI", "WRAPPER_SHA256")


def sha256_file(path, chunk=1 << 20):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def provenance_start(args, out):
    """Resolved command, env, script / data / eval hashes; written before training, completed at the end."""
    import sys

    me = sha256_file(__file__)
    if args.expect_sha256 and args.expect_sha256 != me:
        raise SystemExit(f"deployed trainer sha256 {me} != expected {args.expect_sha256}: republish the artifact")
    prov = {
        "argv": sys.argv,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trainer_sha256": me,
        "env": {k: os.environ[k] for k in PROV_ENV if k in os.environ},
        "train_files": {},
        "eval_files": {},
        "base_model": args.base_model,
        "encoder": args.encoder,
    }
    for spec in args.train:
        path = split_spec(spec)[0]
        prov["train_files"][spec] = sha256_file(path) if os.path.exists(path) else "missing"
    for spec in args.eval:
        path = spec.partition("=")[2].partition("@")[0]
        prov["eval_files"][spec] = sha256_file(path) if os.path.exists(path) else "missing"
    if args.mix and not any(os.path.abspath(split_spec(t)[0]) == os.path.abspath(os.path.join(args.mix, "mix.jsonl"))
                            for t in args.train):
        raise SystemExit(f"--mix {args.mix} given but no --train is its mix.jsonl")
    for key, d, names in (("mix", args.mix, ("spec.yaml", "manifest.parquet", "mix.jsonl")),
                          ("eval_version", args.eval_version, ("inventory.json",))):
        if d:
            prov[key] = {"dir": d}
            for n in names:
                p = os.path.join(d, n)
                prov[key][n] = sha256_file(p) if os.path.exists(p) else "missing"
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print("provenance", json.dumps({k: prov[k] for k in ("trainer_sha256", "env")}), flush=True)
    return prov


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--convert",
        nargs=2,
        metavar=("IN", "OUT"),
        help="decision JSONL -> gliclass JSONL, then exit",
    )
    ap.add_argument("--run-name")
    ap.add_argument("--encoder", default="dmis-lab/UltraBERT-150M")
    ap.add_argument(
        "--base-model",
        default=None,
        help="start from a saved GLiClass model instead of --encoder",
    )
    ap.add_argument(
        "--attn",
        default="kernels-community/flash-attn2",
        help="UltraBERT attention for training",
    )
    ap.add_argument(
        "--eval-attn",
        default=None,
        help="UltraBERT attention for eval and the reload check (default: --attn)",
    )
    ap.add_argument("--no-trackio", action="store_true")
    ap.add_argument("--trackio-project", default="ub-gliclass")
    ap.add_argument("--trackio-space", default="YOUR_NAME/ub-gliclass-trackio")
    ap.add_argument("--trackio-bucket", default="YOUR_NAME/ub-gliclass-trackio")
    ap.add_argument("--mapping", choices=["engine", "enceval"], default="engine")
    ap.add_argument("--readout", choices=["raw", "calibrated"], default="raw")
    ap.add_argument(
        "--train",
        action="append",
        default=[],
        help="path[:N]; gliclass or decision JSONL",
    )
    ap.add_argument(
        "--eval", action="append", default=[], help="name=decision.jsonl[@N]"
    )
    ap.add_argument(
        "--eval-steps",
        default="",
        help="e.g. 1000,2000: extra evals during training (final eval always)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="reservoir-sample N rows over all --train files",
    )
    ap.add_argument(
        "--single-label-loss",
        choices=["bce", "softmax_ce"],
        default="bce",
        help="softmax_ce: CE over the row's labels for exclusive rows (one gold or a distribution)",
    )
    ap.add_argument(
        "--rps-coef",
        type=float,
        default=0.0,
        help="with --single-label-loss softmax_ce: add COEF * ranked probability score on ordinal rows",
    )
    ap.add_argument(
        "--ordinal-sigma",
        type=float,
        default=0.0,
        help="spread hard targets on ordinal rows to a Gaussian over the levels (sigma in levels)",
    )
    ap.add_argument(
        "--brier-coef",
        type=float,
        default=0.0,
        help="with --single-label-loss softmax_ce: add COEF * Brier score over the options (exclusive rows)",
    )
    ap.add_argument(
        "--teacher-blend",
        type=float,
        default=1.0,
        help="ALPHA: target = ALPHA*gold + (1-ALPHA)*teacher",
    )
    ap.add_argument(
        "--teacher-scores",
        action="append",
        default=[],
        help="parquet[#logits column] with id + per-label logits (repeatable), e.g. scores_r0.parquet#pilot_logits",
    )
    ap.add_argument(
        "--teacher-weights",
        default=None,
        help="comma weights per --teacher-scores (default: mean)",
    )
    ap.add_argument(
        "--use-row-blend",
        action="store_true",
        help="rows with a teacher_blend field use it as their gold weight (others: --teacher-blend)",
    )
    ap.add_argument(
        "--distractors",
        type=int,
        default=0,
        help="K: rows with < --distractor-max-labels labels get U[1,K] random negative labels (0 = off)",
    )
    ap.add_argument("--distractor-max-labels", type=int, default=4)
    ap.add_argument(
        "--neg-subset",
        type=float,
        default=0.0,
        help="P: rows with >= --neg-subset-min-labels labels keep a random subset of negatives (0 = off)",
    )
    ap.add_argument("--neg-subset-min-labels", type=int, default=4)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--others-lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="overrides --epochs; 0 = no training",
    )
    ap.add_argument(
        "--batch-size", type=int, default=32, help="max items per optimizer step"
    )
    ap.add_argument(
        "--token-budget",
        type=int,
        default=32768,
        help="max padded tokens per optimizer step",
    )
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument(
        "--eval-max-length", type=int, default=None, help="default: --max-length"
    )
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--focal-alpha", type=float, default=0.75)
    ap.add_argument("--focal-gamma", type=float, default=-1)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument(
        "--bench-lengths",
        default="",
        help="e.g. 4096,8192: fwd+bwd tokens/s before training",
    )
    ap.add_argument("--bench-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save a checkpoint every N steps to <out>/ckpt/s<N> (synced to the bucket)")
    ap.add_argument("--dev-holdout", type=int, default=0,
                    help="hold N train rows out of training (lowest sha256 of the row id: the same rows for every "
                         "seed on one mix) and score them with the training loss every --dev-every steps")
    ap.add_argument("--dev-every", type=int, default=0)
    ap.add_argument("--reload-check-n", type=int, default=64)
    ap.add_argument("--out", default="/tmp/out")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--out-bucket", default="hf://buckets/YOUR_NAME/dm-recipe/runs")
    ap.add_argument(
        "--expect-sha256",
        default=None,
        help="sha256 of scripts/gliclass_ub_train.py at launch; the run stops if this deployed copy differs",
    )
    ap.add_argument("--mix", default=None, help="frozen mix dir (spec.yaml, manifest.parquet): recorded in provenance")
    ap.add_argument("--eval-version", default=None, help="evals/<version> dir: inventory.json recorded in provenance")
    args = ap.parse_args()

    if args.convert:
        src, dst = args.convert
        rows = load_train_rows([src], args.mapping)
        with open(dst, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"wrote {len(rows)} rows to {dst}")
        return
    assert args.run_name, "--run-name is required"
    assert args.max_length <= 32768
    assert args.brier_coef == 0 or args.single_label_loss == "softmax_ce", (
        "--brier-coef needs --single-label-loss softmax_ce"
    )
    assert args.rps_coef == 0 or args.single_label_loss == "softmax_ce", (
        "--rps-coef needs --single-label-loss softmax_ce (the bce path never applies RPS)"
    )
    args.eval_max_length = args.eval_max_length or args.max_length
    args.eval_attn = args.eval_attn or args.attn
    args.final_step = 0
    args.eval_steps_set = sorted(int(x) for x in args.eval_steps.split(",") if x)
    if args.dev_every and args.save_every and args.dev_every % args.save_every:
        raise SystemExit("--dev-every must be a multiple of --save-every, so every dev step has a checkpoint")
    if args.dev_holdout and not args.dev_every:
        raise SystemExit("--dev-holdout needs --dev-every")
    args.evals_by_step = {}

    from transformers import set_seed

    set_seed(args.seed)
    t_start = time.time()
    out = Path(args.out) / args.run_name
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    prov = provenance_start(args, out)
    if args.base_model:
        model, tokenizer = load_model(args.base_model)
        if getattr(model, "calibrator", None) is not None and (args.max_steps or 0) > 0:
            model.remove_calibrator()  # gliclass train.py does the same before further training
    else:
        model, tokenizer = build_model(args)
    model.to(DEVICE)
    attn = set_attention(model, args.attn)
    tracker_init(args)
    tok_info = check_tokenizer(model, tokenizer)
    if (args.max_steps or 0) > 0 or (args.max_steps is None and args.train):
        model.config.problem_type = "multi_label_classification"
        model.config.focal_loss_alpha, model.config.focal_loss_gamma = (
            args.focal_alpha,
            args.focal_gamma,
        )
        model.config.focal_loss_reduction = "mean"
    gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    n_params = sum(p.numel() for p in model.parameters())
    import gliclass
    import transformers

    metrics = {
        "run": args.run_name,
        "encoder": args.encoder,
        "base_model": args.base_model,
        "train_attn": attn,
        "params": n_params,
        "gpu": gpu,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gliclass": gliclass.__version__,
        "tokenizer": tok_info,
    }
    print(
        "setup",
        json.dumps({k: v for k, v in metrics.items() if k != "tokenizer"}),
        flush=True,
    )

    if args.bench_lengths:
        if args.grad_ckpt:
            model.model.encoder_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        metrics["bench"] = bench(
            model, args, [int(x) for x in args.bench_lengths.split(",")]
        )
        if args.grad_ckpt:
            model.model.encoder_model.gradient_checkpointing_disable()
    if args.train and args.max_steps != 0:
        rows = load_train_rows(args.train, args.mapping, args.sample, args.sample_seed)
        metrics["targets"] = mark_targets(rows, args.ordinal_sigma)
        if args.teacher_blend < 1.0 or args.use_row_blend:
            assert args.teacher_scores, (
                "--teacher-blend < 1 / --use-row-blend need --teacher-scores"
            )
            w = (
                [float(x) for x in args.teacher_weights.split(",")]
                if args.teacher_weights
                else None
            )
            metrics["teacher"] = apply_teacher(
                rows, args.teacher_scores, w, args.teacher_blend, args.use_row_blend
            )
        if args.dev_holdout:
            import hashlib

            def id_hash(r):
                return hashlib.sha256(str(r.get("id")).encode()).hexdigest()

            ids = [r.get("id") for r in rows]
            if any(i is None or i == "" for i in ids) or len(set(map(str, ids))) != len(ids):
                raise SystemExit("--dev-holdout needs a unique, non-empty id on every train row (review P2)")
            if args.dev_holdout >= len(rows):
                raise SystemExit("--dev-holdout must be smaller than the number of train rows")
            dev_ids = set(sorted((id_hash(r) for r in rows))[: args.dev_holdout])
            args.dev_rows = [r for r in rows if id_hash(r) in dev_ids]
            rows = [r for r in rows if id_hash(r) not in dev_ids]
            args.dev_rows, _ = drop_label_overflow(args.dev_rows, tokenizer, args.max_length)
            with open(out / "dev_ids.txt", "w") as f:
                f.writelines(f"{r.get('id')}\n" for r in args.dev_rows)
            prov["dev_rows"] = len(args.dev_rows)
            prov["dev_ids_sha256"] = sha256_file(out / "dev_ids.txt")
            print(f"dev holdout: {len(args.dev_rows)} rows held out of training (asked {args.dev_holdout}; "
                  "fewer only if label overflow dropped some)", flush=True)
        if args.distractors > 0 or args.neg_subset > 0:
            metrics["label_aug"] = augment_label_sets(
                rows,
                args.distractors,
                args.distractor_max_labels,
                args.neg_subset,
                args.neg_subset_min_labels,
                args.seed + 7919,
            )
        with open(out / "train_ids.txt", "w") as f:  # the rows this run saw (before augmentation)
            f.writelines(f"{r.get('id')}\n" for r in rows)
        prov["train_rows"] = len(rows)
        prov["train_ids_sha256"] = sha256_file(out / "train_ids.txt")
        rows, metrics["label_overflow_dropped"] = drop_label_overflow(rows, tokenizer, args.max_length)
        prov["label_overflow_dropped"] = metrics["label_overflow_dropped"]
        metrics["train"] = train(model, tokenizer, rows, args, out)
        print("train_summary", json.dumps(metrics["train"]), flush=True)
        args.final_step = metrics["train"]["steps"]
        metrics["evals_by_step"] = {str(k): v for k, v in args.evals_by_step.items()}
    metrics["eval_attn"] = set_attention(model, args.eval_attn)
    metrics["eval"], cache = (
        run_evals(model, tokenizer, args, out) if args.eval else ({}, {})
    )
    if args.save_model:
        save_dir = out / "model"
        save_model(model, tokenizer, save_dir)
        print("saved", save_dir, flush=True)
        if cache:
            (out / "metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )  # kept if the check fails
            metrics["reload_check"] = reload_check(save_dir, args, cache)
    metrics["wall_seconds"] = time.time() - t_start
    prov["final_step"] = args.final_step
    prov["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        "metrics",
        json.dumps({k: v for k, v in metrics.items() if k not in ("tokenizer",)}),
        flush=True,
    )
    tracker_finish()
    if not args.no_sync:
        from huggingface_hub import sync_bucket

        sync_bucket(str(out), f"{args.out_bucket}/{args.run_name}")
        print("synced to", f"{args.out_bucket}/{args.run_name}", flush=True)


if __name__ == "__main__":
    main()
