# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "huggingface_hub>=1.21"]
# ///
"""Stage-1 general decision data for the UltraBERT decision model, with an overlap check.

Sources: tasksource/tasksource-jev-typed-decisions (default config, train split, commercial-use rows) and
knowledgator/gliclass-v2.0 (converted to choice / noul questions). Output rows use the dm_train.py JSONL format.

Removed before sampling (by source name, tasksource only):
  - our eval tasks: github / kubernetes / issue, ragtruth, jev / opendecisions;
  - the Decision Index public suite (37 benchmarks, kit apolinario/decision-index docs/suite.md).
Removed after sampling (by text, both sources), against every eval and calib file in data/prepared:
  - exact match of the normalised state;
  - any shared 13-word shingle between the state and an eval state (states of 13+ words);
  - for OpenDecisions: exact match of the question text plus the same option set.
Writes <out>.jsonl and <out>_overlap.json (counts per reason and per source).
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import polars as pl

OUR_TASKS = r"github|kubernetes|issue|ragtruth|\bjev|opendecision|jb16"
DECISION_INDEX = (
    r"mmlu|\bbbh\b|big.?bench.?hard|hover|when2call|newyorker|new.?yorker|phish|\bbfcl|toolret|api.?bank|banking77"
    r"|clinc|routerbench|schema.?guided|\bsgd\b|contract.?nli|\banli\b|bpomp|humicroedit|pop909|cfcolor|gpqa|ai2_arc"
    r"|winogrande|hellaswag|gsm8k|chess|musr|sata.?bench|simplebench|\bbright\b|\besci\b|\bacos\b|finentity|isarcasm"
    r"|\bvast\b|nli4ct|cruxeval|cladder|\bhle\b|forecastbench|habermas"
)
EVAL_FILES = [
    "opendecisions_eval", "calib_ts_1500", "k8s_test", "k8s_calib", "k8s_dev", "rt_test", "rt_calib", "bl_test",
    "bl_calib", "hv_lf_test", "hv_lf_calib", "hv_if_test", "hv_if_calib", "hv_sd_test", "hv_sd_calib",
]
N = 13
# calib_ts_1500 is tasksource validation data (OpenDecisions temperature fitting only). Its templated sources share
# 13-word boilerplate with their own train rows, so only exact state matches count against it.
SHINGLE_EXEMPT = {"calib_ts_1500"}


def words(text):
    return re.findall(r"\w+", text.lower())


def norm(text):
    return " ".join(words(text))


def shingles(ws):
    return {hashlib.blake2b(" ".join(ws[i : i + N]).encode(), digest_size=8).digest() for i in range(len(ws) - N + 1)}


def state_text(state):
    if isinstance(state, str):
        return state
    return "\n".join(f"{k}: {v if isinstance(v, str) else json.dumps(v)}" for k, v in state.items())


def ts_convert(row):
    """tasksource row -> dm_train question (as in prep_tasksource.convert)."""
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


def gc_convert(row, i, rng):
    """gliclass-v2.0 row -> one question. One true label among 2-16 labels: a choice. Otherwise: a noul question
    about one label (a true label with probability 0.5 when there is one)."""
    labels = [label for label in dict.fromkeys(row["all_labels"] or []) if label]
    true = [label for label in (row["true_labels"] or []) if label in labels]
    if not labels or not row["text"]:
        return None
    base = {"id": f"gliclass-v2.0-{i}", "state": row["text"], "source": "gliclass-v2.0"}
    if len(true) == 1 and 2 <= len(labels) <= 16:
        crit = {str(j + 1): label for j, label in enumerate(labels)}
        key = str(labels.index(true[0]) + 1)
        return {**base, "question": {"type": "choice", "instructions": "Which label fits the text best?",
                                     "criteria": crit}, "target": {key: 1.0}}
    if true and (rng.random() < 0.5 or len(true) == len(labels)):
        label, value = rng.choice(true), 1.0
    else:
        rest = [label for label in labels if label not in true]
        if not rest:
            return None
        label, value = rng.choice(rest), 0.0
    return {**base, "question": {"type": "noul", "instructions": f"Does this statement hold for the text? {label}",
                                 "criteria": {}}, "target": {"true": value, "false": 1 - value}}


def option_set(question):
    c = question["criteria"]
    return frozenset(norm(str(x)) for x in (c.values() if isinstance(c, dict) else c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="/b/data/prepared")
    ap.add_argument("--out", default="/b/data/prepared/stage1")
    ap.add_argument("--n-ts", type=int, default=400_000)
    ap.add_argument("--n-gc", type=int, default=150_000)
    ap.add_argument("--per-source-cap", type=int, default=2000)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--ts-files", default="hf://datasets/tasksource/tasksource-jev-typed-decisions/data/train-*.parquet")
    ap.add_argument("--gc-files", default="hf://datasets/knowledgator/gliclass-v2.0/data/train-*.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    prepared = Path(args.prepared)

    eval_states, eval_shingles, od_questions = {}, {}, set()
    for name in EVAL_FILES:
        for line in open(prepared / f"{name}.jsonl"):
            item = json.loads(line)
            ws = words(state_text(item["state"]))
            eval_states.setdefault(" ".join(ws), name)
            if name in SHINGLE_EXEMPT:
                continue
            for sh in shingles(ws):
                eval_shingles.setdefault(sh, name)
            if name == "opendecisions_eval":
                od_questions.add((norm(item["question"]["instructions"]), option_set(item["question"])))
    print("eval states", len(eval_states), "shingles", len(eval_shingles), flush=True)

    report = {"source_patterns": {"our_tasks": OUR_TASKS, "decision_index": DECISION_INDEX}}

    # ---- tasksource
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def read(pattern):
        if not pattern.startswith("hf://"):
            return pl.read_parquet(pattern)
        return pl.concat([pl.read_parquet(fs.open(p, "rb").read()) for p in sorted(fs.glob(pattern[5:]))],
                         how="diagonal_relaxed")

    df = read(args.ts_files)
    report["ts_rows"] = df.height
    report["ts_by_license"] = dict(Counter(df["license_use"].to_list()))
    df = df.filter(pl.col("license_use") == "commercial")
    src = pl.col("source").str.to_lowercase()
    ours = df.filter(src.str.contains(OUR_TASKS))
    di = df.filter(~src.str.contains(OUR_TASKS) & src.str.contains(DECISION_INDEX))
    report["ts_removed_our_tasks"] = {"rows": ours.height, "sources": sorted(set(ours["source"].to_list()))}
    report["ts_removed_decision_index"] = {"rows": di.height, "sources": sorted(set(di["source"].to_list()))}
    df = df.filter(~src.str.contains(OUR_TASKS) & ~src.str.contains(DECISION_INDEX))
    df = df.filter(pl.col("state").str.len_chars() <= args.max_chars)
    rows = df.sample(fraction=1.0, shuffle=True, seed=args.seed).to_dicts()
    del df
    per, ts = Counter(), []
    for r in rows:
        if per[r["source"]] >= args.per_source_cap:
            continue
        c = ts_convert(r)
        if c is not None:
            per[r["source"]] += 1
            ts.append(c)
    del rows
    report["ts_after_filters_and_cap"] = len(ts)
    report["ts_sources"] = len(per)

    # ---- gliclass-v2.0
    gdf = read(args.gc_files)
    report["gc_rows"] = gdf.height
    grows = gdf.sample(fraction=1.0, shuffle=True, seed=args.seed).head(3 * args.n_gc).to_dicts()
    del gdf
    gc = [c for i, r in enumerate(grows) if (c := gc_convert(r, i, rng)) is not None]
    del grows

    # ---- text overlap, then take the first N clean rows of each source
    def clean(items, n, tag):
        kept, removed, removed_src, removed_file = [], Counter(), Counter(), Counter()
        for it in items:
            if len(kept) >= n:
                break
            ws = words(state_text(it["state"]))
            reason, hit = None, None
            if " ".join(ws) in eval_states:
                reason, hit = "exact_state", eval_states[" ".join(ws)]
            elif len(ws) >= N:
                hit = next((eval_shingles[sh] for sh in shingles(ws) if sh in eval_shingles), None)
                reason = "shingle13" if hit else None
            if reason is None and (norm(it["question"]["instructions"]), option_set(it["question"])) in od_questions:
                reason, hit = "od_question_and_options", "opendecisions_eval"
            if reason:
                removed[reason] += 1
                removed_src[it["source"]] += 1
                removed_file[hit] += 1
                continue
            kept.append(it)
        report[f"{tag}_text_overlap_removed"] = dict(removed)
        report[f"{tag}_text_overlap_by_source"] = dict(removed_src.most_common(50))
        report[f"{tag}_text_overlap_by_eval_file"] = dict(removed_file)
        report[f"{tag}_checked"] = len(kept) + sum(removed.values())
        report[f"{tag}_kept"] = len(kept)
        return kept

    ts = clean(ts, args.n_ts, "ts")
    gc = clean(gc, args.n_gc, "gc")
    out = ts + gc
    rng.shuffle(out)
    report["kinds"] = dict(Counter(it["question"]["type"] for it in out))
    report["total"] = len(out)
    with open(f"{args.out}.jsonl", "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    Path(f"{args.out}_overlap.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if "sources" not in k and "by_source" not in k}, indent=1))


if __name__ == "__main__":
    main()
