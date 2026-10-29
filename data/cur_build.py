# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets", "numpy", "huggingface_hub>=1.21"]
# ///
"""Build the curation training slice and the curation eval sets, as decision
rows ({id, state, question{type, instructions, criteria}, target}, the engine's request shape).

Train slice (real web text, LLM labels; teacher recorded per row):
  - WebOrganizer topic + format, Llama-3.1-405B annotations, `train` split: soft targets from choice_probs over the
    24 labels; criteria = short names or "name: description" (50/50); several instruction wordings.
  - FineWeb-Edu Llama-3-70B annotations (0-5 score): one-hot on 6 levels; short level names or the rubric (50/50).
    The 2,000 documents of the 10-23 Edge bench sample are excluded.
  Essential-Web is NOT trained on: it is the zero-shot transfer test (unseen taxonomies).
Eval sets: the exact samples of the 10-23 Edge zero-shot bench (schema_bench_v2.py: same datasets, splits, shuffle seed
0, n=2,000, sorted short labels, prompt "Classify the <axis> of this web page."), so scores compare directly with
Edge: WebOrganizer topic/format test (+ described variants), FineWeb-Edu, Essential-Web 5 axes.
Licences: FineWeb-Edu annotations and Essential-Web ODC-By; WebOrganizer annotation cards state no licence (text from
DCLM RefinedWeb; labels from Llama-3.1-405B). Released labels: see the dataset card.
usage: python cur_build.py --out /b/data/stage4/cur [--n-topic 20000 --n-format 20000 --n-edu 12000 --n-eval 2000]
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset

MAX_CHARS = 3000  # ~700 tokens; the Edge bench read 512 tokens
EDU_SHORT = [
    "not educational",
    "minimal educational value",
    "some educational value",
    "useful for education",
    "highly educational",
    "outstanding educational value",
]
EDU_RUBRIC = [
    "no educational value",
    "some basic information relevant to education, mixed with ads or irrelevant content",
    "touches on educational topics but is disorganised or not aligned with school standards",
    "introduces key concepts relevant to school curricula, coherent but limited",
    "highly relevant for teaching at grade school level, clear and consistent",
    "outstanding educational value, perfectly suited for teaching at primary or grade school level",
]
ESSENTIAL_AXES = {
    "document_type_v2": "document type",
    "fdc_level_1": "subject area",
    "reasoning_depth": "depth of reasoning",
    "extraction_artifacts": "text extraction quality",
    "education_level": "education level",
}
INSTRUCTIONS = {
    "topic": [
        "Classify the topic of this web page.",
        "What is the main topic of this web page?",
        "Which topic best describes this document?",
        "Assign this web page to one topic category.",
    ],
    "format": [
        "Classify the format of this web page.",
        "What kind of web page or document format is this?",
        "Which format or genre best describes this page?",
        "Identify the type of document this web page is.",
    ],
    "edu": [
        "Classify the educational value of this web page.",
        "How useful is this web page for teaching at primary to grade school level?",
        "Rate the educational value of this document.",
        "How educational is this web page?",
    ],
}


def short_name(label):
    return label.split("\n")[0].strip()


def key(text):
    return hashlib.sha1(text[:2000].encode()).hexdigest()


def wo_label_set(kind):
    """24 full labels ("Name\\nDescription") in canonical (sorted short-name) order, from the validation split."""
    full = {}
    for row in load_dataset(
        f"WebOrganizer/{kind}Annotations-Llama-3.1-405B-FP8", split="validation"
    ):
        full[short_name(row["top_choice"])] = row["top_choice"]
        if len(full) == 24:
            break
    names = sorted(full)
    return names, [full[n] for n in names]


def described(full):
    name, _, desc = full.partition("\n")
    desc = " ".join(desc.split())
    return f"{name.strip()}: {desc}" if desc else name.strip()


def wo_index_map(ds):
    """WebOrganizer index -> short name, from top_choice_index / top_choice; checks that choice_probs uses the same
    index (argmax(choice_probs) == top_choice_index on >= 97 % of 2k rows)."""
    idx_name, agree, n = {}, 0, 0
    for i, r in enumerate(ds):
        idx_name[int(r["top_choice_index"])] = short_name(r["top_choice"])
        if i < 2000:
            agree += int(np.argmax(r["choice_probs"])) == int(r["top_choice_index"])
            n += 1
        elif len(idx_name) == 24:
            break
    print(
        f"index map {len(idx_name)} labels; argmax(choice_probs)==top_choice_index on {agree}/{n}",
        flush=True,
    )
    assert len(idx_name) == 24 and agree >= 0.97 * n, (len(idx_name), agree, n)
    return idx_name


def wo_rows(kind, split_n, names, fulls, rng, exclude):
    ds = load_dataset(
        f"WebOrganizer/{kind}Annotations-Llama-3.1-405B-FP8", split="train"
    )
    idx_name = wo_index_map(ds)
    col = {n: k for k, n in enumerate(names)}
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out, tag = [], kind.lower()
    for i in idx:
        if len(out) >= split_n:
            break
        r = ds[i]
        if key(r["text"]) in exclude:
            continue
        probs = np.zeros(24)
        for j, v in enumerate(r["choice_probs"]):
            probs[col[idx_name[j]]] += float(v)
        if probs.sum() <= 0:
            continue
        probs /= probs.sum()
        use_desc = rng.random() < 0.5
        criteria = {n: (described(f) if use_desc else n) for n, f in zip(names, fulls)}
        target = {n: round(float(v), 4) for n, v in zip(names, probs) if v >= 1e-3}
        out.append(
            {
                "id": f"cur-wo-{tag}-{i}",
                "src": f"weborganizer_{tag}_405b",
                "teacher": "Llama-3.1-405B-FP8 (WebOrganizer)",
                "state": r["text"][:MAX_CHARS],
                "question": {
                    "type": "choice",
                    "instructions": rng.choice(INSTRUCTIONS[tag]),
                    "criteria": criteria,
                },
                "target": target,
            }
        )
    return out


def edu_sample_keys(n):
    """The 10-23 bench sample: train split, shuffle(seed=0), first n."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu-llama3-annotations", split="train")
    ev = ds.shuffle(seed=0).select(range(n))
    return ds, ev


def edu_rows(ds, n, rng, exclude):
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        if len(out) >= n:
            break
        r = ds[i]
        if key(r["text"]) in exclude:
            continue
        score = int(r["score"])
        labels = EDU_RUBRIC if rng.random() < 0.5 else EDU_SHORT
        criteria = {f"level_{k}": lab for k, lab in enumerate(labels)}
        out.append(
            {
                "id": f"cur-fwe-{i}",
                "src": "fineweb_edu_llama3",
                "teacher": "Llama-3-70B-Instruct (FineWeb-Edu)",
                "state": r["text"][:MAX_CHARS],
                "question": {
                    "type": "choice",
                    "instructions": rng.choice(INSTRUCTIONS["edu"]),
                    "criteria": criteria,
                },
                "target": {f"level_{score}": 1.0},
            }
        )
    return out


def eval_rows(name, texts, gold, labels, axis, criteria_text=None):
    """Eval rows in the 10-23 bench form: sorted labels, prompt "Classify the <axis> of this web page."."""
    rows = []
    for i, (t, g) in enumerate(zip(texts, gold)):
        crit = {
            lab: (criteria_text[k] if criteria_text else lab)
            for k, lab in enumerate(labels)
        }
        rows.append(
            {
                "id": f"{name}-{i}",
                "state": t[:MAX_CHARS],
                "question": {
                    "type": "choice",
                    "instructions": f"Classify the {axis} of this web page.",
                    "criteria": crit,
                },
                "target": {g: 1.0},
            }
        )
    return rows


def essential_eval(n):
    ds = load_dataset("EssentialAI/essential-web-v1.0", split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=20000)
    rows = list(ds.take(n))
    out = {}
    for axis in ESSENTIAL_AXES:
        texts, gold, codes = [], [], {}
        for row in rows:
            tax = row["eai_taxonomy"]
            if axis == "fdc_level_1":
                label = tax["free_decimal_correspondence"]["primary"]["labels"][
                    "level_1"
                ]
                code = None
            else:
                label = tax[axis]["primary"]["label"]
                code = tax[axis]["primary"]["code"]
                codes[label] = code
            if not label or label == "Abstain" or code == "-1":
                continue
            texts.append(row["text"])
            gold.append(label)
        labels = sorted(set(gold))
        if axis in {"reasoning_depth", "education_level"}:
            labels = sorted(labels, key=lambda x: int(codes[x]))
        out[axis] = (texts, gold, labels)
    return out


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-topic", type=int, default=20000)
    ap.add_argument("--n-format", type=int, default=20000)
    ap.add_argument("--n-edu", type=int, default=12000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    stats = {}

    # eval sets first (their documents are excluded from training)
    exclude = set()
    for kind, axis in (("Topic", "topic"), ("Format", "format")):
        names, fulls = wo_label_set(kind)
        ev = load_dataset(
            f"WebOrganizer/{kind}Annotations-Llama-3.1-405B-FP8", split="test"
        )
        ev = ev.shuffle(seed=0).select(range(args.n_eval))
        gold = [short_name(x) for x in ev["top_choice"]]
        exclude |= {key(t) for t in ev["text"]}
        write(
            out / f"eval_wo_{axis}.jsonl",
            eval_rows(f"wo_{axis}", ev["text"], gold, names, axis),
        )
        write(
            out / f"eval_wo_{axis}_desc.jsonl",
            eval_rows(
                f"wo_{axis}_desc",
                ev["text"],
                gold,
                names,
                axis,
                [described(f) for f in fulls],
            ),
        )
        train = wo_rows(
            kind,
            args.n_topic if axis == "topic" else args.n_format,
            names,
            fulls,
            rng,
            exclude,
        )
        stats[f"wo_{axis}"] = len(train)
        write(out / f"train_wo_{axis}_decision.jsonl", train)

    edu_ds, edu_ev = edu_sample_keys(args.n_eval)
    exclude |= {key(t) for t in edu_ev["text"]}
    gold = [EDU_SHORT[int(s)] for s in edu_ev["score"]]
    write(
        out / "eval_fwe.jsonl",
        eval_rows("fwe", edu_ev["text"], gold, EDU_SHORT, "educational value"),
    )
    write(
        out / "eval_fwe_desc.jsonl",
        eval_rows(
            "fwe_desc", edu_ev["text"], gold, EDU_SHORT, "educational value", EDU_RUBRIC
        ),
    )
    train = edu_rows(edu_ds, args.n_edu, rng, exclude)
    stats["fwe"] = len(train)
    write(out / "train_fwe_decision.jsonl", train)

    for axis, (texts, gold, labels) in essential_eval(args.n_eval).items():
        write(
            out / f"eval_eai_{axis}.jsonl",
            eval_rows(f"eai_{axis}", texts, gold, labels, ESSENTIAL_AXES[axis]),
        )
        stats[f"eval_eai_{axis}"] = len(texts)

    # one training file for the screen (sg_screen.py)
    allrows = []
    for f in (
        "train_wo_topic_decision.jsonl",
        "train_wo_format_decision.jsonl",
        "train_fwe_decision.jsonl",
    ):
        allrows += [json.loads(x) for x in open(out / f)]
    rng.shuffle(allrows)
    write(out / "cur_train_decision.jsonl", allrows)
    stats["train_total"] = len(allrows)
    (out / "cur_build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
