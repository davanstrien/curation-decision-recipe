# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets", "huggingface_hub>=1.21"]
# ///
"""Curation pivot, prompt-curation slice: SFT / RL prompt tagging as decision rows (engine request shape).

Train (all permissive; labels from the datasets' own columns, teacher recorded per row):
  - AI-MO/NuminaMath-1.5 (Apache-2.0): question type (word problem / proof / MCQ / other), problem validity
    (valid / incomplete / several problems / not a problem), problem type (8 domains).
  - TIGER-Lab/WebInstruct-verified (Apache-2.0): answer type (11), subject category, difficulty (5 ordinal levels).
  - zwhe99/DeepMath-103K (MIT): difficulty (5 ordinal bins of the 1-10 rating), topic (level 2 of the hierarchy).
  - lmarena-ai/arena-human-preference-140k (CC-BY-4.0), 90 % of prompts by id hash: the 7 "hard prompt" criteria
    (multi-label), and math / creative writing / instruction following (yes/no).
Eval (decision rows, never trained on):
  - in-distribution held-out: NuminaMath question type + validity, WebInstruct answer type + difficulty, DeepMath
    difficulty (1k each, rows excluded from training by id / text hash);
  - arena held-out 10 % (2k prompts): each criterion and tag as a yes/no question;
  - transfer: KbsdJames/Omni-MATH (Apache-2.0, unseen dataset and rating scale): difficulty (5 bins of its 1-10
    rating) and domain (level 2 of its hierarchy).
usage: python prompt_build.py --out /b/data/stage4/prompt
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

MAX_CHARS = 3000
QTYPE = {
    "math-word-problem": "A problem with a single final answer (a number, expression or value) that can be checked.",
    "proof": "A proof problem: asks to prove or show a statement; there is no single final answer to check.",
    "MCQ": "A multiple-choice question: the answer is one of the given options.",
    "other": "Something else: not a standard problem with a checkable answer or a proof.",
}
VALID = {
    "Yes": "A valid, complete, self-contained problem.",
    "Incomplete": "Incomplete: information, a figure or part of the statement is missing.",
    "More than one problem": "Several separate problems in one item.",
    "Not a problem": "Not a problem at all (for example a fragment, a heading or a statement with no question).",
}
LEVELS5 = ["very easy", "easy", "medium", "hard", "very hard"]
DIFF_PROMPTS = [
    "How difficult is this problem?",
    "Rate the difficulty of this question.",
    "Classify the difficulty of this problem for a strong student.",
]
CRITERIA = {
    "specificity": "Specificity: the prompt asks for a specific, well-defined output.",
    "domain_knowledge": "Domain knowledge: answering needs knowledge of a specific field.",
    "complexity": "Complexity: the prompt has several parts or layers of reasoning.",
    "problem_solving": "Problem solving: the prompt needs active problem solving, not recall.",
    "creativity": "Creativity: the prompt asks for a creative or original response.",
    "technical_accuracy": "Technical accuracy: the answer must be technically precise to be useful.",
    "real_world": "Real-world application: the prompt is about a practical, real-world task.",
}
TAGS = {
    "math": "Is this prompt a math question or task?",
    "creative_writing": "Is this prompt a creative writing request (story, poem, fiction, song)?",
    "if": "Does this prompt contain explicit instructions on the form of the answer (format, length, style, constraints)?",
}


def h(text):
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


def choice_row(rid, src, teacher, text, instructions, criteria, target):
    return {
        "id": rid,
        "src": src,
        "teacher": teacher,
        "state": text[:MAX_CHARS],
        "question": {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        },
        "target": target,
    }


def noul_row(rid, src, teacher, text, instructions, yes):
    return {
        "id": rid,
        "src": src,
        "teacher": teacher,
        "state": text[:MAX_CHARS],
        "question": {
            "type": "noul",
            "instructions": instructions,
            "criteria": {"true": "yes", "false": "no"},
        },
        "target": {"true" if yes else "false": 1.0},
    }


def bin5(x, edges):
    """edges: 4 thresholds -> level index 0..4"""
    return sum(x > e for e in edges)


def capped(rows, key, cap, rng):
    """Keep at most cap rows per label (rows shuffled first) so rare classes are not drowned."""
    rng.shuffle(rows)
    seen, out = Counter(), []
    for r in rows:
        k = key(r)
        if seen[k] < cap:
            seen[k] += 1
            out.append(r)
    return out


def numina(rng, n_scan, out_train, out_eval):
    ds = (
        load_dataset("AI-MO/NuminaMath-1.5", split="train")
        .shuffle(seed=0)
        .select(range(n_scan))
    )
    ptypes = sorted(
        {p for p in ds["problem_type"] if p and p not in ("NaN", "nan", "None")}
    )
    qt, va, pt = [], [], []
    for i, r in enumerate(ds):
        text = r["problem"] or ""
        if len(text) < 20:
            continue
        held = h(text) % 10 == 0
        teacher = "NuminaMath-1.5 columns"
        if r["question_type"] in QTYPE:
            qt.append(
                (
                    held,
                    choice_row(
                        f"nm-qt-{i}",
                        "numina_qtype",
                        teacher,
                        text,
                        "What kind of problem is this?",
                        QTYPE,
                        {r["question_type"]: 1.0},
                    ),
                )
            )
        if r["problem_is_valid"] in VALID:
            va.append(
                (
                    held,
                    choice_row(
                        f"nm-va-{i}",
                        "numina_valid",
                        teacher,
                        text,
                        "Is this a valid, self-contained problem?",
                        VALID,
                        {r["problem_is_valid"]: 1.0},
                    ),
                )
            )
        if r["problem_type"] in ptypes:
            crit = {p: p for p in ptypes}
            pt.append(
                (
                    held,
                    choice_row(
                        f"nm-pt-{i}",
                        "numina_ptype",
                        teacher,
                        text,
                        "Which area of mathematics is this problem from?",
                        crit,
                        {r["problem_type"]: 1.0},
                    ),
                )
            )
    for name, rows, cap in (
        ("numina_qtype", qt, 2500),
        ("numina_valid", va, 2500),
        ("numina_ptype", pt, 900),
    ):
        tr = capped(
            [r for held, r in rows if not held],
            lambda r: next(iter(r["target"])),
            cap,
            rng,
        )
        ev = [r for held, r in rows if held]
        rng.shuffle(ev)
        out_train += tr
        if name != "numina_ptype":
            out_eval[name] = capped(ev, lambda r: next(iter(r["target"])), 400, rng)[
                :1000
            ]
        print(
            name,
            "train",
            len(tr),
            Counter(next(iter(r["target"])) for r in tr).most_common(),
            flush=True,
        )


def webinstruct(rng, out_train, out_eval):
    ds = load_dataset("TIGER-Lab/WebInstruct-verified", split="train").shuffle(seed=0)
    atypes = sorted(
        {a for a in ds["answer_type"] if a and a not in ("NaN", "nan", "None")}
    )
    cats = sorted({c for c in ds["category"] if c and c not in ("NaN", "nan", "None")})
    order = [
        "Primary School",
        "Junior High School",
        "Senior High School",
        "University",
        "PhD",
    ]
    at, ca, di = [], [], []
    for i, r in enumerate(ds.select(range(min(60000, len(ds))))):
        text = r["question"] or ""
        held = h(text) % 10 == 0
        t = "WebInstruct-verified columns"
        if r["answer_type"] in atypes:
            at.append(
                (
                    held,
                    choice_row(
                        f"wi-at-{i}",
                        "wi_atype",
                        t,
                        text,
                        "What type of answer does this question expect?",
                        {a: a for a in atypes},
                        {r["answer_type"]: 1.0},
                    ),
                )
            )
        if r["category"] in cats:
            ca.append(
                (
                    held,
                    choice_row(
                        f"wi-ca-{i}",
                        "wi_category",
                        t,
                        text,
                        "Which subject is this question from?",
                        {c: c for c in cats},
                        {r["category"]: 1.0},
                    ),
                )
            )
        if r["difficulty"] in order:
            crit = {f"level_{k}": f"{lvl} level" for k, lvl in enumerate(order)}
            di.append(
                (
                    held,
                    choice_row(
                        f"wi-di-{i}",
                        "wi_difficulty",
                        t,
                        text,
                        "At what education level is this question?",
                        crit,
                        {f"level_{order.index(r['difficulty'])}": 1.0},
                    ),
                )
            )
    for name, rows, cap in (
        ("wi_atype", at, 400),
        ("wi_category", ca, 300),
        ("wi_difficulty", di, 1000),
    ):
        tr = capped(
            [r for held, r in rows if not held],
            lambda r: next(iter(r["target"])),
            cap,
            rng,
        )
        ev = [r for held, r in rows if held]
        rng.shuffle(ev)
        out_train += tr
        if name in ("wi_atype", "wi_difficulty"):
            out_eval[name] = ev[:1000]
        print(name, "train", len(tr), flush=True)


def deepmath(rng, out_train, out_eval):
    ds = (
        load_dataset("zwhe99/DeepMath-103K", split="train")
        .shuffle(seed=0)
        .select(range(40000))
    )
    di, to = [], []
    topics = Counter()
    for r in ds:
        parts = [p.strip() for p in (r["topic"] or "").split("->")]
        if len(parts) >= 2:
            topics[parts[1]] += 1
    keep_topics = sorted(t for t, c in topics.items() if c >= 200)
    for i, r in enumerate(ds):
        text = r["question"] or ""
        held = h(text) % 10 == 0
        t = "DeepMath-103K columns (GPT-4o rating)"
        lvl = bin5(float(r["difficulty"]), [3.5, 5.0, 6.5, 8.0])
        crit = {f"level_{k}": name for k, name in enumerate(LEVELS5)}
        di.append(
            (
                held,
                choice_row(
                    f"dm-di-{i}",
                    "dm_difficulty",
                    t,
                    text,
                    rng.choice(DIFF_PROMPTS),
                    crit,
                    {f"level_{lvl}": 1.0},
                ),
            )
        )
        parts = [p.strip() for p in (r["topic"] or "").split("->")]
        if len(parts) >= 2 and parts[1] in keep_topics:
            to.append(
                (
                    held,
                    choice_row(
                        f"dm-to-{i}",
                        "dm_topic",
                        t,
                        text,
                        "Which area of mathematics is this problem from?",
                        {x: x for x in keep_topics},
                        {parts[1]: 1.0},
                    ),
                )
            )
    for name, rows, cap in (("dm_difficulty", di, 1500), ("dm_topic", to, 600)):
        tr = capped(
            [r for held, r in rows if not held],
            lambda r: next(iter(r["target"])),
            cap,
            rng,
        )
        ev = [r for held, r in rows if held]
        rng.shuffle(ev)
        out_train += tr
        if name == "dm_difficulty":
            out_eval[name] = ev[:1000]
        print(
            name,
            "train",
            len(tr),
            "topics",
            keep_topics if name == "dm_topic" else "",
            flush=True,
        )


def first_user_text(conv):
    for turn in conv or []:
        if turn.get("role") == "user":
            return " ".join(
                c.get("text") or ""
                for c in turn.get("content") or []
                if c.get("type") == "text"
            ).strip()
    return ""


def arena(rng, n_train, out_train, out_eval):
    ds = load_dataset("lmarena-ai/arena-human-preference-140k", split="train").shuffle(
        seed=0
    )
    train, ev = [], defaultdict(list)
    seen = set()
    for i, r in enumerate(ds):
        if r.get("language") != "en":
            continue
        text = first_user_text(r["conversation_a"])
        if len(text) < 15 or text in seen:
            continue
        seen.add(text)
        tag = r["category_tag"] or {}
        crit = tag.get("criteria_v0.1") or {}
        flags = {
            "math": (tag.get("math_v0.1") or {}).get("math"),
            "creative_writing": (tag.get("creative_writing_v0.1") or {}).get(
                "creative_writing"
            ),
            "if": (tag.get("if_v0.1") or {}).get("if"),
        }
        t = "LMArena category tags (LLM judge)"
        held = h(text) % 10 == 0
        if held:
            if len(ev["arena_math"]) >= 2000:
                continue
            for k, d in CRITERIA.items():
                if crit.get(k) is not None:
                    ev[f"arena_{k}"].append(
                        noul_row(
                            f"ar-{k}-{i}",
                            "arena",
                            t,
                            text,
                            f"Does this prompt have this property? {d}",
                            bool(crit[k]),
                        )
                    )
            for k, q in TAGS.items():
                if flags[k] is not None:
                    ev[f"arena_{k}"].append(
                        noul_row(f"ar-{k}-{i}", "arena", t, text, q, bool(flags[k]))
                    )
            continue
        if len(train) >= n_train:
            continue
        # explicit 0/1 per criterion: multi-label (non-exclusive); a prompt with no criterion is an all-negative row
        target = {k: 1.0 if crit.get(k) else 0.0 for k in CRITERIA}
        row = choice_row(
            f"ar-cr-{i}",
            "arena_criteria",
            t,
            text,
            "Which of these properties does the prompt have? (any number)",
            dict(CRITERIA),
            target,
        )
        row["multi_label"] = True
        train.append(row)
        k = rng.choice(list(TAGS))
        if flags[k] is not None:
            train.append(
                noul_row(f"ar-{k}-{i}", "arena_tag", t, text, TAGS[k], bool(flags[k]))
            )
    out_train += train
    out_eval.update(ev)
    print("arena train", len(train), {k: len(v) for k, v in ev.items()}, flush=True)


def omni(out_eval):
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    dom = Counter()
    for r in ds:
        d = r["domain"]
        first = d[0] if isinstance(d, list) and d else str(d)
        parts = [p.strip() for p in first.split("->")]
        if len(parts) >= 2:
            dom[parts[1]] += 1
    keep = sorted(k for k, c in dom.items() if c >= 50)
    di, do = [], []
    for i, r in enumerate(ds):
        text = r["problem"] or ""
        lvl = bin5(float(r["difficulty"]), [2.5, 4.0, 5.5, 7.0])
        crit = {f"level_{k}": name for k, name in enumerate(LEVELS5)}
        di.append(
            choice_row(
                f"om-di-{i}",
                "omni",
                "Omni-MATH rating",
                text,
                "How difficult is this problem?",
                crit,
                {f"level_{lvl}": 1.0},
            )
        )
        d = r["domain"]
        first = d[0] if isinstance(d, list) and d else str(d)
        parts = [p.strip() for p in first.split("->")]
        if len(parts) >= 2 and parts[1] in keep:
            do.append(
                choice_row(
                    f"om-do-{i}",
                    "omni",
                    "Omni-MATH domain",
                    text,
                    "Which area of mathematics is this problem from?",
                    {x: x for x in keep},
                    {parts[1]: 1.0},
                )
            )
    out_eval["omni_difficulty"] = di
    out_eval["omni_domain"] = do
    print("omni", len(di), len(do), keep, flush=True)


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--numina-scan", type=int, default=120000)
    ap.add_argument("--arena-train", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train, ev = [], {}
    numina(rng, args.numina_scan, train, ev)
    webinstruct(rng, train, ev)
    deepmath(rng, train, ev)
    arena(rng, args.arena_train, train, ev)
    omni(ev)
    rng.shuffle(train)
    write(out / "prompt_train_decision.jsonl", train)
    for name, rows in ev.items():
        write(out / f"eval_pc_{name}.jsonl", rows)
    stats = {
        "train": len(train),
        "by_src": Counter(r["src"] for r in train),
        "eval": {k: len(v) for k, v in ev.items()},
    }
    (out / "prompt_build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
