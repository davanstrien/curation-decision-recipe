# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "huggingface_hub>=1.21"]
# ///
"""G arm: reasoning-core/procedural-pile (CC BY 4.0; program-generated tasks, verifier answers) -> decision rows.

Only tasks with a small closed answer set that the prompt itself states are used, so the options are exactly the
answers the task allows and the label is the program's answer (target_source "program"). Per task, the gold classes
are balanced (each class capped at the rarest kept class), so a label-prior / options-only baseline is at chance.
Option sets: fixed per task, or parsed from the prompt (qualitative_causal_reasoning "Answer with: ...",
game_best_move "Legal player moves now: ..."). State = the full task prompt; instructions = a short varied line
(the prompt carries the answer format). Rows whose answer is not in the option set are dropped.
Output: {id, src: "procpile:<task>", licence, family, state, question{type choice, instructions, criteria}, target,
target_source}.
usage: uv run g_procpile.py OUT.jsonl [--shards 0,1] [--per-task 1000] [--seed 0]
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict

import polars as pl

REPO = "hf://datasets/reasoning-core/procedural-pile/data"
YN, YNM = ["Yes", "No"], ["Yes", "No", "Maybe"]
FIXED = {
    "multistep_nli": YNM,
    "defeasible_nli": YNM,
    "game_forced_win": YN,
    "table_equivalence": YN,
    "unification_entailment": YN,
    "regex_reasoning": YN,
    "code_runnability": [
        "OK",
        "ZeroDivisionError",
        "IndexError",
        "ValueError",
        "KeyError",
    ],
    "metamath_entailment": ["A", "B"],
    "most_probable_outcome": ["A", "B", "equal"],
    "metamath_core_select": ["A", "B", "C", "D"],
    "constraint_satisfaction": ["SAT", "UNSAT"],
}
PARSED = {
    "qualitative_causal_reasoning": re.compile(r"Answer with:\s*(.+?)\.?\s*$", re.S),
    "game_best_move": re.compile(r"Legal player moves now:\s*([^\n]+?)\.?\s*\n"),
}
INSTRUCTIONS = [
    "",
    "Choose the correct answer.",
    "Pick the answer the rules imply.",
    "Which answer is correct?",
    "Select the option that follows from the problem.",
]


def options_for(task, prompt):
    if task in FIXED:
        if task == "constraint_satisfaction" and "SAT or UNSAT" not in prompt:
            return None
        return FIXED[task]
    m = PARSED[task].search(prompt)
    if not m:
        return None
    opts = [o.strip() for o in re.split(r",|\bor\b", m.group(1)) if o.strip()]
    return opts if len(opts) >= 2 and len(set(opts)) == len(opts) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shards", default="0")
    ap.add_argument("--per-task", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tasks = set(FIXED) | set(PARSED)
    files = [f"{REPO}/train-{int(s):05d}.parquet" for s in args.shards.split(",")]
    df = pl.concat(
        [
            pl.read_parquet(f, columns=["task", "prompt", "answer", "level"])
            for f in files
        ]
    )
    df = df.filter(pl.col("task").is_in(sorted(tasks)))
    by_task = defaultdict(lambda: defaultdict(list))
    dropped = Counter()
    for i, r in enumerate(df.iter_rows(named=True)):
        opts = options_for(r["task"], r["prompt"])
        ans = str(r["answer"]).strip()
        if opts is None or ans not in opts:
            dropped[r["task"]] += 1
            continue
        by_task[r["task"]][ans].append((i, r, opts))
    stats = {}
    with open(args.out, "w") as f:
        for task in sorted(by_task):
            classes = by_task[task]
            # keep classes with >= 5% of the rows; cap every class at the rarest kept class
            total = sum(len(v) for v in classes.values())
            kept = {a: v for a, v in classes.items() if len(v) >= 0.05 * total}
            per_class = min(
                min(len(v) for v in kept.values()), args.per_task // len(kept)
            )
            n = 0
            for ans, items in sorted(kept.items()):
                for i, r, opts in rng.sample(items, per_class):
                    criteria = list(opts)
                    rng.shuffle(criteria)
                    row = {
                        "id": f"procpile-{task}-{i}",
                        "src": f"procpile:{task}",
                        "licence": "cc-by-4.0",
                        "family": f"procpile.{task}",
                        "state": r["prompt"],
                        "question": {
                            "type": "choice",
                            "instructions": rng.choice(INSTRUCTIONS),
                            "criteria": criteria,
                        },
                        "target": {ans: 1.0},
                        "target_source": "program",
                        "level": r["level"],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
            stats[task] = {
                "rows": n,
                "classes": sorted(kept),
                "per_class": per_class,
                "dropped": dropped[task],
            }
    print(json.dumps(stats, indent=1))
    print("total", sum(s["rows"] for s in stats.values()))


if __name__ == "__main__":
    main()
