# /// script
# requires-python = ">=3.11"
# dependencies = ["radon"]
# ///
"""Tool-oracle pilot: real Python functions + ruff / radon labels -> decision rows. No LLM, no model labels.

Code: functions from installed permissively licensed packages (--allow: package names; MIT / BSD / Apache / PSF;
licence read from package metadata, see pkg_licences.json). Each function is parsed with ast and rendered with
ast.unparse, so originals and mutants share one formatting (no style cue; comments dropped in both).
Mutations (one per mutant), each verified by ruff: the original has no finding in RULES and the mutant has exactly the
intended one.
  F841 unused local variable · B006 mutable default argument · E722 bare except · F541 f-string without placeholders ·
  E711 comparison to None with == / != · F632 `is` with a literal
Framings:
  lint_choice  "Which problem does a linter report?" options = 6 rule descriptions + "no problem" (balanced classes)
  lint_yesno   "Does this function have problem X?" (positive = mutant; negative = its lint-clean decoy, else original)
  lint_pair    "Which version passes the linter?" (decoy or original vs mutant, A/B shuffled)
  cc_pair      "Which function has the higher cyclomatic complexity?" (radon; matched length within 25 %,
               complexity gap >= --cc-gap)
usage: uv run tool_build.py SITE_PACKAGES OUT_DIR --allow requests,click,... [--per-package 400]
"""

import argparse
import ast
import copy
import json
import random
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from radon.complexity import cc_visit

RULES = {
    "F841": "A local variable is assigned but never used.",
    "B006": "A mutable value is used as a default argument.",
    "E722": "An except clause catches everything without naming an exception.",
    "F541": "An f-string has no placeholders.",
    "E711": "A value is compared to None with == or != instead of is.",
    "F632": "The `is` operator is used to compare with a literal.",
}
NO_PROBLEM = "The linter reports none of these problems."
CHOICE_Q = [
    "Which problem does a linter report for this function?",
    "What does ruff flag in this code?",
    "Pick the lint finding for this function.",
]
PAIR_Q = [
    "Which version of the function passes the linter?",
    "Which of the two versions has no lint finding?",
]
CC_Q = [
    "Which function has the higher cyclomatic complexity?",
    "Which of these functions has more decision paths?",
]


def functions(path, min_lines=6, max_lines=60):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n = (node.end_lineno or node.lineno) - node.lineno + 1
            if min_lines <= n <= max_lines:
                yield node


def body_start(fn):
    b = fn.body
    return (
        1
        if b
        and isinstance(b[0], ast.Expr)
        and isinstance(getattr(b[0], "value", None), ast.Constant)
        else 0
    )


def mutate(fn, rule, rng, decoy=False):
    """-> mutated copy of fn, or None when the rule does not apply. decoy=True makes the same surface edit in a
    lint-clean form (variable used later, None default, `except Exception`, f-string with a placeholder)."""
    m = copy.deepcopy(fn)
    names = sorted(
        {n.id for n in ast.walk(m) if isinstance(n, ast.Name)}
        | {a.arg for a in m.args.args}
    )
    if rule == "F841":
        base = rng.choice(names) if names else "value"
        new = f"{base}_{rng.choice(['tmp', 'copy', 'old', 'prev', 'cached'])}"
        if new in names:
            return None
        src = rng.choice(names) if names else "None"
        start = body_start(m)
        m.body.insert(start, ast.parse(f"{new} = {src}").body[0])
        if decoy:  # use the new variable: rewrite one later read of src
            later = [
                n
                for stmt in m.body[start + 1 :]
                for n in ast.walk(stmt)
                if isinstance(n, ast.Name)
                and n.id == src
                and isinstance(n.ctx, ast.Load)
            ]
            if not later:
                return None
            rng.choice(later).id = new
        return m
    if rule == "B006":
        if m.args.vararg or m.args.kwarg or m.args.kwonlyargs:
            return None
        m.args.args.append(
            ast.arg(
                arg=rng.choice(["options", "items", "seen", "extra", "cache"]) + "_arg"
            )
        )
        m.args.defaults.append(
            ast.parse("None" if decoy else rng.choice(["[]", "{}"])).body[0].value
        )
        return m
    if rule == "E722":
        i = body_start(m)
        if i >= len(m.body) or isinstance(m.body[i], (ast.Return, ast.Try)):
            return None
        stmt = m.body[i]
        m.body[i] = ast.Try(
            body=[stmt],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id="Exception", ctx=ast.Load()) if decoy else None,
                    name=None,
                    body=[ast.Pass()],
                )
            ],
            orelse=[],
            finalbody=[],
        )
        return m
    if rule == "F541":
        consts = [
            n
            for n in ast.walk(m)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value
            and "{" not in n.value
            and "}" not in n.value
            and len(n.value) < 60
        ]
        consts = [
            c
            for c in consts
            if c is not getattr(getattr(m.body[0], "value", None), "value", None)
            or True
        ]
        if body_start(m):
            consts = [c for c in consts if c is not m.body[0].value]
        if not consts:
            return None
        target = rng.choice(consts)
        if decoy and not names:
            return None
        piece = (
            [
                ast.Constant(value=target.value + " "),
                ast.FormattedValue(
                    value=ast.Name(id=rng.choice(names), ctx=ast.Load()), conversion=-1
                ),
            ]
            if decoy
            else [ast.Constant(value=target.value)]
        )
        for parent in ast.walk(m):
            for field, value in ast.iter_fields(parent):
                if value is target:
                    setattr(parent, field, ast.JoinedStr(values=piece))
                    return m
                if isinstance(value, list) and target in value:
                    value[value.index(target)] = ast.JoinedStr(values=piece)
                    return m
        return None
    if rule in ("E711", "F632") and decoy:
        return None
    if rule == "E711":
        cmps = [
            n
            for n in ast.walk(m)
            if isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and isinstance(n.ops[0], (ast.Is, ast.IsNot))
            and isinstance(n.comparators[0], ast.Constant)
            and n.comparators[0].value is None
        ]
        if not cmps:
            return None
        c = rng.choice(cmps)
        c.ops = [ast.Eq() if isinstance(c.ops[0], ast.Is) else ast.NotEq()]
        return m
    if rule == "F632":
        cmps = [
            n
            for n in ast.walk(m)
            if isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and isinstance(n.ops[0], (ast.Eq, ast.NotEq))
            and isinstance(n.comparators[0], ast.Constant)
            and isinstance(n.comparators[0].value, (int, str))
            and not isinstance(n.comparators[0].value, bool)
        ]
        if not cmps:
            return None
        c = rng.choice(cmps)
        c.ops = [ast.Is() if isinstance(c.ops[0], ast.Eq) else ast.IsNot()]
        return m
    raise ValueError(rule)


def render(fn):
    try:
        return ast.unparse(ast.fix_missing_locations(fn))
    except Exception:  # noqa: BLE001 - unparse can fail on odd trees; skip those functions
        return None


def ruff_codes(sources):
    """-> list of sets of RULES codes reported by ruff for each source string."""
    with tempfile.TemporaryDirectory() as d:
        for i, s in enumerate(sources):
            Path(d, f"f{i}.py").write_text(s + "\n")
        out = subprocess.run(
            [
                "ruff",
                "check",
                "--no-cache",
                "--isolated",
                "--select",
                ",".join(RULES),
                "--output-format",
                "json",
                d,
            ],
            capture_output=True,
            text=True,
        )
        found = [set() for _ in sources]
        for diag in json.loads(out.stdout or "[]"):
            i = int(Path(diag["filename"]).stem[1:])
            found[i].add(diag["code"])
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--allow", required=True)
    ap.add_argument("--per-package", type=int, default=400)
    ap.add_argument("--cc-gap", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    allow = [a.strip().lower().replace("-", "_") for a in args.allow.split(",")]
    originals = []  # (package, source, fn)
    for pkg in allow:
        roots = [
            p
            for p in args.site.iterdir()
            if p.name.lower() == pkg or p.name.lower() == f"{pkg}.py"
        ]
        fns = []
        for root in roots:
            files = [root] if root.is_file() else sorted(root.rglob("*.py"))
            for f in files:
                if "test" in f.parts[-1] or "/tests/" in str(f):
                    continue
                fns += list(functions(f))
        rng.shuffle(fns)
        kept = 0
        for fn in fns:
            src = render(fn)
            if src and len(src) < 2500:
                originals.append((pkg, src, fn))
                kept += 1
            if kept >= args.per_package:
                break
    print(f"originals {len(originals)} from {len(allow)} packages", flush=True)
    clean = ruff_codes([s for _, s, _ in originals])
    originals = [o for o, c in zip(originals, clean) if not c]
    print(f"clean originals {len(originals)}", flush=True)
    mutants = []  # (orig index, rule, mutant source, decoy source or None); one seed per pair -> same edit
    for i, (_, _, fn) in enumerate(originals):
        for rule in rng.sample(list(RULES), 2):
            seed = rng.random()
            m = mutate(fn, rule, random.Random(seed))
            d = mutate(fn, rule, random.Random(seed), decoy=True)
            ms = render(m) if m is not None else None
            ds = render(d) if d is not None else None
            if ms:
                mutants.append((i, rule, ms, ds))
    codes = ruff_codes([m for _, _, m, _ in mutants])
    dcodes = ruff_codes([d or "" for _, _, _, d in mutants])
    verified = [
        (i, r, m, d if d and not dc else None)
        for (i, r, m, d), c, dc in zip(mutants, codes, dcodes)
        if c == {r}
    ]
    print(
        f"mutants {len(mutants)}, verified {len(verified)} {dict(Counter(v[1] for v in verified))}; "
        f"decoys verified clean {sum(v[3] is not None for v in verified)}",
        flush=True,
    )
    rows = []

    def add(kind, rid, state, q, crit, gold, pkg, rule=None):
        rows.append(
            {
                "id": f"tool-{kind}-{rid}",
                "src": f"tool:{kind}",
                "licence": "permissive",
                "family": f"tool.{kind}",
                "state": state,
                "package": pkg,
                "question": {"type": "choice", "instructions": q, "criteria": crit},
                "target": {gold: 1.0},
                "target_source": "program",
                "meta": {"rule": rule},
            }
        )

    crit_choice = {**{k: v for k, v in RULES.items()}, "none": NO_PROBLEM}
    by_rule = defaultdict(list)
    for j, (i, r, s, _) in enumerate(verified):
        by_rule[r].append(j)
    per = min(len(v) for v in by_rule.values())
    for (
        r,
        js,
    ) in by_rule.items():  # lint_choice: balanced over the 6 rules + clean originals
        for j in rng.sample(js, per):
            i, _, s, _ = verified[j]
            keys = list(crit_choice)
            rng.shuffle(keys)
            add(
                "lint_choice",
                f"m{j}",
                s,
                rng.choice(CHOICE_Q),
                {k: crit_choice[k] for k in keys},
                r,
                originals[i][0],
                r,
            )
    # "none" class: clean originals and decoys (same surface edits as mutants, lint-clean)
    clean_pool = [
        ("o", k, originals[k][0], originals[k][1]) for k in range(len(originals))
    ]
    clean_pool += [
        ("d", j, originals[v[0]][0], v[3]) for j, v in enumerate(verified) if v[3]
    ]
    for tag, k, pkg, src in rng.sample(clean_pool, min(per, len(clean_pool))):
        keys = list(crit_choice)
        rng.shuffle(keys)
        add(
            "lint_choice",
            f"{tag}{k}",
            src,
            rng.choice(CHOICE_Q),
            {kk: crit_choice[kk] for kk in keys},
            "none",
            pkg,
            "none",
        )
    for j, (i, r, s, d) in enumerate(
        verified
    ):  # lint_yesno: mutant, or matched negative (decoy first)
        crit = {"yes": "Yes, the linter reports this.", "no": "No, it does not."}
        if rng.random() < 0.5:
            add(
                "lint_yesno",
                f"p{j}",
                s,
                f"Does a linter report this problem? {RULES[r]}",
                crit,
                "yes",
                originals[i][0],
                r,
            )
        else:
            add(
                "lint_yesno",
                f"n{j}",
                d or originals[i][1],
                f"Does a linter report this problem? {RULES[r]}",
                crit,
                "no",
                originals[i][0],
                r,
            )
    for j, (i, r, s, d) in enumerate(
        verified
    ):  # lint_pair: decoy vs mutant (same edit) when a decoy exists
        clean_src = d or originals[i][1]
        a_is_clean = rng.random() < 0.5
        a, b = (clean_src, s) if a_is_clean else (s, clean_src)
        add(
            "lint_pair",
            f"{j}",
            f"Version A:\n{a}\n\nVersion B:\n{b}",
            rng.choice(PAIR_Q),
            {"A": "Version A", "B": "Version B"},
            "A" if a_is_clean else "B",
            originals[i][0],
            r,
        )
    cc = []
    for i, (_, s, _) in enumerate(originals):  # cc_pair
        try:
            blocks = cc_visit(s)
        except Exception:  # noqa: BLE001 - radon parse failures: skip
            continue
        if blocks:
            cc.append((i, max(b.complexity for b in blocks), s.count("\n") + 1))
    rng.shuffle(cc)
    used = set()
    for a in cc:
        if a[0] in used:
            continue
        for b in cc:
            if b[0] in used or b[0] == a[0]:
                continue
            if abs(a[1] - b[1]) >= args.cc_gap and abs(a[2] - b[2]) <= 0.25 * max(
                a[2], b[2]
            ):
                first_high = rng.random() < 0.5
                hi, lo = (a, b) if a[1] > b[1] else (b, a)
                f1, f2 = (hi, lo) if first_high else (lo, hi)
                add(
                    "cc_pair",
                    f"{a[0]}-{b[0]}",
                    f"Function 1:\n{originals[f1[0]][1]}\n\nFunction 2:\n{originals[f2[0]][1]}",
                    rng.choice(CC_Q),
                    {"1": "Function 1", "2": "Function 2"},
                    "1" if first_high else "2",
                    originals[hi[0]][0],
                )
                used |= {a[0], b[0]}
                break
    rng.shuffle(rows)
    with open(args.out / "tool_decision.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    stats = {
        "originals": len(originals),
        "verified_mutants": len(verified),
        "rows": dict(Counter(r["src"] for r in rows)),
        "packages": dict(Counter(r["package"] for r in rows)),
    }
    (args.out / "tool_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
