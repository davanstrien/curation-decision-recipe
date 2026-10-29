# /// script
# requires-python = ">=3.11"
# ///
"""DecisionBank (jgeuter/DecisionBank, config DecisionBank-260k, sha fbeee677) -> decision rows for mix_build.py.

Targets are 50 % source label +
50 % Qwen3.8-27B (decider / human_soft / jeff10_llm parts), Qwen3.8-27B with thinking (written questions) or code
(programmatic questions); no Jev output per the card.
Kept: rows whose source_license is in ALLOW (permissive and share-alike; NC, research-only, unknown,
"other" terms, GPL and "mixed ... other" dropped). Dropped: the HelpSteer families (our holdouts use HelpSteer2 /
HelpSteer3); mix_build.py decontaminates the rest against every eval text.
Mapping: noul (yes/no or true/false options, any order) -> noul with criteria {true, false}; choice -> choice with
criteria {option id: description}; score -> choice with level_<k> keys in the stored (low -> high) order, so the
trainer treats it as ordinal like our graded rows. Target = target_distribution, or one-hot on `label` for the
programmatic rows without exact probabilities. JSON states are parsed back to objects.
usage: uv run scripts/db_build.py DATA_DIR OUT_DIR   (DATA_DIR = .../data/DecisionBank-260k)
"""

import json
import sys
from collections import Counter
from pathlib import Path

ALLOW = {
    "cc-by-sa-4.0 (ours; written with Qwen3.8-27B)": "sa",
    "cc-by-sa-4.0 (ours)": "sa",
    "cc-by-sa-4.0 (ours; written with Qwen3.6-27B)": "sa",
    "cc-by-sa-4.0": "sa",
    "cc-by-sa-3.0": "sa",
    "cdla-sharing-1.0": "sa",
    "apache-2.0": "permissive",
    "cc-by-4.0": "permissive",
    "cc-by-3.0": "permissive",
    "mit": "permissive",
    "cc0-1.0": "permissive",
}
DROP_FAMILIES = {"helpsteer2", "helpsteer3_pref"}


def convert(r):
    opts = r["options"]
    probs = r["target_distribution"]
    if probs is None:
        probs = [1.0 if i == r["label"] else 0.0 for i in range(len(opts))]
    state = json.loads(r["state"]) if r["state_is_json"] else r["state"]
    ids = [o["id"] for o in opts]
    if r["qtype"] == "noul":
        low = [x.lower() for x in ids]
        if sorted(low) == ["no", "yes"]:
            pos, neg = low.index("yes"), low.index("no")
        elif sorted(low) == ["false", "true"]:
            pos, neg = low.index("true"), low.index("false")
        else:
            return None, "noul options"
        q = {"type": "noul", "instructions": r["question"],
             "criteria": {"true": opts[pos]["description"], "false": opts[neg]["description"]}}
        target = {"true": probs[pos], "false": probs[neg]}
    elif r["qtype"] == "choice":
        q = {"type": "choice", "instructions": r["question"], "criteria": {o["id"]: o["description"] for o in opts}}
        target = {o["id"]: p for o, p in zip(opts, probs, strict=True)}
    elif r["qtype"] == "score":
        keys = [f"level_{i}" for i in range(len(opts))]
        q = {"type": "choice", "instructions": r["question"],
             "criteria": {k: o["description"] for k, o in zip(keys, opts, strict=True)}}
        target = {k: p for k, p in zip(keys, probs, strict=True)}
    else:
        return None, f"qtype {r['qtype']}"
    if len(set(q["criteria"].values())) < len(q["criteria"]):
        return None, "duplicate option text"
    target = {k: float(v) for k, v in target.items() if v > 0}
    if not target:
        return None, "empty target"
    return {"id": f"db/{r['id']}", "src": f"db_{r['family']}", "teacher": r["target_origin"],
            "licence_class": ALLOW[r["source_license"]], "licence": r["source_license"],
            "state": state, "question": q, "target": target}, None


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    kept_by = Counter()
    with open(out / "decisionbank_decision.jsonl", "w") as fo:
        for f in sorted(data.glob("train-*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                stats["read"] += 1
                if r["source_license"] not in ALLOW:
                    stats["drop licence"] += 1
                    continue
                if r["family"] in DROP_FAMILIES:
                    stats["drop helpsteer"] += 1
                    continue
                row, why = convert(r)
                if row is None:
                    stats[f"drop {why}"] += 1
                    continue
                stats["kept"] += 1
                kept_by[(r["part"], r["qtype"])] += 1
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"stats": dict(stats), "kept_by_part_qtype": {f"{p}/{q}": n for (p, q), n in sorted(kept_by.items())}}
    (out / "build_stats.json").write_text(json.dumps(report, indent=1))
    (out / "SOURCE.json").write_text(json.dumps({
        "dataset": "jgeuter/DecisionBank", "config": "DecisionBank-260k",
        "revision": "fbeee67758966fe78bfb008b28fc771924c0bc71", "teacher": "see row teacher (Qwen3.8-27B mix)",
        "note": "private screen only; licence allow-list in scripts/db_build.py"}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
