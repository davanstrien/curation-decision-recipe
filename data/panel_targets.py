# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Panel-arm targets: replace every slice label with a target built from panel votes, on the
same rows (same id, state, question), plus new rows for the generic Step 1 axes.

Inputs
  --slice SRC=FILE        the screened slice files of the third-party arm (prompt / web decision rows)
  --items-dir DIR         dm-curation/step1 (<pool>/{items.jsonl, spec.json}): text -> item id per pool
  --votes DIR [DIR ...]   label dirs (<pool>/votes.jsonl), e.g. labels/panel-v1 labels/panel-v1-mimo; votes of the
                          same item merge across dirs; last record per id wins inside a dir
Targets (per row, from the raters with a valid vote; a row with fewer than --min-raters votes is dropped)
  choice   vote share per option (soft)
  scale    rank-normalised: per pool, axis and rater, a level's interval on that rater's CDF is spread over the
           pooled levels it overlaps, in proportion to the overlap (removes a rater's scale offset, e.g. one rater who
           rates most items 4); target = mean of these distributions over raters. (A first version snapped each level
           to the pooled level at its CDF midpoint: with mass piled on one level, two raters who both said 3 mapped to
           2 and 4.)
           --scale-mode hard (Step 2 control, review S2): the median of the raw rater levels, one hard target (no
           rank normalisation, no soft mass); ties between two middle levels take the lower level.
  multi    per label: 1 if more than half of the raters chose it, else 0 (one rater over-flags "missing" on 34% of
           WildChat items in Step 1a); arena_tag rows (one yes/no row per tag) use the same rule
Outputs (--out DIR)
  <name>_panel_decision.jsonl   the slice rows with panel targets (teacher = the raters)
  <name>_tp_matched_decision.jsonl  the same rows with their third-party targets (like-for-like cur-tp arm)
  generic_panel_decision.jsonl  new rows: generic axes for every prompt-pool item (difficulty is already a native
                                row for DeepMath, so it is skipped there)
  panel_targets_stats.json      rows kept / dropped per source, raters per row, rank maps per scale axis
usage: uv run scripts/panel_targets.py --slice prompt=prompt.jsonl --slice web=web.jsonl --items-dir step1 \
         --votes labels/panel-v1 labels/panel-v1-mimo --out panel_slice
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SRC_AXIS = {  # slice src -> (pool, axis in the step1 spec)
    "numina_qtype": ("numina", "numina_qtype"), "numina_valid": ("numina", "numina_valid"),
    "numina_ptype": ("numina", "numina_ptype"), "wi_atype": ("wi", "wi_atype"), "wi_category": ("wi", "wi_category"),
    "wi_difficulty": ("wi", "wi_difficulty"), "dm_difficulty": ("deepmath", "difficulty"),
    "dm_topic": ("deepmath", "dm_topic"), "arena_criteria": ("arena", "arena_criteria"),
    "arena_tag": ("arena", "arena_tags"), "weborganizer_topic_405b": ("web", "topic"),
    "weborganizer_format_405b": ("web", "format"), "fineweb_edu_llama3": ("web", "education"),
}
ARENA_TAGS = ("math", "creative_writing", "if")
GENERIC = ("task", "answer_format", "domain", "difficulty", "input_quality", "defects")
SHORT = {"tencent/Hy3:deepinfra": "Hy3", "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra": "DeepSeek-V4-Flash",
         "google/gemma-4-26B-A4B-it:deepinfra": "Gemma-4-26B", "XiaomiMiMo/MiMo-V2.6-Flash-RL:novita": "MiMo-V2.6-Flash"}


def jl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def load_votes(dirs, pool):
    """-> {item id: {model: {axis: value}}} merged over dirs; None votes never overwrite a value"""
    out = defaultdict(dict)
    for d in dirs:
        p = Path(d) / pool / "votes.jsonl"
        if not p.exists():
            continue
        for r in jl(p):
            for m, v in r["votes"].items():
                if v is not None:
                    out[r["id"]][m] = v
    return out


def rank_maps(votes, axis, n_levels):
    """per rater: level -> {pooled level: weight}, the overlap of the level's interval on the rater's CDF with each
    pooled level's interval on the pooled CDF (weights sum to 1)"""
    per_rater = defaultdict(Counter)
    pooled = Counter()
    for v in votes.values():
        for m, ans in v.items():
            x = ans.get(axis)
            if x is not None:
                per_rater[m][x] += 1
                pooled[x] += 1

    def intervals(c):
        n, acc, out = sum(c.values()), 0, []
        for k in range(n_levels):
            out.append((acc / n, (acc + c[k]) / n))
            acc += c[k]
        return out

    pool_iv = intervals(pooled)
    maps = {}
    for m, c in per_rater.items():
        mp = {}
        for k, (lo, hi) in enumerate(intervals(c)):
            if hi <= lo:  # level never used by this rater
                mp[k] = {k: 1.0}
                continue
            w = {j: (min(hi, phi) - max(lo, plo)) / (hi - lo) for j, (plo, phi) in enumerate(pool_iv)}
            mp[k] = {j: round(x, 4) for j, x in w.items() if x > 1e-9}
        maps[m] = mp
    return maps


def target_for(kind, axis, keys, item_votes, maps, min_raters, tag=None, scale_mode="rank"):
    """-> (target dict, n raters) or (None, n raters)"""
    vals = [(m, v[axis]) for m, v in item_votes.items() if v.get(axis) is not None]
    if len(vals) < min_raters:
        return None, len(vals)
    n = len(vals)
    if kind == "choice":
        c = Counter(x for _, x in vals)
        return {k: round(c[k] / n, 4) for k in keys if c[k]}, n
    if kind == "scale" and scale_mode == "hard":
        lv = sorted(x for _, x in vals)
        return {keys[lv[(n - 1) // 2]]: 1.0}, n
    if kind == "scale":
        acc = Counter()
        for m, x in vals:
            for j, w in maps[m][x].items():
                acc[j] += w / n
        return {keys[j]: round(acc[j], 4) for j in range(len(keys)) if round(acc[j], 4)}, n
    c = Counter(k for _, x in vals for k in set(x))  # multi
    if tag is not None:  # one yes/no row for one tag
        return ({"true": 1.0} if c[tag] > n / 2 else {"false": 1.0}), n
    return {k: float(c[k] > n / 2) for k in keys}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", action="append", required=True, help="NAME=FILE")
    ap.add_argument("--items-dir", required=True)
    ap.add_argument("--votes", nargs="+", required=True)
    ap.add_argument("--min-raters", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-mode", choices=("rank", "hard"), default="rank")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items_dir = Path(args.items_dir)
    pools = sorted({p for p, _ in SRC_AXIS.values()})
    specs, text_id, votes, maps = {}, {}, {}, {}
    for pool in pools:
        specs[pool] = {a["name"]: a for a in json.loads((items_dir / pool / "spec.json").read_text())["axes"]}
        text_id[pool] = {it["text"]: it["id"] for it in jl(items_dir / pool / "items.jsonl")}
        votes[pool] = load_votes(args.votes, pool)
        maps[pool] = {name: rank_maps(votes[pool], name, len(ax["labels"]))
                      for name, ax in specs[pool].items() if ax["kind"] == "scale"}
    raters = sorted({m for pool in pools for v in votes[pool].values() for m in v})
    teacher = "panel votes: " + " + ".join(SHORT.get(m, m) for m in raters)
    stats = {"raters": raters, "min_raters": args.min_raters, "slices": {}, "rank_maps": {
        pool: {ax: {SHORT.get(m, m): mp for m, mp in mm.items()} for ax, mm in pm.items()} for pool, pm in maps.items()}}

    for spec_arg in args.slice:
        name, path = spec_arg.split("=", 1)
        st = defaultdict(Counter)
        with open(path) as f, open(out / f"{name}_panel_decision.jsonl", "w") as fo, \
                open(out / f"{name}_tp_matched_decision.jsonl", "w") as fm:
            for line in f:
                r = json.loads(line)
                pool, axis = SRC_AXIS[r["src"]]
                ax = specs[pool][axis]
                iid = text_id[pool].get(r["state"])
                if iid is None:
                    st[r["src"]]["dropped_no_item"] += 1
                    continue
                crit_keys = list(r["question"]["criteria"])
                if ax["kind"] == "multi" and r["src"] == "arena_tag":
                    tag = next(t for t in ARENA_TAGS if f"-{t}-" in r["id"])
                    tgt, n = target_for("multi", axis, None, votes[pool].get(iid, {}), None, args.min_raters, tag)
                else:
                    keys = [k for k, _ in ax["labels"]]
                    if ax["kind"] != "multi" and sorted(keys) != sorted(crit_keys):
                        raise SystemExit(f"{r['src']}: row options {crit_keys} != panel axis options {keys}")
                    tgt, n = target_for(ax["kind"], axis, keys, votes[pool].get(iid, {}), maps[pool].get(axis),
                                        args.min_raters, scale_mode=args.scale_mode)
                st[r["src"]][f"raters_{n}"] += 1
                if tgt is None:
                    st[r["src"]]["dropped_few_raters"] += 1
                    continue
                st[r["src"]]["kept"] += 1
                fm.write(json.dumps(r, ensure_ascii=False) + "\n")  # third-party arm on exactly the same rows
                fo.write(json.dumps({**r, "target": tgt, "teacher": teacher, "third_party_target": r["target"]},
                                    ensure_ascii=False) + "\n")
        stats["slices"][name] = {s: dict(c) for s, c in st.items()}

    st = defaultdict(Counter)
    with open(out / "generic_panel_decision.jsonl", "w") as fo:
        for pool in pools:
            if pool == "web":
                continue
            for text, iid in sorted(text_id[pool].items(), key=lambda x: x[1]):
                for axis in GENERIC:
                    if pool == "deepmath" and axis == "difficulty":
                        continue
                    ax = specs[pool][axis]
                    keys = [k for k, _ in ax["labels"]]
                    tgt, n = target_for(ax["kind"], axis, keys, votes[pool].get(iid, {}), maps[pool].get(axis),
                                        args.min_raters, scale_mode=args.scale_mode)
                    if tgt is None:
                        st[f"{pool}/{axis}"]["dropped_few_raters"] += 1
                        continue
                    st[f"{pool}/{axis}"]["kept"] += 1
                    row = {"id": f"pn-{iid}-{axis}", "src": f"panel_{axis}", "teacher": teacher, "state": text,
                           "question": {"type": "choice", "instructions": ax["question"],
                                        "criteria": {k: d for k, d in ax["labels"]}},
                           "target": tgt}
                    if ax["kind"] == "multi":
                        row["multi_label"] = True
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["generic"] = {k: dict(c) for k, c in st.items()}
    (out / "panel_targets_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({"slices": stats["slices"], "generic_kept": sum(c["kept"] for c in st.values())}, indent=1))


if __name__ == "__main__":
    main()
