# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-learn"]
# ///
"""Curation gate from prediction files, on the pre-registered main metric per set
(evals inventory `main_metric`: top1, auroc on P(true), or spearman on the expected level; scored with
scripts/cur_score.py so the point estimates match it exactly).

Units: every non-holdout set is one unit, except that a set and its `_desc` twin form one unit (mean of the two) and the
`pc_arena_*` sets form one unit (mean AUROC). Holdouts (`ho_*`) are reported per set, outside the mean.
Uncertainty:
  item bootstrap   items resampled within each set, the same draw for every run and for both `_desc` twins (twins hold
                   the same texts under different ids; aligned by eval state text); arm diff = mean(A seeds) - mean(B)
  unit jackknife   the arm diff recomputed leaving out one unit at a time (how much one unit drives the mean)
  seed spread      per unit, max over arms of (max - min over that arm's seeds)
Neither interval includes seed variance beyond the seeds given. Every run must have every inventory set.
usage: uv run scripts/cur_gate.py EVAL_DIR ROOT A=run1,run2 B=run3,run4 [--boot 1000] [--json out.json]
  ROOT/<run>/predictions/cur_<set>__desc.jsonl
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

_spec = importlib.util.spec_from_file_location("cs", Path(__file__).resolve().parent / "cur_score.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

# gold written by the same labeller as the third-party training targets (the tp arm learns the eval's teacher).
# Caveat (review): "independent" gold is still LLM-labelled (EAI) or shares a scale convention (Omni vs DeepMath).
TEACHER_MATCHED = ("arena", "fwe", "pc_dm_difficulty", "pc_numina_qtype", "pc_numina_valid", "pc_wi_atype",
                   "pc_wi_difficulty", "wo_format", "wo_topic")


def unit_of(s):
    if s.startswith("ho_"):
        return None
    if s.startswith("pc_arena_"):
        return "arena"
    return s.removesuffix("_desc")


def item_arrays(s, ev, preds, inv):
    """per-item values for the main metric, in eval-id order: (metric, ids, arrays)"""
    metric = inv[s]["main_metric"]
    ids = list(ev)
    if metric == "top1":
        return metric, ids, {"x": np.array([float(preds[i][0][int(preds[i][1].argmax())] == ev[i]["gold"]) for i in ids])}
    if metric == "auroc":
        p = np.array([cs.softmax(preds[i][1])[preds[i][0].index("true")] for i in ids])
        return metric, ids, {"p": p, "y": np.array([ev[i]["gold"] == "true" for i in ids])}
    levels = cs.level_order(ev[ids[0]]["keys"])
    keep = [i for i in ids if ev[i]["gold"] in levels]  # drops non-level gold (e.g. "Indeterminate"), as cur_score
    exp, g = [], []
    for i in keep:
        pid, lg = preds[i]
        pr = cs.softmax(lg[[pid.index(k) for k in levels]])
        exp.append(float(pr @ np.arange(len(levels))))
        g.append(levels.index(ev[i]["gold"]))
    return metric, keep, {"e": np.array(exp), "g": np.array(g)}


def metric_value(metric, a, idx=None):
    sel = (lambda v: v) if idx is None else (lambda v: v[idx])
    if metric == "top1":
        return float(sel(a["x"]).mean())
    if metric == "auroc":
        y = sel(a["y"])
        return float(roc_auc_score(y, sel(a["p"]))) if 0 < y.sum() < len(y) else float("nan")
    return float(spearmanr(sel(a["e"]), sel(a["g"])).statistic)


def gate(values):
    """values: set -> metric -> (gate mean, per-unit)"""
    units = {}
    for s, v in values.items():
        u = unit_of(s)
        if u is not None:
            units.setdefault(u, []).append(v)
    pu = {u: float(np.mean(x)) for u, x in units.items()}
    return float(np.mean(list(pu.values()))), pu


def state_keys(path):
    with open(path) as f:
        return {(r := json.loads(x))["id"]: hashlib.sha1(r["state"].encode()).hexdigest() for x in f if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("root")
    ap.add_argument("arms", nargs=2, help="NAME=run1,run2")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--json")
    args = ap.parse_args()
    eval_dir, root = Path(args.eval_dir), Path(args.root)
    inv = json.loads((eval_dir / "inventory.json").read_text())["sets"]
    arms = {}
    for a in args.arms:
        name, runs = a.split("=", 1)
        arms[name] = runs.split(",")
    (na, ra), (nb, rb) = arms.items()
    sets = sorted(inv)
    evals = {s: cs.load_eval(eval_dir / inv[s]["file"]) for s in sets}

    arr, point, metric_of, ids_of = {}, {}, {}, {}
    for r in ra + rb:
        arr[r], point[r] = {}, {}
        for s in sets:
            f = cs.find_pred_file(root / r / "predictions", s)
            if f is None:
                raise SystemExit(f"{r}: no predictions for {s} (every run must have every inventory set)")
            preds, _ = cs.load_preds(f, evals[s], False)
            ref = cs.score_set(s, evals[s], preds, inv, {})[inv[s]["main_metric"]]
            metric, ids, a = item_arrays(s, evals[s], preds, inv)
            v = metric_value(metric, a)
            if abs(v - ref) > 1e-6:
                raise SystemExit(f"{r} {s}: {metric} {v} != cur_score {ref}")
            arr[r][s], point[r][s], metric_of[s], ids_of[s] = a, v, metric, ids

    res = {"arms": arms, "metric": metric_of, "runs": {}, "units": {}, "holdouts": {}}
    gm = {}
    for r in ra + rb:
        gm[r] = gate(point[r])
        res["runs"][r] = {"gate_mean": round(gm[r][0], 4), "units": {u: round(x, 4) for u, x in gm[r][1].items()}}
    units = sorted(gm[ra[0]][1])

    def arm(runs, u=None):
        return float(np.mean([gm[r][0] if u is None else gm[r][1][u] for r in runs]))

    for u in units:
        spreads = [max(gm[r][1][u] for r in x) - min(gm[r][1][u] for r in x) for x in (ra, rb) if len(x) > 1]
        res["units"][u] = {na: round(arm(ra, u), 4), nb: round(arm(rb, u), 4),
                           "diff_pts": round(100 * (arm(ra, u) - arm(rb, u)), 2),
                           "seed_spread_pts": round(100 * max(spreads), 2) if spreads else None,
                           "gold": "teacher-matched" if u in TEACHER_MATCHED else "independent"}
    diff = 100 * (arm(ra) - arm(rb))
    jack = sorted(round(100 * float(np.mean([v["diff_pts"] / 100 for k, v in res["units"].items() if k != u])), 2)
                  for u in units)

    # item bootstrap; `_desc` twins share draws through the eval state text
    twin_of = {s: s.removesuffix("_desc") for s in sets}
    base_order = {}
    for s in sets:
        if twin_of[s] == s:
            keys = state_keys(eval_dir / inv[s]["file"])
            base_order[s] = [keys[i] for i in ids_of[s]]
    pos = {}
    for s in sets:
        b = twin_of[s]
        if b != s and b in base_order:
            keys = state_keys(eval_dir / inv[s]["file"])
            where = {k: j for j, k in enumerate(base_order[b])}
            if len(where) != len(base_order[b]):
                raise SystemExit(f"{b}: duplicate state texts; twin bootstrap alignment needs unique states")
            pos[s] = np.array([where[keys[i]] for i in ids_of[s]])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(args.boot):
        draws = {s: rng.integers(0, len(ids_of[s]), len(ids_of[s])) for s in sets if twin_of[s] == s}
        idx = {}
        for s in sets:
            if s in draws:
                idx[s] = draws[s]
            elif s in pos:  # twin: map the base draw onto this file's rows
                inv_pos = np.empty(len(pos[s]), dtype=int)
                inv_pos[pos[s]] = np.arange(len(pos[s]))
                idx[s] = inv_pos[draws[twin_of[s]]]
            else:
                idx[s] = rng.integers(0, len(ids_of[s]), len(ids_of[s]))
        ga = np.mean([gate({s: metric_value(metric_of[s], arr[r][s], idx[s]) for s in sets})[0] for r in ra])
        gb = np.mean([gate({s: metric_value(metric_of[s], arr[r][s], idx[s]) for s in sets})[0] for r in rb])
        boots.append(100 * (ga - gb))
    boots = np.sort(np.array(boots))
    res["gate"] = {na: round(arm(ra), 4), nb: round(arm(rb), 4), "diff_pts": round(diff, 2),
                   "item_ci95": [round(float(np.quantile(boots, 0.025)), 2), round(float(np.quantile(boots, 0.975)), 2)],
                   "unit_jackknife_range": [jack[0], jack[-1]],
                   "cross_seed_pairs": sorted(round(100 * (gm[x][0] - gm[y][0]), 2) for x in ra for y in rb)}
    for g in ("teacher-matched", "independent"):
        us = [u for u in units if res["units"][u]["gold"] == g]
        res[f"split_{g}"] = {"units": len(us), "diff_pts": round(float(np.mean([res["units"][u]["diff_pts"] for u in us])), 2),
                             "max_seed_spread_pts": max(res["units"][u]["seed_spread_pts"] or 0 for u in us)}
    for s in sets:
        if s.startswith("ho_"):
            res["holdouts"][s] = {"metric": metric_of[s], na: round(float(np.mean([point[r][s] for r in ra])), 4),
                                  nb: round(float(np.mean([point[r][s] for r in rb])), 4)}
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
    g = res["gate"]
    print(f"gate mean (main metrics) {na} {g[na]:.4f}  {nb} {g[nb]:.4f}  diff {diff:+.2f} pts  item-CI95 {g['item_ci95']}  "
          f"unit-jackknife {g['unit_jackknife_range']}  cross-seed pairs {g['cross_seed_pairs']}")
    for k in ("teacher-matched", "independent"):
        print(f"  {k}: {res[f'split_{k}']}")
    print(f"{'unit':30}{na:>9}{nb:>9}{'diff':>8}{'spread':>8}  gold")
    for u in units:
        x = res["units"][u]
        print(f"{u:30}{x[na]:9.3f}{x[nb]:9.3f}{x['diff_pts']:+8.1f}{x['seed_spread_pts'] or 0:8.1f}  {x['gold']}")
    print("holdouts:")
    for s, x in res["holdouts"].items():
        print(f"  {s:28}{x['metric']:>9}{x[na]:9.3f}{x[nb]:9.3f}{100 * (x[na] - x[nb]):+8.1f}")


if __name__ == "__main__":
    main()
