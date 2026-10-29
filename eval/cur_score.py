# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-learn"]
# ///
"""Score curation eval predictions (gliclass_ub_train.py / cur_edge_eval.py output) against a frozen eval version.

Completeness contract (per set; a set that breaks it is reported INVALID with the reason and gets no metrics):
  exactly one prediction per eval id (no missing, no duplicates); unknown ids are an error unless --filter-extra
  (old prediction files written for a larger eval, e.g. Omni before dedup); each prediction's `ids` equal the eval
  row's label keys (as a set); logits finite and one per key. A set with no prediction file is reported MISSING.
Set -> file map: eval files come from EVAL_DIR/inventory.json when present (else eval_*.jsonl); prediction files are
  PRED_DIR/cur_<set>__*.jsonl or PRED_DIR/<set>__*.jsonl, plus LEGACY aliases below; more than one match is an error.
Metrics:
  all sets   top1, majority, macro_f1 (classes = the gold classes present in the eval set; fixed per set, so the
             denominator does not depend on the model), nll (softmax over the row's labels)
  yes / no   auroc of P(true)
  ordinal    (graded sets) probabilities renormalised over the LEVEL logits only (non-levels such as "Indeterminate"
             are dropped from the softmax; rows whose gold is a non-level are left out, n_ordinal counts the rest):
             spearman (expected level), mae for three decoders (expected / argmax / median), far-miss rate
             (|argmax - gold| >= 2) with its count, rps and nll over the levels (decoder-free), and baselines chosen on
             the eval labels (oracle constants, labelled as such): mae_const, within1_const, rps_prior / nll_prior
             (the gold level histogram as a constant distribution). within1 (argmax) is reported next to its constant.
  extras     fwe*: auroc_ge3 (ranking score = expected level); ho_polaris_passrate: spearman vs 1 - raw pass rate;
             pc_omni_difficulty_dedup: spearman vs the original 1-10 rating (EVAL_DIR/omni_meta.jsonl)
usage: uv run scripts/cur_score.py EVAL_DIR PRED_DIR [PRED_DIR ...] [--sets a,b] [--filter-extra] [--json out.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

NON_LEVELS = {"Indeterminate", "Abstain"}
ORDINAL_NAMES = {"fwe", "fwe_desc", "eai_reasoning_depth", "eai_education_level"}
# prediction files written before the Omni dedup used the old set names
LEGACY = {
    "pc_omni_difficulty_dedup": ["cur_pc_omni_difficulty__*.jsonl", "pc_omni_difficulty__*.jsonl"],
    "pc_omni_domain_dedup": ["cur_pc_omni_domain__*.jsonl", "pc_omni_domain__*.jsonl"],
}


class Invalid(Exception):
    pass


def is_ordinal(name, inv):
    if inv and name in inv:
        return inv[name]["main_metric"] == "spearman"
    return name in ORDINAL_NAMES or "difficulty" in name


def load_eval(path):
    """id -> row (keys in criteria order, gold key, meta)"""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            q = r["question"]
            keys = ["true", "false"] if q["type"] == "noul" else list(q["criteria"])
            gold = max(r["target"], key=r["target"].get)
            if r["id"] in out:
                raise Invalid(f"duplicate eval id {r['id']}")
            out[r["id"]] = {"keys": keys, "gold": gold, "meta": r.get("meta") or {}}
    return out


def load_preds(path, ev, filter_extra):
    preds, extra = {}, 0
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            if p["id"] not in ev:
                if filter_extra:
                    extra += 1
                    continue
                raise Invalid(f"unknown prediction id {p['id']} (use --filter-extra for older, larger eval files)")
            if p["id"] in preds:
                raise Invalid(f"duplicate prediction id {p['id']}")
            e = ev[p["id"]]
            if set(p["ids"]) != set(e["keys"]) or len(p["ids"]) != len(e["keys"]):
                raise Invalid(f"label keys differ for {p['id']}: {p['ids']} vs {e['keys']}")
            lg = np.asarray(p["logits"], dtype=float)
            if lg.shape != (len(p["ids"]),) or not np.all(np.isfinite(lg)):
                raise Invalid(f"bad logits for {p['id']}")
            preds[p["id"]] = (list(p["ids"]), lg)
    missing = len(ev) - len(preds)
    if missing:
        raise Invalid(f"{missing} of {len(ev)} eval ids have no prediction")
    return preds, extra


def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


def median_level(p):
    return int(np.searchsorted(np.cumsum(p), 0.5))


def ordinal_metrics(rows, n_levels):
    """rows: (level probs over K levels, gold level, meta)"""
    if not rows:
        return {"n_ordinal": 0}
    P = np.stack([r[0] for r in rows])
    g = np.array([r[1] for r in rows])
    lv = np.arange(n_levels)
    exp = P @ lv
    amax = P.argmax(1)
    med = np.array([median_level(p) for p in P])
    T = np.zeros_like(P)
    T[np.arange(len(g)), g] = 1
    rps = (((P.cumsum(1) - T.cumsum(1))[:, :-1]) ** 2).sum(1) / max(n_levels - 1, 1)
    nll = -np.log(np.clip(P[np.arange(len(g)), g], 1e-12, None))
    prior = np.bincount(g, minlength=n_levels) / len(g)
    rps_prior = (((prior.cumsum() - T.cumsum(1))[:, :-1]) ** 2).sum(1) / max(n_levels - 1, 1)
    nll_prior = -np.log(np.clip(prior[g], 1e-12, None))
    far = np.abs(amax - g) >= 2
    m = {
        "n_ordinal": int(len(g)),
        "spearman": float(spearmanr(exp, g).statistic) if len(set(g)) > 1 else None,
        "mae_expected": float(np.abs(exp - g).mean()),
        "mae_argmax": float(np.abs(amax - g).mean()),
        "mae_median": float(np.abs(med - g).mean()),
        "far_miss": float(far.mean()),
        "far_miss_n": int(far.sum()),
        "within1": float((np.abs(amax - g) <= 1).mean()),
        "rps": float(rps.mean()),
        "nll_levels": float(nll.mean()),
        "mae_const": float(min(np.abs(c - g).mean() for c in lv)),
        "within1_const": float(max((np.abs(c - g) <= 1).mean() for c in lv)),
        "rps_prior": float(rps_prior.mean()),
        "nll_prior": float(nll_prior.mean()),
        "_expected": exp,
    }
    return m


def score_set(name, ev, preds, inv, omni_rating):
    gold, pred, nll, p_true, ordinal_rows = [], [], [], [], []
    ordinal = is_ordinal(name, inv)
    level_keys = None
    for rid, e in ev.items():
        ids, lg = preds[rid]
        probs = softmax(lg)
        gold.append(e["gold"])
        pred.append(ids[int(probs.argmax())])
        nll.append(-np.log(max(probs[ids.index(e["gold"])], 1e-12)))
        if "true" in ids:
            p_true.append(float(probs[ids.index("true")]))
        if ordinal:
            levels = level_order(e["keys"])
            level_keys = level_keys or levels
            if e["gold"] in levels:
                idx = [ids.index(k) for k in levels]
                ordinal_rows.append((softmax(lg[idx]), levels.index(e["gold"]), rid, e["meta"]))
    gold_a, pred_a = np.array(gold), np.array(pred)
    classes = sorted(set(gold))
    vals, counts = np.unique(gold_a, return_counts=True)
    m = {
        "n": len(gold),
        "top1": float((gold_a == pred_a).mean()),
        "majority": float(counts.max() / counts.sum()),
        "macro_f1": float(f1_score(gold_a, pred_a, labels=classes, average="macro", zero_division=0)),
        "macro_f1_classes": len(classes),
        "nll": float(np.mean(nll)),
    }
    if set(gold) <= {"true", "false"} and len(vals) == 2:
        m["auroc"] = float(roc_auc_score(gold_a == "true", p_true))
    if ordinal:
        om = ordinal_metrics([(r[0], r[1], r[3]) for r in ordinal_rows], len(level_keys or []))
        exp = om.pop("_expected", None)
        m.update(om)
        if exp is not None and len(exp):
            gl = np.array([r[1] for r in ordinal_rows])
            if name.startswith("fwe"):
                m["auroc_ge3"] = float(roc_auc_score(gl >= 3, exp))
            if name == "ho_polaris_passrate":
                hard = np.array([1 - r[3]["pass"] / r[3]["total"] for r in ordinal_rows])
                m["spearman_vs_passrate"] = float(spearmanr(exp, hard).statistic)
            if name == "pc_omni_difficulty_dedup" and omni_rating:
                rt = np.array([omni_rating[r[2]] for r in ordinal_rows])
                m["spearman_vs_rating"] = float(spearmanr(exp, rt).statistic)
    return m


def find_pred_file(run, s):
    """exactly one match across the canonical and legacy patterns, else Invalid"""
    pats = [f"cur_{s}__*.jsonl", f"{s}__*.jsonl"] + LEGACY.get(s, [])
    files = sorted({f for pat in pats for f in run.glob(pat)})
    if len(files) > 1:
        raise Invalid(f"ambiguous prediction files {[f.name for f in files]}")
    return files[0] if files else None


def level_order(keys):
    """levels low -> high: level_<k> keys sorted numerically (as the trainer); other key sets in file order"""
    lv = [k for k in keys if k not in NON_LEVELS]
    if lv and all(re.fullmatch(r"level_\d+", k) for k in lv):
        return sorted(lv, key=lambda k: int(k.split("_")[1]))
    return lv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("pred_dirs", nargs="+")
    ap.add_argument("--sets", default="", help="comma list (default: every set in the eval dir)")
    ap.add_argument("--filter-extra", action="store_true", help="drop prediction ids that are not in the eval file")
    ap.add_argument("--json", default=None, help="write the full results here")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any set is MISSING or INVALID")
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    inv_path = eval_dir / "inventory.json"
    inv = json.loads(inv_path.read_text())["sets"] if inv_path.exists() else None
    if inv:
        files = {s: eval_dir / v["file"] for s, v in inv.items()}
    else:
        files = {p.stem.removeprefix("eval_"): p for p in eval_dir.glob("eval_*.jsonl")}
    sets = sorted(files) if not args.sets else args.sets.split(",")
    unknown = [s for s in sets if s not in files]
    if unknown:
        raise SystemExit(f"unknown sets {unknown}; known: {sorted(files)}")
    omni_rating = {}
    if (eval_dir / "omni_meta.jsonl").exists():
        with open(eval_dir / "omni_meta.jsonl") as f:
            for line in f:
                r = json.loads(line)
                omni_rating[r["difficulty_id"]] = r["rating"]
    evals = {s: load_eval(files[s]) for s in sets}
    results, status = {}, {}
    for run in [Path(p) for p in args.pred_dirs]:
        name = run.parent.name if run.name == "predictions" else run.name
        results[name], status[name] = {}, {}
        for s in sets:
            try:
                f = find_pred_file(run, s)
                if f is None:
                    status[name][s] = "MISSING"
                    continue
                preds, extra = load_preds(f, evals[s], args.filter_extra)
                results[name][s] = score_set(s, evals[s], preds, inv, omni_rating)
                results[name][s]["file"] = f.name
                results[name][s]["filtered_extra"] = extra
                status[name][s] = "ok"
            except Invalid as e:
                status[name][s] = f"INVALID: {e}"
    names = list(results)
    print("| set | metric | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for s in sets:
        bad = [f"{n}: {status[n][s]}" for n in names if status[n][s] != "ok"]
        if bad:
            print(f"| {s} | status | " + " | ".join(status[n][s][:40] for n in names) + " |")
        ok = [n for n in names if status[n][s] == "ok"]
        if not ok:
            continue
        ref = results[ok[0]][s]
        metrics = ["top1", "macro_f1", "auroc"]
        if "n_ordinal" in ref:
            metrics += ["spearman", "mae_expected", "mae_argmax", "mae_median", "mae_const", "far_miss", "far_miss_n",
                        "within1", "within1_const", "rps", "rps_prior", "nll_levels", "nll_prior", "n_ordinal"]
        metrics += [k for k in ("auroc_ge3", "spearman_vs_passrate", "spearman_vs_rating") if k in ref]
        for k in metrics:
            if k not in ref:
                continue
            cells = []
            for n in names:
                v = results[n].get(s, {}).get(k)
                cells.append("–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
            label = f"{k} (majority {ref['majority']:.3f})" if k == "top1" else k
            print(f"| {s} | {label} | " + " | ".join(cells) + " |")
    out = {"results": results, "status": status, "eval_dir": str(eval_dir)}
    if inv_path.exists():
        out["inventory_version"] = json.loads(inv_path.read_text()).get("version")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    else:
        print(json.dumps(out))
    bad = sum(v != "ok" for st in status.values() for v in st.values())
    if bad:
        print(f"{bad} (run, set) cells are MISSING or INVALID", file=sys.stderr)
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
