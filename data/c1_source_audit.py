# /// script
# requires-python = ">=3.11"
# dependencies = ["scikit-learn", "numpy", "scipy"]
# ///
"""C1 source-level shortcut audit of a gliclass-format pool (no GPU).

Per source (src field; up to --cap single-label rows by reservoir sample), text-blind baselines vs chance:
  chance     mean of 1 / number of options
  majority   the source's most frequent gold label string always wins when present (label prior)
  longest    the gold is the unique longest label (characters)
  options    out-of-fold options-only ranker: TF-IDF word 1-2-grams of "prompt || label" + relative length cues,
             logistic regression on "this option is the gold", 5-fold grouped by row; no text (state) at all
Option position is not a feature: training shuffles labels per example.
A source is flagged when options - chance >= --flag-gap or majority >= --flag-majority.
Writes a JSON report (per source) and a TSV; prints the flagged sources sorted by row count.
usage: uv run c1_source_audit.py pool.jsonl --out DIR [--cap 3000]
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def as_list(v):
    if isinstance(v, str):
        try:
            v = json.loads(v.replace("'", '"'))
        except json.JSONDecodeError:
            return None
    return v


def single_label(row):
    """-> (labels, gold index) for a single-label row with >= 2 distinct labels, else None."""
    labels = as_list(row.get("all_labels"))
    tl = as_list(row.get("true_labels"))
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or len(set(map(str, labels))) < len(labels)
    ):
        return None
    labels = [str(x) for x in labels]
    if isinstance(tl, dict):
        gold = [i for i, lab in enumerate(labels) if float(tl.get(lab, 0.0)) > 0.5]
    elif isinstance(tl, list):
        gold = [i for i, lab in enumerate(labels) if lab in set(map(str, tl))]
    else:
        return None
    return (labels, gold[0]) if len(gold) == 1 else None


def options_ranker(rows):
    """Out-of-fold options-only accuracy over rows [(prompt, labels, gold)]."""
    texts, groups, y, cues = [], [], [], []
    for i, (prompt, labels, gold) in enumerate(rows):
        lens = np.array([len(lab) for lab in labels], dtype=float)
        for j, lab in enumerate(labels):
            texts.append(f"{prompt} || {lab}")
            groups.append(i)
            y.append(int(j == gold))
            cues.append(
                [
                    lens[j] / (lens.mean() + 1e-9),
                    float(lens[j] == lens.max()),
                    float(lens[j] == lens.min()),
                ]
            )
    groups, y, cues = np.array(groups), np.array(y), csr_matrix(np.array(cues))
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(texts, y, groups):
        vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
        )
        xtr = hstack([vec.fit_transform([texts[k] for k in tr]), cues[tr]]).tocsr()
        xte = hstack([vec.transform([texts[k] for k in te]), cues[te]]).tocsr()
        if len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(xtr, y[tr])
        scores[te] = clf.decision_function(xte)
    right = []
    for i in range(len(rows)):
        m = groups == i
        right.append(int(y[m][np.argmax(scores[m])] == 1))
    return float(np.mean(right))


def audit(src, sample):
    chance = float(np.mean([1 / len(labels) for _, labels, _ in sample]))
    golds = Counter(labels[g] for _, labels, g in sample)
    top_label, _ = golds.most_common(1)[0]
    majority = []
    for _, labels, g in sample:
        if top_label in labels:
            majority.append(int(labels[g] == top_label))
        else:
            majority.append(1 / len(labels))
    longest = []
    for _, labels, g in sample:
        lens = [len(lab) for lab in labels]
        longest.append(int(lens[g] == max(lens) and lens.count(max(lens)) == 1))
    n_sets = len({tuple(sorted(labels)) for _, labels, _ in sample})
    out = {
        "src": src,
        "n_audited": len(sample),
        "chance": chance,
        "majority": float(np.mean(majority)),
        "majority_label": top_label[:80],
        "longest": float(np.mean(longest)),
        "n_label_sets": n_sets,
    }
    out["options"] = options_ranker(sample) if len(sample) >= 50 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--flag-gap", type=float, default=0.25)
    ap.add_argument("--flag-majority", type=float, default=0.9)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    res, seen, total, multi = defaultdict(list), Counter(), Counter(), Counter()
    with open(args.pool) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            src = row["src"]
            total[src] += 1
            sl = single_label(row)
            if sl is None:
                multi[src] += 1
                continue
            item = (str(row.get("prompt") or ""), sl[0], sl[1])
            seen[src] += 1
            if len(res[src]) < args.cap:
                res[src].append(item)
            else:
                k = rng.randrange(seen[src])
                if k < args.cap:
                    res[src][k] = item
    print(f"pool rows {sum(total.values())}, sources {len(total)}", flush=True)
    report = []
    for src in sorted(total, key=lambda s: -total[s]):
        if not res[src]:
            report.append(
                {
                    "src": src,
                    "rows": total[src],
                    "single_label": 0,
                    "multi_or_bad": multi[src],
                }
            )
            continue
        a = audit(src, res[src])
        a.update(rows=total[src], single_label=seen[src], multi_or_bad=multi[src])
        gap = (a["options"] - a["chance"]) if a["options"] is not None else 0.0
        a["options_gap"] = gap
        a["flag"] = bool(gap >= args.flag_gap or a["majority"] >= args.flag_majority)
        report.append(a)
        print(
            json.dumps(
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items()}
            ),
            flush=True,
        )
    (args.out / "c1_source_audit.json").write_text(json.dumps(report, indent=1))
    cols = [
        "src",
        "rows",
        "single_label",
        "n_label_sets",
        "chance",
        "majority",
        "longest",
        "options",
        "options_gap",
        "flag",
        "majority_label",
    ]
    with open(args.out / "c1_source_audit.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for a in report:
            f.write(
                "\t".join(
                    str(round(a[c], 4) if isinstance(a.get(c), float) else a.get(c, ""))
                    for c in cols
                )
                + "\n"
            )
    flagged = [a for a in report if a.get("flag")]
    print(
        f"flagged {len(flagged)} of {len(report)} sources, {sum(a['rows'] for a in flagged)} rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
