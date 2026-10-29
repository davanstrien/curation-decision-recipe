# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "scikit-learn", "numpy"]
# ///
"""Catalogue real-label pilot: National Library of Scotland catalogue (CC0, NLS Data Foundry 2022) -> decision rows.

Records: Dublin Core parquet (`library-catalogue-metadata/nls-2022/parquet`). State = title + creator (first, roles
stripped) + description (if any) + year. Label = a curator-assigned topical subject heading (LCSH/FAST string; trailing
punctuation removed). Heading filter: topical only (no personal-name headings with dates, no headings equal to the
creator), frequency >= --min-freq in the sample, at most --per-heading gold rows per heading (flat label prior).
Records whose normalised title appears in our hv_* or bl_* eval/calib files are excluded (--exclude-evals).
Framings:
  subject_choice  "Which subject heading did the library assign?" gold + 1 near-miss (a heading of the most similar
                  other record by TF-IDF that shares no heading with this record) + distractors drawn in proportion to
                  gold frequency (options-only gate). None of the record's own headings is ever a distractor.
  subject_yesno   "Is this record catalogued under <heading>?" positive = own heading, negative = its near-miss.
usage: uv run cat_nls_build.py OUT_DIR --parquet nls-catalogue-4.parquet [...] --exclude-evals evals_*.jsonl
"""

import argparse
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

Q_CHOICE = [
    "Which subject heading did the library assign to this record?",
    "Pick the catalogue subject for this book.",
    "Which subject heading fits this catalogue record?",
    "Route this record to its subject heading.",
]
Q_YESNO = [
    "Is this record catalogued under the subject heading: {h}?",
    "Did the library assign the subject: {h}?",
]
PERSONAL = re.compile(r"\d{3,4}-|\b\d{4}\b|,\s*[A-Z][a-z]+\.?$")


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())[:60]


def clean_heading(h):
    return re.sub(r"[\s.,;:/]+$", "", h.strip())


def first(xs):
    return xs[0] if xs else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--exclude-evals", nargs="*", default=[])
    ap.add_argument("--min-freq", type=int, default=30)
    ap.add_argument("--per-heading", type=int, default=40)
    ap.add_argument("--max-records", type=int, default=12000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    excl = set()
    for pat in args.exclude_evals:
        for f in glob.glob(pat):
            for line in open(f):
                st = json.loads(line).get("state") or {}
                if isinstance(st, dict) and st.get("title"):
                    excl.add(norm_title(st["title"]))
    print(f"excluded eval titles: {len(excl)}", flush=True)
    cols = ["title", "creator", "subject", "description", "language", "year"]
    df = pl.concat([pl.read_parquet(p, columns=cols) for p in args.parquet])
    df = df.filter(
        (pl.col("subject").list.len() > 0) & (pl.col("title").list.len() > 0)
    )
    recs = []
    dropped = Counter()
    for r in df.iter_rows(named=True):
        title = first(r["title"]).strip(" /:")
        if norm_title(title) in excl:
            dropped["eval_title"] += 1
            continue
        creator = re.sub(
            r",?\s*(author|editor|translator|issuing body|compiler|illustrator)\.?",
            "",
            first(r["creator"]),
        ).strip(" ,.")
        heads = []
        for h in r["subject"]:
            h = clean_heading(h)
            if (
                not h
                or len(h) > 60
                or PERSONAL.search(h)
                or (creator and h.lower() in creator.lower())
            ):
                continue
            if h not in heads:
                heads.append(h)
        if not heads:
            dropped["no_topical_heading"] += 1
            continue
        desc = " ".join(
            d
            for d in r["description"]
            if not re.match(r"(Includes|Title from|System requirements)", d)
        )
        recs.append(
            {
                "title": title,
                "creator": creator,
                "desc": desc[:1200],
                "year": r["year"],
                "heads": heads,
                "lang": first(r["language"]),
            }
        )
    freq = Counter(h for r in recs for h in r["heads"])
    keep = {h for h, n in freq.items() if n >= args.min_freq}
    recs = [dict(r, heads=[h for h in r["heads"] if h in keep]) for r in recs]
    recs = [r for r in recs if r["heads"]]
    rng.shuffle(recs)
    per, chosen = Counter(), []
    for r in recs:  # one gold heading per record, flat over headings
        options = [h for h in r["heads"] if per[h] < args.per_heading]
        if not options:
            continue
        g = rng.choice(options)
        per[g] += 1
        chosen.append(dict(r, gold=g))
        if len(chosen) >= args.max_records:
            break
    print(
        f"records {len(recs)}, chosen {len(chosen)}, headings {len(per)}; dropped {dict(dropped)}",
        flush=True,
    )

    def state(r):
        s = {"title": r["title"]}
        if r["creator"]:
            s["creator"] = r["creator"]
        if r["year"]:
            s["year"] = r["year"]
        if r["desc"]:
            s["description"] = r["desc"]
        return s

    text = [f"{r['title']} {r['desc']}" for r in chosen]
    x = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"
    ).fit_transform(text)
    nn = NearestNeighbors(n_neighbors=20, metric="cosine").fit(x)
    _, idx = nn.kneighbors(x)
    heads_list = sorted(per)
    weights_all = {h: per[h] for h in heads_list}
    rows = []
    for i, r in enumerate(chosen):
        own = set(r["heads"])
        near = None
        for j in idx[i][1:]:
            cand = [h for h in chosen[j]["heads"] if h not in own and h in per]
            if cand and not (set(chosen[j]["heads"]) & own):
                near = rng.choice(cand)
                break
        if near is None:
            continue
        pool = [h for h in heads_list if h not in own and h != near]
        w = [weights_all[h] for h in pool]
        distract = [near]
        while len(distract) < args.k - 1:
            k = rng.choices(range(len(pool)), weights=w)[0]
            distract.append(pool.pop(k))
            w.pop(k)
        opts = [r["gold"]] + distract
        rng.shuffle(opts)
        crit = {f"S{n}": h for n, h in enumerate(opts)}
        gold_key = f"S{opts.index(r['gold'])}"
        base = {
            "src": "nls:subject",
            "licence": "cc0-1.0",
            "state": state(r),
            "target_source": "metadata",
            "meta": {"gold_heading": r["gold"], "near": near, "lang": r["lang"]},
        }
        rows.append(
            {
                **base,
                "id": f"nls-choice-{i}",
                "family": "catalogue.subject_choice",
                "question": {
                    "type": "choice",
                    "instructions": rng.choice(Q_CHOICE),
                    "criteria": crit,
                },
                "target": {gold_key: 1.0},
            }
        )
        pos = rng.random() < 0.5
        h = r["gold"] if pos else near
        rows.append(
            {
                **base,
                "id": f"nls-yesno-{i}",
                "family": "catalogue.subject_yesno",
                "question": {
                    "type": "choice",
                    "instructions": rng.choice(Q_YESNO).format(h=h),
                    "criteria": {"yes": "Yes", "no": "No"},
                },
                "target": {"yes" if pos else "no": 1.0},
            }
        )
    rng.shuffle(rows)
    with open(args.out / "nls_decision.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    stats = {
        "records_chosen": len(chosen),
        "rows": dict(Counter(r["family"] for r in rows)),
        "headings": len(per),
        "excluded_eval_titles": len(excl),
        "dropped": dict(dropped),
        "lang": dict(Counter(r["meta"]["lang"] for r in rows).most_common(8)),
        "with_description": float(np.mean([bool(r["desc"]) for r in chosen])),
    }
    (args.out / "nls_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
