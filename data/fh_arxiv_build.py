# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "requests", "scikit-learn", "numpy"]
# ///
"""Real-label pilot from the Hub firehose: arXiv papers (dated after the DI suite) -> decision rows.

Input: Hub firehose items (a snapshot of Hub activity; source == "paper"): arXiv id, title, categories (author-chosen,
arXiv-moderated; first = primary). Abstracts are fetched from the arXiv API (metadata incl. abstracts is CC0),
cached in --cache. No model labels are used.
Two framings, both with real labels:
  cat    "Which arXiv category is this paper's primary category?" State = title + abstract. Options = gold primary +
         --k-cat - 1 distractors from the frequent categories; one distractor is the category most often co-listed
         with the gold (natural near-miss); none of the paper's own cross-lists is ever a distractor. Gold classes
         are capped per class (--per-class) to flatten the label prior. Options are category names + codes.
  title  "Which title belongs to this abstract?" State = abstract. Options = gold title + the --k-title - 1 most
         similar other titles by TF-IDF cosine (hard distractors). Label = the real pairing.
usage: uv run fh_arxiv_build.py firehose_all.parquet OUT_DIR [--cache abstracts.jsonl]
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import polars as pl
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
CAT_NAMES = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GR": "Graphics",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control (EESS)",
    "stat.ML": "Machine Learning (Statistics)",
    "math.OC": "Optimization and Control",
    "q-bio.NC": "Neurons and Cognition",
}
CAT_Q = [
    "Which arXiv category is this paper's primary category?",
    "Classify this paper into its main arXiv category.",
    "Route this paper to the arXiv listing it belongs to.",
    "What is the primary subject area of this paper?",
]
TITLE_Q = [
    "Which title belongs to this abstract?",
    "Pick the paper title that matches this abstract.",
    "Which of these is the title of the paper described?",
]


def fetch_abstracts(ids, cache):
    have = {}
    if cache.exists():
        for line in open(cache):
            r = json.loads(line)
            have[r["id"]] = r["abstract"]
    todo = [i for i in ids if i not in have]
    with open(cache, "a") as f:
        for k in range(0, len(todo), 100):
            batch = todo[k : k + 100]
            for attempt in range(4):
                try:
                    resp = requests.get(
                        API,
                        params={"id_list": ",".join(batch), "max_results": 100},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    break
                except requests.RequestException:
                    time.sleep(10 * (attempt + 1))
            else:
                continue
            root = ElementTree.fromstring(resp.text)
            for e in root.findall("a:entry", NS):
                aid = e.find("a:id", NS).text.rsplit("/", 1)[-1].split("v")[0]
                abstract = " ".join(e.find("a:summary", NS).text.split())
                have[aid] = abstract
                f.write(json.dumps({"id": aid, "abstract": abstract}) + "\n")
            print(f"abstracts {len(have)} / {len(ids)}", flush=True)
            time.sleep(3)
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out", type=Path)
    ap.add_argument("--cache", type=Path, default=Path("abstracts.jsonl"))
    ap.add_argument("--k-cat", type=int, default=6)
    ap.add_argument("--k-title", type=int, default=5)
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(args.items).filter(pl.col("source") == "paper").unique("id")
    papers = [r for r in df.iter_rows(named=True) if r["categories"]]
    abstracts = fetch_abstracts([p["id"] for p in papers], args.cache)
    papers = [
        dict(p, abstract=abstracts[p["id"]])
        for p in papers
        if len(abstracts.get(p["id"], "")) > 200
    ]
    print(f"papers with abstracts: {len(papers)}", flush=True)
    prim = Counter(p["categories"][0] for p in papers)
    frequent = [c for c, n in prim.items() if n >= 40 and c in CAT_NAMES]
    co = {c: Counter() for c in frequent}
    for p in papers:
        g = p["categories"][0]
        if g in co:
            co[g].update(c for c in p["categories"][1:] if c in frequent and c != g)
    rows, per_class = [], Counter()
    order = papers[:]
    rng.shuffle(order)
    for p in order:  # framing 1: primary category
        g = p["categories"][0]
        if g not in frequent or per_class[g] >= args.per_class:
            continue
        own = set(p["categories"])
        pool = [c for c in frequent if c not in own]
        near = [c for c, _ in co[g].most_common() if c not in own][:1]
        # distractors drawn in proportion to how often each category is gold, so a category's frequency in the
        # options says nothing about whether it is the answer (options-only gate)
        cand = [c for c in pool if c not in near]
        weights = [min(prim[c], args.per_class) for c in cand]
        distract = list(near)
        while len(distract) < args.k_cat - 1 and cand:
            k = rng.choices(range(len(cand)), weights=weights)[0]
            distract.append(cand.pop(k))
            weights.pop(k)
        opts = [g] + distract
        rng.shuffle(opts)
        style = rng.choice(["name", "code", "both"])
        crit = {
            c: CAT_NAMES[c]
            if style == "name"
            else c
            if style == "code"
            else f"{CAT_NAMES[c]} ({c})"
            for c in opts
        }
        rows.append(
            {
                "id": f"fh-arxiv-cat-{p['id']}",
                "src": "firehose:arxiv_cat",
                "licence": "cc0-1.0",
                "family": "arxiv.primary_category",
                "state": f"{p['title']}\n\n{p['abstract']}",
                "question": {
                    "type": "choice",
                    "instructions": rng.choice(CAT_Q),
                    "criteria": crit,
                },
                "target": {g: 1.0},
                "target_source": "metadata",
                "meta": {"near": near, "created": p["created_at"]},
            }
        )
        per_class[g] += 1
    n_cat = len(rows)
    titles = [p["title"] for p in papers]
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"
    )
    xa = vec.fit_transform([p["abstract"] for p in papers])
    xt = vec.transform(titles)
    sims = (xa @ xt.T).tocsr()
    for i, p in enumerate(papers):  # framing 2: title <-> abstract
        s = np.asarray(sims[i].todense()).ravel()
        s[i] = -1
        cand = [
            j
            for j in np.argsort(-s)[: args.k_title * 3]
            if titles[j].lower() != p["title"].lower()
        ]
        if len(cand) < args.k_title - 1:
            continue
        opts = [i] + cand[: args.k_title - 1]
        rng.shuffle(opts)
        crit = {f"T{k}": titles[j] for k, j in enumerate(opts)}
        gold = f"T{opts.index(i)}"
        rows.append(
            {
                "id": f"fh-arxiv-title-{p['id']}",
                "src": "firehose:arxiv_title",
                "licence": "cc0-1.0",
                "family": "arxiv.title_match",
                "state": p["abstract"],
                "question": {
                    "type": "choice",
                    "instructions": rng.choice(TITLE_Q),
                    "criteria": crit,
                },
                "target": {gold: 1.0},
                "target_source": "metadata",
                "meta": {"created": p["created_at"]},
            }
        )
    rng.shuffle(rows)
    with open(args.out / "fh_arxiv_decision.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    stats = {
        "papers": len(papers),
        "frequent_cats": len(frequent),
        "cat_rows": n_cat,
        "title_rows": len(rows) - n_cat,
        "per_class": dict(per_class),
    }
    (args.out / "fh_arxiv_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
