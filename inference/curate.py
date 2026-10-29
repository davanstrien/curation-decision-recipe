# /// script
# requires-python = ">=3.11"
# dependencies = ["gliclass>=0.1.21", "torch", "transformers>=5.20", "kernels>=0.17.2,<0.18", "numpy",
#                 "datasets>=4", "huggingface_hub>=1.32"]
#
# # Defaults for `hf jobs uv run`; command-line flags override them.
# [tool.hf-jobs]
# flavor = "l4x1"
# timeout = "1h"
# secrets = ["HF_TOKEN"]
# ///
"""Curate a Hub dataset with davanstrien/ultrabert-380m-curation in one command on Hugging Face Jobs.

Reads a text column from a Hub dataset, asks four curation questions about every row (topic, format, educational
value, reasoning depth), and pushes a dataset repo with two configs: `scored` (every row with its answers) and
`subset` (rows with educational value "useful for education" or higher that are not spam, a listing, a product page
or incomplete). Needs a CUDA GPU from the Ampere generation or newer (l4x1, a10g, a100, rtx-pro-6000; not t4).

    hf download davanstrien/ultrabert-380m-curation curate.py --local-dir .
    hf jobs uv run curate.py -- --dataset HuggingFaceFW/fineweb-edu --config sample-10BT --text-column text \\
        --limit 10000 --out-repo YOUR_NAME/fineweb-edu-curated

The model was trained on web text cut at 3,000 characters, so only the first 3,000 characters of each row are
scored. The pushed rows keep their full text and every other column.
"""

import argparse
import importlib.util
import itertools
import sys
import time

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

MODEL = "davanstrien/ultrabert-380m-curation"
QUESTIONS = {
    "topic": ("Classify the topic of this web page.", [
        "Adult Content", "Art & Design", "Crime & Law", "Education & Jobs", "Electronics & Hardare",
        "Entertainment", "Fashion & Beauty", "Finance & Business", "Food & Dining", "Games", "Health",
        "History & Geography", "Home & Hobbies", "Industrial", "Literature", "Politics", "Religion",
        "Science, Math & Technology", "Social Life", "Software", "Software Development", "Sports & Fitness",
        "Transportation", "Travel & Tourism"]),
    "format": ("Classify the format of this web page.", [
        "Academic Writing", "Content Listing", "Creative Writing", "Customer Support Page",
        "Discussion Forum / Comment Section", "FAQs", "Incomplete Content", "Knowledge Article",
        "Legal Notices", "Listicle", "News Article", "Nonfiction Writing", "Organizational About Page",
        "Organizational Announcement", "Personal About Page", "Personal Blog", "Product Page", "Q&A Forum",
        "Spam / Ads", "Structured Data", "Technical Writing", "Transcript / Interview",
        "Tutorial / How-To Guide", "User Reviews"]),
    "edu": ("Classify the educational value of this web page.", [
        "not educational", "minimal educational value", "some educational value", "useful for education",
        "highly educational", "outstanding educational value"]),
    "reasoning": ("Classify the depth of reasoning of this web page.", [
        "No Reasoning", "Basic Reasoning", "Intermediate Reasoning", "Advanced Reasoning", "Indeterminate"]),
}
# The topic label is spelled as in the WebOrganizer training labels ("Hardare"); the output uses the correct spelling.
RENAME = {"Electronics & Hardare": "Electronics & Hardware"}
DROP_FORMATS = {"Spam / Ads", "Content Listing", "Product Page", "Incomplete Content"}
MAX_CHARS = 3000


def load_rows(args):
    """The first --limit rows (all rows if --limit is not set), streamed so only the needed files are read."""
    stream = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    if args.limit:
        stream = itertools.islice(stream, args.limit)
    return list(stream)


def load_fast_scorer():
    """Import FastScorer from fast_inference.py in the model repo (a single file, not a pip package)."""
    path = hf_hub_download(MODEL, "fast_inference.py")
    spec = importlib.util.spec_from_file_location("fast_inference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fast_inference"] = module  # registered by name, as a normal import would be
    spec.loader.exec_module(module)
    return module.FastScorer(MODEL)  # loads, fuses and compiles the model (about a minute)


def answer_columns(scorer, texts):
    """One column per answer: the top label and its probability; the level and expected level for education."""
    cols = {f"{name}{suffix}": [] for name in QUESTIONS for suffix in ("", "_p")}
    cols |= {"edu_level": [], "edu_expected": [], "scored": []}
    requests = [(t[:MAX_CHARS], labels, prompt) for prompt, labels in QUESTIONS.values() for t in texts]
    results = scorer.score_requests(requests)
    for qi, (name, (_, labels)) in enumerate(QUESTIONS.items()):
        for di in range(len(texts)):
            probs = results[qi * len(texts) + di]
            top = max(probs, key=probs.get) if probs else ""
            cols[name].append(RENAME.get(top, top))
            cols[f"{name}_p"].append(probs[top] if probs else 0.0)
            if name == "edu":
                cols["edu_level"].append(labels.index(top) if probs else -1)
                cols["edu_expected"].append(sum(i * probs[x] for i, x in enumerate(labels)) if probs else -1.0)
    for di in range(len(texts)):
        # A request is unscored only when its labels did not fit in the token limit (None from score_requests).
        cols["scored"].append(all(results[qi * len(texts) + di] is not None for qi in range(len(QUESTIONS))))
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="Hub dataset id to read")
    ap.add_argument("--config", default=None, help="dataset config (subset) name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    ap.add_argument("--out-repo", required=True, help="dataset repo to push, e.g. YOUR_NAME/my-data-curated")
    ap.add_argument("--public", action="store_true", help="push a public repo (default: private)")
    ap.add_argument("--chunk", type=int, default=8192, help="rows per packed scoring call")
    args = ap.parse_args()

    start = time.time()
    rows = load_rows(args)
    if not rows:
        sys.exit("no rows read")
    if args.text_column not in rows[0]:
        sys.exit(f"column {args.text_column!r} not found; columns: {list(rows[0])}")
    print(f"read {len(rows):,} rows in {time.time() - start:.0f}s")

    scorer = load_fast_scorer()
    texts = [r[args.text_column] or "" for r in rows]
    cols = {}
    t0 = time.time()
    for i in range(0, len(texts), args.chunk):
        part = answer_columns(scorer, texts[i:i + args.chunk])
        for k, v in part.items():
            cols.setdefault(k, []).extend(v)
        done = min(i + args.chunk, len(texts))
        print(f"scored {done:,}/{len(texts):,} rows ({done / (time.time() - t0):.0f} rows/s)")
    cols["in_subset"] = [ok and lvl >= 3 and fmt not in DROP_FORMATS
                         for ok, lvl, fmt in zip(cols["scored"], cols["edu_level"], cols["format"], strict=True)]

    scored = Dataset.from_list(rows)
    for k, v in cols.items():
        scored = scored.add_column(k, v)
    subset = scored.filter(lambda r: r["in_subset"])
    private = not args.public
    scored.push_to_hub(args.out_repo, config_name="scored", private=private)
    subset.push_to_hub(args.out_repo, config_name="subset", private=private)
    print(f"{len(scored):,} rows scored ({sum(not s for s in cols['scored'])} unscored), {len(subset):,} in the "
          f"subset ({len(subset) / len(scored):.1%}); pushed https://huggingface.co/datasets/{args.out_repo} "
          f"in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
