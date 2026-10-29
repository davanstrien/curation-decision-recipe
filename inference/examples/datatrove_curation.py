"""Curate a web dataset with davanstrien/ultrabert-380m-curation and datatrove on Hugging Face Jobs.

Reads parquet files from a Hub dataset, asks four curation questions about every page (topic, format, educational
value, reasoning depth), writes every page with its answers, and writes a filtered subset (educational value
"useful for education" or higher, and not spam, a listing, a product page or incomplete). Scoring uses the packed GPU
path from the model repo (fast_inference.py). Pages are cut at 3,000 characters for the model only; the written rows
keep the full text, and no rows are dropped from the scored output.

The script is the coordinator: run it on your machine and it launches the GPU Jobs (one per task). Use the same
datatrove version on both sides:

    OUT="hf://buckets/YOUR_NAME/YOUR_BUCKET/run1"
    uv run --python 3.12 --with "datatrove[io,processing]==0.10.1" --with "huggingface_hub>=1.32" \\
        datatrove_curation.py --out "$OUT" --limit 2000

Example (FineWeb CC-MAIN-2025-26, --glob "000_0000*.parquet" --limit 170000 --tasks 6): 1,020,000 pages in about
19 minutes, about $5; 63,788 pages in the subset.
"""

import argparse

from datatrove.executor.jobs import JobsPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

MODEL = "davanstrien/ultrabert-380m-curation"


class CurationScorer(PipelineStep):
    name = "curation scorer"
    type = "🏷 - TAGGER"

    def __init__(self, model_id=MODEL, chunk_docs=2048):
        super().__init__()
        self.model_id = model_id
        self.chunk_docs = chunk_docs

    def run(self, data, rank=0, world_size=1):
        # Everything the step needs is imported here: datatrove ships this class to the Job, not this module.
        import os
        import sys

        from huggingface_hub import hf_hub_download

        sys.path.insert(0, os.path.dirname(hf_hub_download(self.model_id, "fast_inference.py")))
        from fast_inference import FastScorer

        questions = {
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
        rename = {"Electronics & Hardare": "Electronics & Hardware"}
        drop_formats = {"Spam / Ads", "Content Listing", "Product Page", "Incomplete Content"}
        scorer = FastScorer(self.model_id)

        def score_chunk(docs):
            texts = [d.text[:3000] for d in docs]
            requests = [(t, labels, prompt) for prompt, labels in questions.values() for t in texts]
            results = scorer.score_requests(requests)
            for qi, (name, (_, labels)) in enumerate(questions.items()):
                for di, d in enumerate(docs):
                    probs = results[qi * len(docs) + di]
                    top = max(probs, key=probs.get) if probs else ""
                    d.metadata[name] = rename.get(top, top)  # "" only if labels were truncated (never at 3,000 chars)
                    d.metadata[f"{name}_p"] = probs[top] if probs else 0.0
                    if name == "edu":
                        d.metadata["edu_level"] = labels.index(top) if probs else -1
                        d.metadata["edu_expected"] = sum(i * probs[x] for i, x in enumerate(labels)) if probs else -1.0
            for d in docs:
                for k in list(d.metadata):  # nested or missing source values become strings: one parquet schema
                    if k not in questions and not k.startswith(tuple(questions)):
                        v = d.metadata[k]
                        if v is None or not isinstance(v, (str, int, float, bool)):
                            d.metadata[k] = "" if v is None else str(v)
                d.metadata["cut_at_3000_chars"] = len(d.text) > 3000
                d.metadata["in_subset"] = d.metadata["edu_level"] >= 3 and d.metadata["format"] not in drop_formats

        buf = []
        for doc in data:
            buf.append(doc)
            if len(buf) >= self.chunk_docs:
                with self.track_time(unit="chunk"):
                    score_chunk(buf)
                self.stat_update("docs_scored", value=len(buf))
                yield from buf
                buf = []
        if buf:
            score_chunk(buf)
            self.stat_update("docs_scored", value=len(buf))
            yield from buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output prefix, e.g. hf://buckets/<you>/<bucket>/run1")
    ap.add_argument("--source", default="hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2025-26")
    ap.add_argument("--glob", default="*.parquet")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--id-key", default="id")
    ap.add_argument("--limit", type=int, default=-1, help="pages per task; -1 = all")
    ap.add_argument("--tasks", type=int, default=1, help="parallel GPU Jobs; input files are split across them")
    ap.add_argument("--flavor", default="rtx-pro-6000")
    ap.add_argument("--timeout", default="2h")
    args = ap.parse_args()
    JobsPipelineExecutor(
        job_name="curation-demo",
        pipeline=[
            ParquetReader(args.source, glob_pattern=args.glob, limit=args.limit, text_key=args.text_key,
                          id_key=args.id_key),
            CurationScorer(),
            ParquetWriter(f"{args.out}/scored", max_file_size=64 * 2**20),
            LambdaFilter(lambda doc: doc.metadata["in_subset"]),
            ParquetWriter(f"{args.out}/subset", max_file_size=64 * 2**20),
        ],
        tasks=args.tasks,
        workers=args.tasks,
        flavor=args.flavor,
        python="3.12",
        dependencies=["datatrove[io,processing]==0.10.1", "huggingface_hub>=1.32", "gliclass>=0.1.21", "torch",
                      "transformers>=5.20", "kernels>=0.17.2,<0.18", "numpy"],
        logging_dir=f"{args.out}/logs",
        timeout=args.timeout,
        max_retries=0,  # a retry reruns the whole task on the GPU; rerun the script to resume incomplete tasks
    ).run()


if __name__ == "__main__":
    main()
