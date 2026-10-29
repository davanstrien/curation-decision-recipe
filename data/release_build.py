# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "huggingface_hub>=1.32"]
# ///
"""Build the released dataset from the training mix (cur-cand v1) and upload it to a private dataset repo.

One parquet per config (source group), under data/<config>/. Release rules:
- base: keep rows whose source repo has a verified permissive, share-alike or other open licence
  (analysis/release/base_licence_map.json); drop GPL, unresolved and no-licence rows. Fill `licence` per row.
- web: page text is not redistributed; `state` is set to "" and `source_dataset` / `source_row` point at the
  source row (train split) so the text can be rebuilt.
- nested fields (state when it is a record, question, target, true_labels) are stored as JSON strings;
  `state_is_record` says whether a decision row's state was a record (dict or list) or text.
usage (CPU Job, dm-curation mounted at /c): python release_build.py --repo YOUR_NAME/<name> [--dry-run]
"""

import argparse
import collections
import json
import tempfile
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi

MIX = Path("/c/mixes/cur-cand/v1")
LICMAP = Path("/c/analysis/release/base_licence_map.json")
CONFIG = {
    "base-r1g": "base", "decisionbank": "decisionbank", "prompt-slice-tpm": "prompt", "web-slice-tpm": "web",
    "nls": "nls", "tool": "tool", "harvard-modern": "harvard", "fh-arxiv": "arxiv", "procpile": "procpile",
}
# Licence per config where the whole source shares one licence (dataset card, Sources table); None = per src.
CONFIG_LICENCE = {"nls": "cc0-1.0", "harvard": "cc0-1.0", "arxiv": "cc0-1.0", "procpile": "cc-by-4.0"}
# Licence by src prefix for configs that mix sources (dataset card, Sources table).
SRC_LICENCE = {"numina_": "apache-2.0", "wi_": "apache-2.0", "dm_": "mit", "arena_": "cc-by-4.0",
               "fineweb_edu": "odc-by", "weborganizer_": None,
               "tool": "per package: mit / bsd / apache-2.0 / psf"}
# Base rows: the class follows the source repo's resolved licence, not the mix-level "permissive" default.
BASE_CLASS = {"permissive": "permissive", "sa": "sa", "other_open": "other_open"}
# Web rows: the id suffix is the row index in the source dataset's train split (scripts/cur_build.py).
WEB_SOURCE = {"cur-wo-topic-": "WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8",
              "cur-wo-format-": "WebOrganizer/FormatAnnotations-Llama-3.1-405B-FP8",
              "cur-fwe-": "HuggingFaceFW/fineweb-edu-llama3-annotations"}
COLUMNS = ["id", "config", "src", "row_type", "teacher", "licence_class", "licence", "state", "state_is_record", "question", "target",
           "text", "all_labels", "true_labels", "prompt", "multi_label", "source_dataset", "source_row"]


def as_json(v):
    return None if v is None else json.dumps(v, ensure_ascii=False, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--create-pr", action="store_true", help="upload as a pull request to the dataset repo")
    args = ap.parse_args()

    licmap = json.loads(LICMAP.read_text())
    man = pl.read_parquet(MIX / "manifest.parquet", columns=["id", "source", "row_type"])
    source_of = dict(zip(man["id"].to_list(), man["source"].to_list()))

    rows = collections.defaultdict(list)
    stats = collections.Counter()
    extra_keys = collections.Counter()
    no_licence = collections.Counter()
    with open(MIX / "mix.jsonl") as f:
        for line in f:
            r = json.loads(line)
            config = CONFIG[Path(source_of[r["id"]]).parent.name]
            stats[f"{config}:in"] += 1
            licence = r.get("licence") or CONFIG_LICENCE.get(config)
            if licence is None:
                for prefix, lic in SRC_LICENCE.items():
                    if r.get("src", "").startswith(prefix):
                        licence = lic
                        break
            if config == "base":
                repo = "/".join(r["src"].split("/")[:2])
                entry = licmap[repo]
                if not entry["keep"]:
                    stats[f"base:drop_{entry['status']}"] += 1
                    continue
                licence = entry["licence"]
                licence_class = BASE_CLASS[entry["status"]]
            if config != "base":
                licence_class = r.get("licence_class")
            if licence and ("-sa" in licence or "sharing" in licence):
                assert licence_class == "sa", (r["id"], licence, licence_class)
            row_type = "gliclass" if "all_labels" in r else "decision"
            state = r.get("state")
            source_dataset = source_row = None
            if config == "web":
                state = ""
                prefix = next(p for p in WEB_SOURCE if r["id"].startswith(p))
                source_dataset, source_row = WEB_SOURCE[prefix], int(r["id"][len(prefix):])
            out = {
                "id": r["id"], "config": config, "src": r.get("src"), "row_type": row_type,
                "teacher": r.get("teacher"), "licence_class": licence_class, "licence": licence,
                "state": state if state is None or isinstance(state, str) else as_json(state),
                "state_is_record": isinstance(state, (dict, list)) if row_type == "decision" else None,
                "question": as_json(r.get("question")), "target": as_json(r.get("target")),
                "text": r.get("text"), "all_labels": r.get("all_labels"),
                "true_labels": as_json(r.get("true_labels")), "prompt": r.get("prompt"),
                "multi_label": bool(r.get("multi_label", False)),
                "source_dataset": source_dataset, "source_row": source_row,
            }
            for k in r:
                if k not in out:
                    extra_keys[k] += 1
            if licence is None:
                no_licence[(config, out["src"])] += 1
            rows[config].append(out)
            stats[f"{config}:out"] += 1
            stats[f"class:{licence_class}"] += 1

    schema = {c: pl.Utf8 for c in COLUMNS}
    schema["all_labels"] = pl.List(pl.Utf8)
    schema["multi_label"] = pl.Boolean
    schema["state_is_record"] = pl.Boolean
    schema["source_row"] = pl.Int64
    print(json.dumps(dict(sorted(stats.items())), indent=1))
    print("dropped keys (not released):", dict(extra_keys))
    print("no licence (config, src):", no_licence.most_common())
    print("total out:", sum(len(v) for v in rows.values()))
    for config, rs in rows.items():
        ex = {k: (v[:120] if isinstance(v, str) else v) for k, v in rs[0].items()}
        print(config, json.dumps(ex, ensure_ascii=False)[:600])
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        for config, rs in rows.items():
            p = Path(d) / "data" / config
            p.mkdir(parents=True)
            pl.DataFrame(rs, schema=schema).write_parquet(p / "train.parquet")
        (Path(d) / "release_stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1))
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=d,
                          commit_message="Add release build of the cur-cand v1 training mix",
                          create_pr=args.create_pr)
    print("uploaded", args.repo)


if __name__ == "__main__":
    main()
