# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow", "numpy", "polars", "huggingface_hub>=1.21", "torch"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Schema-gen pilot, last step: screen the final decision rows (context + LLM-written question) and convert them.

Screen: the Stage 2 screen (prep_stage2.check: canary with the real BIG-bench GUID, exact text, 13-gram shingles on
text / prompt / labels, option sets) against the Decision Index 0.3 suite and our prepared eval files, on the
engine-mapped row (render_state(state) as text, instructions as prompt, option descriptions as labels). The
contexts were screened before (sg_contexts.py); this pass catches LLM-written instructions or options that copy
reference wording. Conversion: `gliclass_ub_train.py --convert` (engine mapping), so strings match eval.
Writes <name>_decision.jsonl, <name>.jsonl (gliclass rows), <name>_removed.jsonl, <name>_screen_stats.json to --out.

usage (HF Jobs):
  hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN -v hf://buckets/YOUR_NAME/dm-recipe:/b \\
    -v hf://buckets/YOUR_NAME/jobs-artifacts/s8-schemagen:/s:ro scripts/sg_screen.py \\
    --input /b/data/stage3/schemagen/final.jsonl --suite-repo YOUR_NAME/decision-index-suite-0.3
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, Path("/s")):
    if (p / "prep_stage2.py").exists():
        sys.path.insert(0, str(p))
        break
from gliclass_ub_train import decision_to_gliclass  # noqa: E402
from prep_stage2 import Hasher, check, load_reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--prepared", default="/b/data/prepared")
    ap.add_argument("--suite-repo", default=None)
    ap.add_argument("--suite-dir", default=None)
    ap.add_argument("--out", default="/b/data/stage3")
    # `hf jobs` swallows --name, so use --out-name on Jobs (--name kept for local runs)
    ap.add_argument("--out-name", "--name", dest="name", default="schemagen_pilot")
    ap.add_argument("--work", default="/tmp/sg_screen")
    args = ap.parse_args()
    hx = Hasher()
    report = {}
    ref = load_reference(args, hx, report)
    work, out = Path(args.work), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    kept, removed, skipped, flags = [], Counter(), Counter(), Counter()
    with open(args.input) as f, open(work / f"{args.name}_removed.jsonl", "w") as fr:
        for line in f:
            item = json.loads(line)
            g, why = decision_to_gliclass(item, "engine")
            if g is None:
                skipped[why] += 1
                continue
            hit, fl = check({"text": g["text"], "prompt": g["prompt"], "all_labels": g["all_labels"]}, ref, hx, None)
            if hit:
                removed[hit[0]] += 1
                fr.write(json.dumps({"id": item["id"], "src": item["src"], "reason": hit[0], "benchmark": hit[1],
                                     "field": hit[2]}) + "\n")
                continue
            flags.update(fl)
            kept.append(item)
    dec = work / f"{args.name}_decision.jsonl"
    with open(dec, "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in kept)
    trainer = next(p for p in (HERE / "gliclass_ub_train.py", Path("/s/gliclass_ub_train.py")) if p.exists())
    subprocess.run([sys.executable, str(trainer), "--convert", str(dec), str(work / f"{args.name}.jsonl"),
                    "--mapping", "engine"], check=True)
    stats = {"input": args.input, "kept": len(kept), "removed": dict(removed), "skipped": dict(skipped),
             "flags": dict(flags.most_common(30)), "reference": {k: v for k, v in report.items() if k != "reference"}}
    (work / f"{args.name}_screen_stats.json").write_text(json.dumps(stats, indent=1))
    for suffix in ("_decision.jsonl", ".jsonl", "_removed.jsonl", "_screen_stats.json"):
        shutil.copyfile(work / f"{args.name}{suffix}", out / f"{args.name}{suffix}")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
