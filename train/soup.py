# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "safetensors", "numpy"]
# ///
"""Uniform weight soup (2026-10-27, idea from the DecisionBank / Jeff 1.1 card: each Jeff model averages three
training runs in weight space). Averages model.safetensors of N run model dirs (same architecture; float32
accumulation, saved in the first model's dtypes) and copies config / tokenizer files from the first.
Caveat: our seeds also change the head initialisation, so the runs do not share a full starting point (Jeff's runs
start from one pretrained LLM); a broken soup shows up as a large eval drop.
Note: a full soup of our seeds gives near-flat logits (head tensors cancel). --average-prefix P
averages only tensors whose name starts with P (e.g. model.encoder_model) and takes the rest from the first model.
usage: uv run scripts/soup.py [--average-prefix P] OUT_DIR MODEL_DIR MODEL_DIR [...]
"""

import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    args = sys.argv[1:]
    prefix = ""
    if args and args[0] == "--average-prefix":
        prefix, args = args[1], args[2:]
    out, dirs = Path(args[0]), [Path(d) for d in args[1:]]
    assert len(dirs) >= 2, "need at least two models"
    out.mkdir(parents=True, exist_ok=False)
    first = load_file(dirs[0] / "model.safetensors")
    acc = {k: v.float().clone() for k, v in first.items()}
    for d in dirs[1:]:
        sd = load_file(d / "model.safetensors")
        assert sd.keys() == acc.keys(), f"key mismatch in {d}"
        for k, v in sd.items():
            assert v.shape == acc[k].shape, (d, k)
            acc[k] += v.float()
    soup = {k: (v / len(dirs)).to(first[k].dtype).contiguous() if k.startswith(prefix) else first[k].contiguous()
            for k, v in acc.items()}
    n_avg = sum(k.startswith(prefix) for k in soup)
    save_file(soup, out / "model.safetensors", metadata={"format": "pt"})
    for f in dirs[0].iterdir():
        if f.name != "model.safetensors" and f.is_file():
            shutil.copy(f, out / f.name)
    (out / "SOUP.json").write_text(json.dumps({"members": [str(d) for d in dirs], "average_prefix": prefix,
                                               "averaged_tensors": n_avg}, indent=1))
    print(f"soup of {len(dirs)} -> {out} ({n_avg} of {len(soup)} tensors averaged)")


if __name__ == "__main__":
    main()
