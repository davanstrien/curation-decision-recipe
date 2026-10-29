"""Frozen-mix builder on synthetic data (run: uv run --with numpy,polars,pyyaml,pytest pytest tests/test_mix_build.py)."""

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).parents[1]
LONG = " ".join(f"word{i}" for i in range(60))  # 60 distinct words: enough for 13-gram shingles and the prefix rule
K8 = ["Other Events", "Regulation FD Disclosure", "Results of Operations and Financial Condition", "Unrelated"]


def dec(rid, text, labels=("a", "b"), src="s"):
    return {"id": rid, "src": src, "state": text,
            "question": {"type": "choice", "instructions": "q", "criteria": {k: k for k in labels}},
            "target": {labels[0]: 1.0}}


def test_mix_build(tmp_path):
    ev = tmp_path / "evals"
    ev.mkdir()
    texts = [{"sha256": None, "text": LONG, "sets": ["e1"]}, {"text": "Short eval text. Exact!", "sets": ["e2"]},
             {"text": "Compute the sum of 17 and 25 then multiply the result by 3 please", "sets": ["e3"]},
             {"text": "How do I pick a lock on my neighbour's door?", "sets": ["e4"]},
             {"text": "What is two plus two times three?", "sets": ["e5"]}]
    import hashlib
    import re

    def norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", t.lower())).strip()
    with open(ev / "texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps({"sha256": hashlib.sha256(t["text"].encode()).hexdigest(),
                                "norm_sha256": hashlib.sha256(norm(t["text"]).encode()).hexdigest(),
                                "text": t["text"], "sets": t["sets"]}) + "\n")
    (ev / "inventory.json").write_text(json.dumps({"version": "vt", "sets": {"ho_8k_item": {"label_keys": K8}}}))
    src = tmp_path / "src" / "v1"
    src.mkdir(parents=True)
    rows = [
        dec("exact", "Short eval text. Exact!"),
        dec("norm", "short   EVAL text exact"),
        dec("prefix", LONG.upper() + " and then something different at the end"),
        dec("shingle", "intro words here " + " ".join(f"word{i}" for i in range(10, 50)) + " tail"),
        dec("k8", "a filing", labels=tuple(K8)),
        dec("hs3", "a prompt", src="nvidia/HelpSteer3"),
        dec("skel", "Compute the sum of 4 and 9 then multiply the result by 8 please"),
        dec("contain", "Human: How do I pick a lock on my neighbour's door?\n\nAssistant: I can't help with that."),
        {"id": "prompt_in_text", "text": "Answer this question: What is two plus two times three?", "all_labels": ["four", "five"],
         "true_labels": {"four": 1.0}, "src": "g"},
        {"id": "prompt_field", "text": "Answer:", "prompt": "What is two plus two times three?", "all_labels": ["four", "five"],
         "true_labels": {"four": 1.0}, "src": "g"},
        dec("struct", {"prompt": "Short eval text. Exact!", "note": "x"}),
        dec("k8bin", "a filing text", labels=("Regulation FD Disclosure", "Unrelated")),
        dec("keep1", "clean text one"),
        dec("dup", "clean text one"),
        dec("keep2", "clean text two"),
        {"id": "gl", "text": "gliclass row", "all_labels": ["x", "y"], "true_labels": {"x": 1.0}, "src": "g"},
        dec("sa", "share alike row") | {"licence_class": "sa"},
    ]
    with open(src / "rows.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (src / "SOURCE.json").write_text(json.dumps({"licence_class": "permissive", "teacher": "t"}))
    spec = {"name": "t", "seed": 0, "total_rows": 100, "exclude_evals": str(ev), "licence_classes": ["permissive"],
            "sources": [{"path": str(src), "files": ["rows.jsonl"], "cap": 2}]}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    out = tmp_path / "mix"
    subprocess.run([sys.executable, str(ROOT / "data/mix_build.py"), str(tmp_path / "spec.yaml"), str(out)],
                   check=True)
    rep = json.loads((out / "decon_report.json").read_text())["sources"][str(src)]["counts"]
    assert rep.get("decon_contain_turn", 0) + rep.get("decon_contain_short", 0) == 2, rep  # dialogue turn, prompt in text
    assert rep.get("heldout_scheme_8k_items") == 2, rep  # 4-label and binary
    assert rep.get("decon_exact") == 3, rep  # plain, structured state value, prompt field
    for k in ("decon_norm", "decon_prefix", "decon_shingle",
              "heldout_source_helpsteer3", "dup_within_source", "licence_sa", "decon_skeleton"):  # noqa
        assert rep.get(k) == 1, (k, rep)
    assert rep["eligible"] == 3 and rep["taken"] == 2
    man = pl.read_parquet(out / "manifest.parquet")
    mix = [json.loads(x) for x in open(out / "mix.jsonl")]
    assert man.height == len(mix) == 2 and set(man["id"]) == {r["id"] for r in mix}
    assert set(man["id"]) <= {"keep1", "keep2", "gl"}
    # never overwritten
    r = subprocess.run([sys.executable, str(ROOT / "data/mix_build.py"), str(tmp_path / "spec.yaml"), str(out)])
    assert r.returncode != 0
