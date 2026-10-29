# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "polars", "pyyaml"]
# ///
"""Frozen-mix builder .

Input: a spec.yaml and source directories of JSONL rows. A source row is a decision row ({id, state, question,
target}) or a GLiClass row ({id, text, all_labels, true_labels, prompt?}); both are kept as they are (the trainer
converts decision rows with the engine mapping). Each source dir may hold SOURCE.json with defaults
{licence_class, teacher, licence, revision}; a row's own fields win.

spec.yaml:
  name: cur-v1
  seed: 0
  total_rows: 300000            # shares are of this total; caps are absolute
  exclude_evals: scratch/evals/v1   # evals/<version> dir: texts.jsonl + inventory.json (never trained on)
  licence_classes: [permissive, sa, publisher]
  sources:
    - {path: sources/prompt-slice/v1, files: [prompt_screened_decision.jsonl], share: 0.35}
    - {path: sources/base-r1g/v1, files: [base.jsonl], cap: 100000, drop_src: c1_drop_sources.txt}
  `files` is REQUIRED per source: slice directories also hold eval_*.jsonl and *_removed.jsonl files (a glob would
  put eval rows, e.g. the Omni rows removed by the eval dedup, into the mix). `share` needs total_rows > 0.

Steps, per source: read rows -> drop_src -> licence class allowed? -> forbidden by an eval rule? -> decontaminate
against every eval text -> within-source exact duplicate (text, instructions, labels) removal -> seeded sample
(share x total_rows, or cap; never upsampled: a short source is reported). Then shuffle all (seed).
Decontamination (the row's input text, rendered as the trainer renders it), in order:
  exact      sha256 of the rendered text equals an eval text's
  norm       sha256 of the normalised text (lowercase, alphanumerics, single spaces) equals an eval's
  prefix     normalised 160-char prefix equals an eval's (the Omni rule; texts >= 100 normalised chars)
  skeleton   the normalised text with digits and LaTeX commands removed equals an eval's (>= 40 chars; catches
             renamed-number twins of maths problems)
  contain    an eval text sits inside the row: contain_turn = a line / dialogue turn of the row equals an eval text
             (normalised, >= 20 chars); contain_short = every 4-gram of a short eval (6-12 words, >= 30 chars) occurs
             in a row at most 3x its length; contain_eval = >= 50 % of one eval's 13-gram shingles (>= 8) occur in the
             row
  shingle    >= 30 % of the row's word 13-gram shingles occur in eval texts (>= 8 of them; rows with 3-7 shingles
             need >= 50 %)
  Shingle and containment hits name the eval set of the best-matching eval text.
Eval rules (from evals/<version>/inventory.json rules, hard-coded here): no row from an 8-K source or with any
distinctive 8-K item name as a label (held-out scheme, binary tasks included); no row from a HelpSteer3 source.
Every check runs on each input view of a row (rendered input, prompt / instructions, prompt + input, string values
of a structured state). Short eval texts (4-12 words) are also matched inside rows by their 4-grams.
Outputs in OUT (never overwritten): spec.yaml (copy), mix.jsonl (trainer input), mix.parquet, manifest.parquet
(id, source, file, file_sha256, text_sha256, licence_class, teacher, row_type), decon_report.json (per source and
reason, per eval set hit counts, 5 examples per reason), build.json (command, hashes, counts).
usage: uv run scripts/mix_build.py spec.yaml OUT
"""

import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import yaml

SHINGLE = 13
SHINGLE_FRAC = 0.30
SHINGLE_MIN = 8
PREFIX = 160


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", (t or "").lower())).strip()


def as_text(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def render_state(state):
    """as gliclass_ub_train.render_state (engine mapping)"""
    if state in (None, "", {}, []):
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "\n".join(f"- {as_text(x)}" for x in state)
    lines = []
    for k, v in state.items():
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            lines += [f"{k}:"] + [f"- {x}" for x in v]
        else:
            lines.append(f"{k}: {as_text(v)}")
    return "\n".join(lines)


def row_view(r):
    """-> (row_type, input text, instructions, label strings)"""
    if "question" in r and "state" in r:
        q = r["question"]
        c = q.get("criteria") or {}
        labels = [as_text(v) if v is not None else k for k, v in c.items()] if isinstance(c, dict) else \
            [as_text(x) for x in c]
        return "decision", render_state(r["state"]), as_text(q.get("instructions") or ""), labels
    return "gliclass", str(r.get("text", "")), str(r.get("prompt") or ""), [str(x) for x in r.get("all_labels", [])]


def input_views(r, rtype, text, instr):
    """every text the model reads for this row: rendered input, prompt / instructions, the prompt + input together,
    and each string value of a structured state (so a copied eval text is found whichever field holds it)"""
    views = [text]
    if instr.strip():
        views += [instr, instr + " " + text]
    st = r.get("state")
    if isinstance(st, dict):
        views += [v for v in st.values() if isinstance(v, str) and v.strip()]
    elif isinstance(st, list):
        views += [v for v in st if isinstance(v, str) and v.strip()]
    return views


def skeleton(t):
    t = re.sub(r"\\[a-zA-Z]+", " ", (t or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", t)).strip()


TURN_RE = re.compile(r"\n|(?:^|(?<=\s))(?:human|assistant|user|system)\s*:", re.IGNORECASE)


def grams(words, n):
    return {hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}


def shingles(t):
    w = norm(t).split()
    if len(w) < SHINGLE:
        return np.zeros(0, dtype=np.int64)
    hs = [hash(" ".join(w[i : i + SHINGLE])) for i in range(len(w) - SHINGLE + 1)]
    return np.unique(np.array(hs, dtype=np.int64))


class EvalIndex:
    def __init__(self, eval_dir):
        d = Path(eval_dir)
        self.inv = json.loads((d / "inventory.json").read_text())
        self.exact, self.norm, self.prefix, self.skel, self.short = {}, {}, {}, {}, {}
        self.short_evals, self.short_index = [], {}
        sh, sh_set = [], []
        self.set_names = []
        with open(d / "texts.jsonl") as f:
            for line in f:
                e = json.loads(line)
                sets = e["sets"]
                self.exact[e["sha256"]] = sets
                n = norm(e["text"])
                self.norm[sha(n)] = sets
                if len(n) >= 100:
                    self.prefix[n[:PREFIX]] = sets
                sk = skeleton(e["text"])
                if len(sk) >= 40:
                    self.skel[sha(sk)] = sets
                if len(n) >= 20:
                    self.short[n] = sets
                w = n.split()
                if 6 <= len(w) < SHINGLE and len(n) >= 30:  # too short for 13-grams: containment by 4-grams
                    g = grams(w, 4)
                    self.short_evals.append((g, sets))
                    for x in g:
                        self.short_index.setdefault(x, []).append(len(self.short_evals) - 1)
                s = shingles(e["text"])
                sh.append(s)
                self.set_names.append(sets[0])
                sh_set.append(np.full(len(s), len(self.set_names) - 1, dtype=np.int32))
        self.n_shingles = np.array([len(x) for x in sh], dtype=np.int64)
        allsh = np.concatenate(sh) if sh else np.zeros(0, dtype=np.int64)
        allset = np.concatenate(sh_set) if sh_set else np.zeros(0, dtype=np.int32)
        order = np.argsort(allsh, kind="stable")
        self.shingles, self.shingle_set = allsh[order], allset[order]  # duplicates kept; first match names a set
        k8 = self.inv["sets"].get("ho_8k_item", {}).get("label_keys", [])
        self.k8 = set(k8)
        print(f"eval index: {len(self.exact)} texts, {len(self.shingles)} shingles, version {self.inv['version']}",
              flush=True)

    def check(self, text):
        """-> (reason, eval sets) or None"""
        h = sha(text)
        if h in self.exact:
            return "exact", self.exact[h]
        n = norm(text)
        if sha(n) in self.norm:
            return "norm", self.norm[sha(n)]
        if len(n) >= 100 and n[:PREFIX] in self.prefix:
            return "prefix", self.prefix[n[:PREFIX]]
        sk = skeleton(text)
        if len(sk) >= 40 and sha(sk) in self.skel:
            return "skeleton", self.skel[sha(sk)]
        for seg in TURN_RE.split(text):
            ns = norm(seg)
            if len(ns) >= 20 and ns in self.short:
                return "contain_turn", self.short[ns]
        w = n.split()
        if self.short_index and 6 <= len(w) <= 3 * (SHINGLE + 2):  # a short eval must be >= 1/3 of the row
            cand = Counter()
            rg = grams(w, 4)
            for x in rg:
                for j in self.short_index.get(x, ()):
                    cand[j] += 1
            for j, c in cand.most_common(3):
                g, sets = self.short_evals[j]
                if c >= len(g) and g <= rg and len(w) <= 3 * (len(g) + 3):
                    return "contain_short", sets
        s = shingles(text)
        if len(s) and len(self.shingles):
            pos = np.searchsorted(self.shingles, s)
            pos[pos >= len(self.shingles)] = 0
            match = self.shingles[pos] == s
            hits = int(match.sum())
            if hits:
                per_text = np.bincount(self.shingle_set[pos[match]])
                top = int(per_text.argmax())
                if (hits >= SHINGLE_MIN and hits / len(s) >= SHINGLE_FRAC) or (
                        3 <= len(s) < SHINGLE_MIN and hits / len(s) >= 0.5):
                    return "shingle", [self.set_names[top]]
                if self.n_shingles[top] >= 8 and per_text[top] / self.n_shingles[top] >= 0.5:
                    return "contain_eval", [self.set_names[top]]
        return None

    def forbidden(self, src, labels):
        s = src.lower()
        if "helpsteer3" in s:
            return "heldout_source_helpsteer3"
        if any(x in s for x in ("sec-8k", "8-k", "8k-")):
            return "heldout_scheme_8k_items"
        hit = self.k8 & set(labels)
        if hit - {"Other Events"} or len(hit) >= 2:  # any distinctive item name, binary tasks included
            return "heldout_scheme_8k_items"
        return None


def main():
    spec_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: mix versions are never overwritten")
    spec = yaml.safe_load(spec_path.read_text())
    rng = random.Random(spec["seed"])
    allowed = set(spec.get("licence_classes", ["permissive"]))
    ev = EvalIndex(spec["exclude_evals"])
    report = {"eval_version": ev.inv["version"], "sources": {}}
    mixed, manifest = [], []
    for s in spec["sources"]:
        sdir = Path(s["path"])
        meta = json.loads((sdir / "SOURCE.json").read_text()) if (sdir / "SOURCE.json").exists() else {}
        drop = set()
        if s.get("drop_src"):
            drop = {x.strip() for x in open(s["drop_src"]) if x.strip()}
        if not s.get("files"):
            sys.exit(f"{sdir}: `files` is required (the directory may hold eval files)")
        if "share" in s and not spec.get("total_rows"):
            sys.exit(f"{sdir}: `share` needs total_rows > 0")
        files = s["files"]
        st, examples, hits = Counter(), defaultdict(list), Counter()
        pool, seen = [], {}
        for fn in files:
            fp = sdir / fn
            fsha = sha_file(fp)
            with open(fp) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    st["read"] += 1
                    src = str(r.get("src", meta.get("src", sdir.parent.name)))
                    lic = r.get("licence_class", s.get("licence_class", meta.get("licence_class")))
                    if src in drop:
                        st["drop_src"] += 1
                        continue
                    if lic not in allowed:
                        st[f"licence_{lic}"] += 1
                        continue
                    rtype, text, instr, labels = row_view(r)
                    why = ev.forbidden(src, labels)
                    if why:
                        st[why] += 1
                        continue
                    hit = None
                    for view in input_views(r, rtype, text, instr):
                        hit = ev.check(view)
                        if hit:
                            break
                    if hit:
                        st[f"decon_{hit[0]}"] += 1
                        for e in hit[1]:
                            hits[e] += 1
                        if len(examples[hit[0]]) < 5:
                            examples[hit[0]].append({"id": r.get("id"), "sets": hit[1][:3], "text": text[:160]})
                        continue
                    key = sha(text + "\x00" + instr + "\x00" + "\x01".join(labels))
                    tgt = json.dumps(r.get("target", r.get("true_labels")), sort_keys=True)
                    if key in seen:
                        st["dup_within_source" if seen[key] == tgt else "dup_conflicting_label_kept_first"] += 1
                        continue
                    seen[key] = tgt
                    r.setdefault("src", src)
                    r["licence_class"] = lic
                    r.setdefault("teacher", s.get("teacher", meta.get("teacher", "unknown")))
                    if r.get("id") is None:
                        r["id"] = f"{sdir.parent.name}-{key[:16]}"
                    pool.append((r, {"id": r["id"], "source": str(sdir), "file": fn, "file_sha256": fsha,
                                     "text_sha256": sha(text), "licence_class": lic, "teacher": r["teacher"],
                                     "revision": str(meta.get("revision", meta.get("job", ""))),
                                     "src": src, "row_type": rtype}))
        want = s["cap"] if "cap" in s else round(s["share"] * spec["total_rows"])
        rng.shuffle(pool)
        take = pool[:want]
        st["eligible"] = len(pool)
        st["taken"] = len(take)
        st["short_by"] = max(0, want - len(pool))
        mixed += [t[0] for t in take]
        manifest += [t[1] for t in take]
        report["sources"][str(sdir)] = {"counts": dict(st), "eval_set_hits": dict(hits), "examples": examples,
                                        "source_meta": meta, "want": want}
        print(sdir, json.dumps(dict(st)), flush=True)
    ids = [m["id"] for m in manifest]
    assert len(ids) == len(set(ids)), "duplicate row ids across sources"
    order = list(range(len(mixed)))
    rng.shuffle(order)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec_path, out / "spec.yaml")
    with open(out / "mix.jsonl", "w") as f:
        f.writelines(json.dumps(mixed[i], ensure_ascii=False) + "\n" for i in order)
    pl.DataFrame([manifest[i] for i in order]).write_parquet(out / "manifest.parquet")
    pl.DataFrame({"id": [mixed[i]["id"] for i in order],
                  "row": [json.dumps(mixed[i], ensure_ascii=False) for i in order]}).write_parquet(out / "mix.parquet")
    (out / "decon_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    build = {
        "command": sys.argv,
        "spec_sha256": sha_file(out / "spec.yaml"),
        "builder_sha256": sha_file(__file__),
        "eval_inventory_sha256": sha_file(Path(spec["exclude_evals"]) / "inventory.json"),
        "rows": len(mixed),
        "by_source": Counter(m["source"] for m in manifest),
        "by_licence_class": Counter(m["licence_class"] for m in manifest),
        "by_teacher": Counter(m["teacher"] for m in manifest).most_common(30),
        "mix_jsonl_sha256": sha_file(out / "mix.jsonl"),
        "manifest_sha256": sha_file(out / "manifest.parquet"),
    }
    (out / "build.json").write_text(json.dumps(build, indent=1))
    print(json.dumps(build, indent=1))


if __name__ == "__main__":
    main()
