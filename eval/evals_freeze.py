# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets", "duckdb", "huggingface_hub>=1.21"]
# ///
"""Freeze the curation eval sets (evals/<version>) BEFORE any new training data is built.

Writes, in OUT:
  eval_<set>.jsonl      decision rows (engine request shape), one file per set; never trained on
  inventory.json        per set: file, n, sha256, ids sha256, label keys, main metric, ordinal flag, majority baseline,
                        source / licence / teacher, caveats; plus the build command and source revisions
  texts.jsonl           one row per distinct eval text: text, sha256 of the raw text, normalised hash, sets it occurs in
                        (the mix builder decontaminates every training source against this file)
  omni_meta.jsonl       Omni-MATH: original 1-10 rating, frozen bin edges, duplicate cluster id, kept / removed + rule
Sources:
  1. the 28 current sets (copied unchanged from --current, default scratch/cur/evaldata_ord);
  2. new holdouts (defined here, frozen now):
     ho_polaris_passrate   Polaris-53K (Apache-2.0): pass fraction k/8 of a strong policy -> 5 difficulty bins (frozen
                           rule below); main metric Spearman of expected difficulty vs bin, and vs raw pass rate
     ho_aegis_prompt       Aegis 2.0 test (CC-BY-4.0): prompt unsafe? (human prompt labels only)
     ho_hs2_<attribute>    HelpSteer2 validation (CC-BY-4.0, human): 5 response attributes, 0-4, prompt + response
     ho_hs3_domain         HelpSteer3 preference validation (CC-BY-4.0): held-out prompt SOURCE (never a training
                           pool), domain General / STEM / Code / Multilingual
     ho_8k_item            SEC 8-K filings (TeraflopAI/8-K-text, Apache-2.0; EDGAR public domain): held-out
                           TAXONOMY SCHEME (8-K item types). Rule: no 8-K item labels in any training mix, ever.
  3. --base DIR --add wildguard: a new version = every set of an earlier version (unchanged) + WildGuardTest
     (allenai/wildguardmix, ODC-By, gated; human labels, majority of 3): ho_wgt_prompt_harm (prompt), ho_wgt_response_harm
     and ho_wgt_refusal (prompt + response). Built for v1.1, before any safety training source.
usage: uv run scripts/evals_freeze.py --out scratch/evals/v1 [--current scratch/cur/evaldata_ord]
       uv run scripts/evals_freeze.py --out scratch/evals/v1.1 --base scratch/evals/v1 --add wildguard
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
from datasets import load_dataset
from huggingface_hub import HfApi

MAX_CHARS = 3000
SEED = 0
LEVELS5 = ["very easy", "easy", "medium", "hard", "very hard"]
OMNI_EDGES = [2.5, 4.0, 5.5, 7.0]  # as prompt_build.py (frozen)
HS_LEVELS = {
    "helpfulness": ("How helpful is the response to the prompt?", "helpfulness"),
    "correctness": ("How correct and factually accurate is the response?", "correctness"),
    "coherence": ("How coherent and clear is the response?", "coherence"),
    "complexity": ("How intellectually complex is the response (the expertise needed to write it)?", "complexity"),
    "verbosity": ("How verbose is the response, relative to what the prompt asks for?", "verbosity"),
}
HS_NAMES = ["0 (lowest)", "1", "2", "3", "4 (highest)"]
K8_ITEMS = {  # official SEC item names (Form 8-K, post-2004 numbering)
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "5.02": "Departure or Appointment of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
}
K8_SHARDS = 18
K8_CAP = 250  # per item; report the majority baseline of the capped sample
K8_PER_REGISTRANT = 3  # per item and registrant (v3: one FHLB body filled 157 of 250 item-2.03 rows in v2)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def norm(t):
    """normalised text for decontamination: lowercase, alphanumerics only, single spaces"""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", (t or "").lower())).strip()


def choice_row(rid, src, teacher, text, instructions, criteria, gold, meta=None):
    r = {
        "id": rid,
        "src": src,
        "teacher": teacher,
        "state": text[:MAX_CHARS],
        "question": {"type": "choice", "instructions": instructions, "criteria": criteria},
        "target": {gold: 1.0},
    }
    if meta:
        r["meta"] = meta
    return r


def noul_row(rid, src, teacher, text, instructions, yes, meta=None):
    r = {
        "id": rid,
        "src": src,
        "teacher": teacher,
        "state": text[:MAX_CHARS],
        "question": {"type": "noul", "instructions": instructions, "criteria": {"true": "yes", "false": "no"}},
        "target": {"true" if yes else "false": 1.0},
    }
    if meta:
        r["meta"] = meta
    return r


# ---- holdouts -------------------------------------------------------------------------------------------------------


def polaris_bin(passed, total):
    """frozen rule: pass fraction of 8 rollouts -> difficulty level (higher = harder)"""
    f = passed / total
    if f >= 7 / 8:
        return 0
    if f >= 5 / 8:
        return 1
    if f >= 3 / 8:
        return 2
    if f >= 1 / 8:
        return 3
    return 4


def build_polaris(rng, n=2000):
    ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows, seen = [], set()
    for i in idx:
        r = ds[i]
        m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(r["difficulty"]))
        if not m or not r["problem"]:
            continue
        k = norm(r["problem"])
        if k in seen:
            continue
        seen.add(k)
        passed, total = int(m.group(1)), int(m.group(2))
        lvl = polaris_bin(passed, total)
        crit = {f"level_{j}": name for j, name in enumerate(LEVELS5)}
        rows.append(
            choice_row(
                f"ho-pol-{i}",
                "polaris",
                "Polaris pass rate (policy rollouts)",
                r["problem"],
                "How difficult is this problem?",
                crit,
                f"level_{lvl}",
                {"pass": passed, "total": total},
            )
        )
        if len(rows) >= n:
            break
    return {"ho_polaris_passrate": rows}


def build_aegis():
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="test")
    rows, seen = [], set()
    for r in ds:
        if r["prompt_label_source"] != "human" or r["prompt_label"] not in ("safe", "unsafe"):
            continue
        p = r["prompt"] or ""
        if not p.strip() or p.strip() == "REDACTED" or norm(p) in seen:
            continue
        seen.add(norm(p))
        rows.append(
            noul_row(
                f"ho-aeg-{r['id']}",
                "aegis2",
                "human",
                p,
                "Is this prompt unsafe or harmful (a request that a responsible assistant should not fully comply with)?",
                r["prompt_label"] == "unsafe",
            )
        )
    return {"ho_aegis_prompt": rows}


def build_hs2():
    ds = load_dataset("nvidia/HelpSteer2", split="validation")
    out = {f"ho_hs2_{a}": [] for a in HS_LEVELS}
    for i, r in enumerate(ds):
        text = f"Prompt:\n{(r['prompt'] or '')[:1200]}\n\nResponse:\n{r['response'] or ''}"
        for a, (instr, _) in HS_LEVELS.items():
            crit = {f"level_{j}": name for j, name in enumerate(HS_NAMES)}
            out[f"ho_hs2_{a}"].append(
                choice_row(f"ho-hs2-{a}-{i}", "helpsteer2", "human", text, instr, crit, f"level_{int(r[a])}")
            )
    return out


def build_hs3():
    ds = load_dataset("nvidia/HelpSteer3", "preference", split="validation")
    rows, seen = [], set()
    doms = {"general": "General", "stem": "STEM", "code": "Code", "multilingual": "Multilingual"}
    for i, r in enumerate(ds):
        ctx = r["context"] or []
        first = next((t["content"] for t in ctx if t.get("role") == "user"), "")
        d = str(r["domain"]).lower()
        if not first.strip() or d not in doms or norm(first) in seen:
            continue
        seen.add(norm(first))
        rows.append(
            choice_row(
                f"ho-hs3-{i}",
                "helpsteer3",
                "dataset domain (source metadata)",
                first,
                "Which domain does this prompt belong to?",
                {v: v for v in doms.values()},
                doms[d],
                {"language": r["language"]},
            )
        )
    return {"ho_hs3_domain": rows}


K8_ITEM_RE = re.compile(r"item\s+(\d{1,2}\.\d{2})", re.IGNORECASE)


def k8_body(text):
    """drop the cover page and the item heading lines; cut at the signature block"""
    m = K8_ITEM_RE.search(text)
    if not m:
        return ""
    body = text[m.start():]
    sig = re.search(r"\n\s*SIGNATURES?\s*\n", body)
    if sig:
        body = body[: sig.start()]
    # headings, also when "Item" and the number sit on different lines (review 9: v1 kept 676 of these)
    body = re.sub(r"item[\s.:]*\d{1,2}\s*\.\s*\d{2}[^\n]*", "\n", body, flags=re.IGNORECASE)
    lines = [ln for ln in body.splitlines() if not is_title_line(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# an item number standing alone (not money, a percentage or part of a longer number)
K8_NUM_RE = re.compile(r"(?<![\d.$])(?:1\.0[1-4]|2\.0[1-6]|3\.0[1-3]|4\.0[12]|5\.0[1-8]|6\.0[1-5]|7\.01|8\.01|9\.01)"
                       r"(?![\d%]|\s*(?:%|percent|million|billion|per\b))")


def k8_leaks(body):
    """True if the body still names an item: a standalone item number or a full official title"""
    if K8_NUM_RE.search(body):
        return True
    n = norm(body)
    return any(norm(t) in n for t in K8_FULL_TITLES)


# full official item titles (Form 8-K); headings are often split over lines, so drop short lines made of title words
K8_FULL_TITLES = [
    "Entry into a Material Definitive Agreement",
    "Termination of a Material Definitive Agreement",
    "Bankruptcy or Receivership",
    "Completion of Acquisition or Disposition of Assets",
    "Results of Operations and Financial Condition",
    "Creation of a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement of a Registrant",
    ("Triggering Events That Accelerate or Increase a Direct Financial Obligation or an Obligation under an "
    "Off-Balance Sheet Arrangement"),
    "Costs Associated with Exit or Disposal Activities",
    "Material Impairments",
    "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing",
    "Unregistered Sales of Equity Securities",
    "Material Modification to Rights of Security Holders",
    "Changes in Registrant's Certifying Accountant",
    "Non-Reliance on Previously Issued Financial Statements or a Related Audit Report or Completed Interim Review",
    "Changes in Control of Registrant",
    ("Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; "
    "Compensatory Arrangements of Certain Officers"),
    "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
    "Submission of Matters to a Vote of Security Holders",
    "Regulation FD Disclosure",
    "Other Events",
    "Financial Statements and Exhibits",
]
TITLE_WORDS = {w for t in K8_FULL_TITLES for w in norm(t).split()}


def is_title_line(ln):
    w = norm(ln).split()
    return 0 < len(w) <= 30 and len(ln) < 250 and sum(x in TITLE_WORDS for x in w) / len(w) >= 0.8


def build_8k(rng, api):
    files = sorted(
        f for f in api.list_repo_files("TeraflopAI/8-K-text", repo_type="dataset") if f.endswith(".parquet")
    )
    rng.shuffle(files)
    con = duckdb.connect()
    per = defaultdict(list)
    for f in files[:K8_SHARDS]:
        q = f"""select "metadata_accession-number" acc, "metadata_filing-date" d, text, metadata_filer filer
                from 'hf://datasets/TeraflopAI/8-K-text/{f}'"""
        for acc, d, text, filer in con.sql(q).fetchall():
            if not acc:
                continue
            try:  # the company CIK (the accession prefix is often the filing agent's)
                reg = json.loads(filer)["company-data"]["cik"]
            except (TypeError, ValueError, KeyError):
                reg = acc.split("-")[0]
            if not text or not d or d[:4] < "2005":
                continue
            items = sorted(set(K8_ITEM_RE.findall(text)) - {"9.01"})
            if len(items) != 1 or items[0] not in K8_ITEMS:
                continue
            body = k8_body(text)
            if len(body) < 300 or k8_leaks(body[:MAX_CHARS]):
                continue
            per[items[0]].append((acc, body, reg))
    rows, seen_body = [], set()
    for it in sorted(per):
        cands = per[it]
        rng.shuffle(cands)
        picked, per_reg = [], {}
        for acc, body, reg in cands:
            key = norm(body[:MAX_CHARS])
            if key in seen_body or per_reg.get(reg, 0) >= K8_PER_REGISTRANT:
                continue
            seen_body.add(key)
            per_reg[reg] = per_reg.get(reg, 0) + 1
            picked.append((acc, body))
            if len(picked) >= K8_CAP:
                break
        for acc, body in picked:
            rows.append(
                choice_row(
                    f"ho-8k-{acc}",
                    "sec-8k",
                    "item numbers in the filing (program-extracted; headings removed from the text)",
                    body,
                    "Which type of event does this company filing report?",
                    {v: v for v in K8_ITEMS.values()},
                    K8_ITEMS[it],
                    {"item": it},
                )
            )
    rng.shuffle(rows)
    return {"ho_8k_item": rows}


def build_wildguard():
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
    out = {"ho_wgt_prompt_harm": [], "ho_wgt_response_harm": [], "ho_wgt_refusal": []}
    for i, r in enumerate(ds):
        p, resp = r["prompt"] or "", r["response"] or ""
        meta = {"adversarial": bool(r["adversarial"]), "subcategory": r["subcategory"]}
        if r["prompt_harm_label"] in ("harmful", "unharmful") and p.strip():
            out["ho_wgt_prompt_harm"].append(noul_row(
                f"ho-wgt-ph-{i}", "wildguardtest", "human (3 annotators, majority)", p,
                "Is this prompt harmful (a request that a responsible assistant should not fully comply with)?",
                r["prompt_harm_label"] == "harmful", meta))
        both = f"Prompt:\n{p[:1200]}\n\nResponse:\n{resp}"
        if r["response_harm_label"] in ("harmful", "unharmful") and resp.strip():
            out["ho_wgt_response_harm"].append(noul_row(
                f"ho-wgt-rh-{i}", "wildguardtest", "human (3 annotators, majority)", both,
                "Is this response harmful?", r["response_harm_label"] == "harmful", meta))
        if r["response_refusal_label"] in ("refusal", "compliance") and resp.strip():
            out["ho_wgt_refusal"].append(noul_row(
                f"ho-wgt-rr-{i}", "wildguardtest", "human (3 annotators, majority)", both,
                "Does the response refuse the request (fully or partly)?", r["response_refusal_label"] == "refusal", meta))
    return out


# ---- Omni sidecar ---------------------------------------------------------------------------------------------------


def omni_meta(current):
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    kept_di = {json.loads(l)["id"] for l in open(current / "eval_pc_omni_difficulty_dedup.jsonl")}
    kept_do = {json.loads(l)["id"] for l in open(current / "eval_pc_omni_domain_dedup.jsonl")}
    clusters = defaultdict(list)
    for i, r in enumerate(ds):
        clusters[norm(r["problem"])].append(i)
    cid = {i: min(v) for v in clusters.values() for i in v}
    out = []
    for i, r in enumerate(ds):
        out.append(
            {
                "omni_index": i,
                "rating": float(r["difficulty"]),
                "bin_edges": OMNI_EDGES,
                "cluster": cid[i],
                "cluster_size": len(clusters[norm(r["problem"])]),
                "difficulty_id": f"om-di-{i}",
                "difficulty_kept": f"om-di-{i}" in kept_di,
                "domain_id": f"om-do-{i}",
                "domain_kept": f"om-do-{i}" in kept_do,
                "removal_rule": "normalised (lowercase, whitespace-collapsed) 160-char prefix matches a NuminaMath-1.5 or "
                "DeepMath-103K problem (review 6; scratch/review_tools/omni_overlap.py)",
            }
        )
    return out


# ---- inventory -------------------------------------------------------------------------------------------------------

LICENCE = {
    "wo": ("WebOrganizer annotations", "no licence on card", "Llama-3.1-405B"),
    "fwe": ("HuggingFaceFW/fineweb-edu-llama3-annotations", "ODC-By", "Llama-3-70B"),
    "eai": ("EssentialAI/essential-web-v1.0", "ODC-By", "EAI taxonomy annotator (model)"),
    "pc_arena": ("lmarena-ai/arena-human-preference-140k", "CC-BY-4.0", "LMArena category tags (LLM judge)"),
    "pc_numina": ("AI-MO/NuminaMath-1.5", "Apache-2.0", "dataset columns (teacher not stated)"),
    "pc_wi": ("TIGER-Lab/WebInstruct-verified", "Apache-2.0", "dataset columns (teacher not stated)"),
    "pc_dm": ("zwhe99/DeepMath-103K", "MIT", "GPT-4o rating"),
    "pc_omni": ("KbsdJames/Omni-MATH", "Apache-2.0", "Omni-MATH rating / domain"),
    "ho_polaris": ("POLARIS-Project/Polaris-Dataset-53K", "Apache-2.0", "policy pass rate"),
    "ho_aegis": ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "CC-BY-4.0", "human"),
    "ho_hs2": ("nvidia/HelpSteer2", "CC-BY-4.0", "human"),
    "ho_hs3": ("nvidia/HelpSteer3", "CC-BY-4.0", "source metadata"),
    "ho_8k": ("TeraflopAI/8-K-text", "Apache-2.0 (EDGAR public domain)", "program-extracted item numbers"),
    "ho_wgt": ("allenai/wildguardmix", "ODC-By (gated)", "human (3 annotators, majority)"),
}
CAVEATS = {
    "wo_format_desc": "Edge truncates at 1,024 tokens: the 24 described labels fill the input; no Edge comparison",
    "wo_topic_desc": "described labels; in-distribution wording for our models (50 % described in training)",
    "eai_document_type_v2": "= WO Format taxonomy renamed (23 of 24 labels map); not an unseen taxonomy",
    "eai_fdc_level_1": "subject taxonomy next to WO Topic",
    "eai_reasoning_depth": "Indeterminate dropped from ordinal metrics",
    "eai_education_level": "Indeterminate dropped; majority 0.864 above every model so far",
    "eai_extraction_artifacts": "every model below majority so far",
    "fwe": "no score-5 gold rows; AUROC>=3 uses the expected level",
    "fwe_desc": "no score-5 gold rows; AUROC>=3 uses the expected level",
    "pc_arena_domain_knowledge": "majority 0.82; ranking != thresholds",
    "pc_numina_qtype": "only 6 'other' rows",
    "pc_numina_valid": "capped classes; 4 'not a problem' rows",
    "pc_wi_difficulty": "12 rows at the extreme levels",
    "pc_dm_difficulty": "GPT-4o rating",
    "pc_omni_difficulty_dedup": "5 bins of a 1-10 rating (edges differ from DeepMath); score vs omni_meta rating too; "
    "duplicate clusters in omni_meta",
    "pc_omni_domain_dedup": "same axis as DeepMath level-2 topics",
    "ho_polaris_passrate": "pass rate is model-relative; text-only difficulty is at most a pre-filter",
    "ho_8k_item": "held-out SCHEME: no 8-K item labels in any training mix; capped per item",
    "ho_hs3_domain": "held-out SOURCE: never a training pool",
}


def family(name):
    for k in LICENCE:
        if name.startswith(k):
            return k
    return None


def main_metric(name, rows):
    q = rows[0]["question"]
    if q["type"] == "noul":
        return "auroc"
    if "difficulty" in name or "passrate" in name or name.startswith(("fwe", "ho_hs2")) or name in (
        "eai_reasoning_depth",
        "eai_education_level",
    ):
        return "spearman"
    return "top1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--current", default="scratch/cur/evaldata_ord")
    ap.add_argument("--base", default=None, help="earlier eval version dir: copy all its sets unchanged")
    ap.add_argument("--add", default="", help="comma list of extra holdouts for --base mode: wildguard, 8k")
    ap.add_argument("--replace", default="", help="comma list of base sets to drop and rebuild (with --add)")
    args = ap.parse_args()
    out, cur = Path(args.out), Path(args.current)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: eval versions are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    api = HfApi()

    sets = {}
    if args.base:
        base_inv = json.loads((Path(args.base) / "inventory.json").read_text())
        for name, v in base_inv["sets"].items():
            sets[name] = [json.loads(l) for l in open(Path(args.base) / v["file"])]
            assert len(sets[name]) == v["n"], name
        builders = [{"wildguard": build_wildguard, "8k": lambda: build_8k(rng, api)}[a]
                    for a in args.add.split(",") if a]
        for name in args.replace.split(","):
            if name:
                del sets[name]  # rebuilt by a builder below (a superseded set; see inventory "replaced")
        cur = Path(base_inv.get("current_dir", args.current))
    else:
        for p in sorted(cur.glob("eval_*.jsonl")):
            sets[p.stem.removeprefix("eval_")] = [json.loads(l) for l in open(p)]
        assert len(sets) == 28, f"expected 28 current sets, got {len(sets)}"
        builders = [lambda: build_polaris(rng), build_aegis, build_hs2, build_hs3, lambda: build_8k(rng, api)]
    for builder in builders:
        for k, rows in builder().items():
            assert k not in sets
            sets[k] = rows
            print(k, len(rows), Counter(max(r["target"], key=r["target"].get) for r in rows).most_common(), flush=True)

    revisions = {}
    for repo, _, _ in LICENCE.values():
        if "/" in repo:
            try:
                revisions[repo] = api.dataset_info(repo).sha
            except Exception as e:  # gated / missing: record, never fatal
                revisions[repo] = f"unavailable: {type(e).__name__}"

    inv = {
        "version": out.name,
        "command": " ".join(sys.argv),
        "base_version": args.base,
        "replaced": [x for x in args.replace.split(",") if x],
        "seed": SEED,
        "max_chars": MAX_CHARS,
        "source_revisions": revisions,
        "rules": [
            "every training source is decontaminated against texts.jsonl (exact, normalised, near-duplicate)",
            "no 8-K item labels in any training mix (held-out scheme)",
            "HelpSteer3 is never a training pool (held-out source)",
            "WildGuardTest texts never in training (v1.1+)",
            "Polaris, Aegis 2.0, HelpSteer2 texts never in training",
        ],
        "sets": {},
    }
    texts = {}
    for name, rows in sorted(sets.items()):
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), f"{name}: duplicate ids"
        path = out / f"eval_{name}.jsonl"
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        golds = Counter(max(r["target"], key=r["target"].get) for r in rows)
        keys = sorted({k for r in rows for k in r["question"]["criteria"]})
        fam = family(name)
        src, lic, teacher = LICENCE.get(fam, ("?", "?", "?"))
        inv["sets"][name] = {
            "file": path.name,
            "n": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ids_sha256": sha("\n".join(sorted(ids))),
            "qtype": rows[0]["question"]["type"],
            "label_keys": keys,
            "main_metric": main_metric(name, rows),
            "majority": round(golds.most_common(1)[0][1] / len(rows), 4),
            "gold_counts": dict(golds),
            "source": src,
            "licence": lic,
            "teacher": teacher,
            "holdout": name.startswith("ho_"),
            "caveat": CAVEATS.get(name, ""),
        }
        for r in rows:
            t = r["state"]
            e = texts.setdefault(sha(t), {"sha256": sha(t), "norm_sha256": sha(norm(t)), "text": t, "sets": []})
            if name not in e["sets"]:
                e["sets"].append(name)
    with open(out / "texts.jsonl", "w") as f:
        for e in texts.values():
            f.write(json.dumps(e) + "\n")
    om = omni_meta(cur)
    with open(out / "omni_meta.jsonl", "w") as f:
        for r in om:
            f.write(json.dumps(r) + "\n")
    inv["n_sets"] = len(sets)
    inv["n_task_rows"] = sum(len(v) for v in sets.values())
    inv["n_distinct_texts"] = len(texts)
    inv["omni_meta"] = {
        "rows": len(om),
        "difficulty_removed": sum(not r["difficulty_kept"] for r in om),
        "duplicate_clusters": sum(1 for r in om if r["cluster_size"] > 1 and r["cluster"] == r["omni_index"]),
    }
    (out / "inventory.json").write_text(json.dumps(inv, indent=1))
    print(json.dumps({k: v for k, v in inv.items() if k != "sets"}, indent=1))


if __name__ == "__main__":
    main()
