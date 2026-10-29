# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///
"""Catalogue decision rows from the Harvard Library catalogue (HOLLIS 2022, CC0; MARC 21), modern English records
with a summary (520) or a contents note (505). Labels are cataloguer-assigned fields that never enter `state`.

Question types (curation-shaped remaps of bibliographic fields):
  hvm_lc_domain   choice   LC Classification main class (first letters of 050/090 call number) -> 20 areas of
                           knowledge with descriptions (E and F merged). Options: gold + 1 neighbouring class
                           uniformly drawn distractors (same flat prior as gold: options carry no label signal;
                           v1 forced a neighbouring class and drew distractors by frequency, which leaked the gold to
                           an options-only baseline at 52 %, review B1); at most --per-class
                           gold rows per class (flat label prior).
  hvm_lit_form    choice   MARC 008/33 literary form, with the hv_lf option set and descriptions (in-scheme with the
                           hv_lf dev set; that set is 1700-1925 books, these are 1950+). not_fiction capped.
  hvm_gov_pub     yes/no   MARC 008/28 government publication: positive = any government level code, negative =
                           blank; unknown / not coded records skipped; balanced.
Held-out schemes (not built here, used as evals): hv_if (item form) and hv_sd (subject domain from LCSH rules).
State = title, author, publisher, year, physical description, summary (520 $a) and contents (505 $a, truncated);
subject headings, genre terms and call numbers are left out (they give the answers away).
Exclusions: MARC ids and normalised titles of every hv_* / bl_* file under data/prepared (all splits).
Instructions: three phrasings per question type, picked at random per row.
usage: uv run scripts/cat_harvard_build.py OUT_DIR --parquet harvard-marc-50.parquet [...] [--per-class 500]
"""

import argparse
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

import polars as pl

LC_CLASSES = {
    "A": "General works: encyclopedias, general periodicals, museums, academies and learned societies, yearbooks.",
    "B": "Philosophy, psychology and religion: philosophical systems, logic, ethics, psychology, religions, theology.",
    "C": "Auxiliary sciences of history: history of civilization, archaeology, genealogy, heraldry, numismatics, "
         "epigraphy, collective biography.",
    "D": "World history: history of Europe, Asia, Africa, Oceania and the world (not the Americas).",
    "EF": "History of the Americas: United States, Canada, Latin America (national and local history).",
    "G": "Geography, anthropology and recreation: atlases, maps, travel, environment, ethnology, sports, folklore.",
    "H": "Social sciences: economics, business, finance, sociology, statistics, social problems, family, labour.",
    "J": "Political science: political theory, government, public administration, international relations.",
    "K": "Law: law of all countries, legal history, constitutions, international law.",
    "L": "Education: theory and practice of teaching, schools, universities, educational history.",
    "M": "Music: musical scores, music history and criticism, instruction in music.",
    "N": "Fine arts: visual arts, architecture, sculpture, drawing, painting, decorative arts.",
    "P": "Language and literature: linguistics, languages, literary history and criticism, literary works.",
    "Q": "Science: mathematics, computer science, astronomy, physics, chemistry, geology, biology.",
    "R": "Medicine: health, public health, clinical medicine, surgery, nursing, pharmacy, dentistry.",
    "S": "Agriculture: farming, plant and animal culture, forestry, fisheries, hunting.",
    "T": "Technology: engineering, building, manufacturing, chemical technology, photography, home economics, crafts.",
    "U": "Military science: armies, tactics, military administration and history of warfare (land forces).",
    "V": "Naval science: navies, navigation, shipbuilding, marine engineering.",
    "Z": "Bibliography and library science: books, printing, publishing, libraries, information resources.",
}
LC_INSTR = ["Which area of knowledge does this work mainly belong to?",
            "What is the main subject area of this publication?",
            "Assign this book to one broad field of knowledge."]

LF_CODES = {"0": "not_fiction", "1": "fiction", "f": "fiction", "p": "poetry", "d": "drama", "i": "letters",
            "j": "short_stories", "e": "essays", "s": "speeches", "h": "humor_satire"}
LF_CRITERIA = {
    "not_fiction": "Not fiction: factual or practical prose such as history, science, religion, law, "
                   "travel, biography, reports, manuals and reference works.",
    "fiction": "Fiction: novels, romances, tales and other invented prose narratives.",
    "poetry": "Poetry: poems, verse, ballads, hymns or songs written in verse.",
    "drama": "Drama: plays, comedies, tragedies, farces and other works written for performance.",
    "letters": "Letters: a collection of correspondence or works written in the form of letters.",
    "short_stories": "Short stories: a collection of short prose fiction pieces.",
    "essays": "Essays: a collection of short prose essays or sketches on various topics.",
    "speeches": "Speeches: orations, addresses, lectures or other works composed to be spoken aloud.",
    "humor_satire": "Humor and satire: comic, humorous or satirical writing.",
}
LF_INSTR = ["What is the literary form of this book?", "What kind of writing is this work?",
            "Which literary form best describes this publication?"]
GOV_POS = set("acfilmo")  # 008/28 government levels; "s" (state) left out: mostly US state university presses (review S1)
GOV_INSTR = ["Was this item published by a government body or an intergovernmental organisation?",
             "Is this a government publication?",
             "Did a government agency (national, state, local or international) issue this publication?"]


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:60]


def subfield(marc, tag, code):
    out = []
    for f in marc:
        if f[0] == tag and len(f) == 3:
            out += [v for c, v in f[2] if c == code]
    return out


def lc_main(call):
    if re.match(r"\s*MLC", call or ""):  # MLCS / MLCM: LC shelf numbers, not classes
        return None
    m = re.match(r"\s*((?:[A-Z]{1,2}|K[A-Z]{1,2}|DJK|DAW))\s?\d", call or "")  # skips "Microfiche", "LAW", "AQ" ...
    if not m:
        return None
    c = m.group(1)[0]
    if c in "EF":
        return "EF"
    return c if c in LC_CLASSES else None


def clean(s, n=None):
    s = re.sub(r"\s+", " ", (s or "")).strip(" /:;,.")
    return s[:n].rsplit(" ", 1)[0] + " ..." if n and len(s) > n else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--exclude-glob", default="data/prepared/[hb][vl]*.jsonl")
    ap.add_argument("--min-year", type=int, default=1950)
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--lf-not-fiction", type=int, default=700)  # flat with the other classes (review S4)
    ap.add_argument("--lf-per-class", type=int, default=700)
    ap.add_argument("--gov-per-side", type=int, default=1000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not 2 <= args.k <= len(LC_CLASSES):
        raise SystemExit(f"--k must be in 2..{len(LC_CLASSES)}")
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    excl_ids, excl_titles = set(), set()
    for f in glob.glob(args.exclude_glob):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                excl_ids.add(str(r.get("id", "")).split("-", 1)[-1])
                st = r.get("state")
                if isinstance(st, dict):
                    excl_titles.add(norm_title(st.get("title")))
    cols = ["id", "f008", "record_type", "bib_level", "language_code", "title", "author", "publisher",
            "physical_description", "lc_class", "marc_json", "year"]
    df = pl.concat([pl.read_parquet(p, columns=cols) for p in args.parquet])
    df = df.filter((pl.col("language_code") == "eng") & (pl.col("record_type") == "a") & (pl.col("bib_level") == "m")
                   & (pl.col("year") >= args.min_year)
                   & (pl.col("marc_json").str.contains('"520"') | pl.col("marc_json").str.contains('"505"')))
    st = Counter(read=df.height)
    recs, seen = [], set()
    for r in df.iter_rows(named=True):
        t = norm_title(r["title"])
        if r["id"] in excl_ids or t in excl_titles:
            st["excluded_eval"] += 1
            continue
        if t in seen:
            st["dup_title"] += 1
            continue
        marc = json.loads(r["marc_json"])
        summary = clean(" ".join(subfield(marc, "520", "a")), 1500)
        contents = clean(" ".join(subfield(marc, "505", "a")), 800)
        state = {"title": clean(r["title"]), "author": clean(r["author"]),
                 "publisher": clean("; ".join(r["publisher"] or [])), "year": str(r["year"]),
                 "physical_description": clean("; ".join(r["physical_description"] or []))}
        if summary:
            state["summary"] = summary
        if contents:
            state["contents"] = contents
        if not (summary or contents):  # 520 / 505 present but no $a text (e.g. enhanced 505 with $t only)
            st["no_note_text"] += 1
            continue
        seen.add(t)  # only eligible records claim a title
        state = {k: v for k, v in state.items() if v}
        f008 = r["f008"] or ""
        recs.append({"id": r["id"], "state": state, "lc": lc_main((r["lc_class"] or [None])[0]),
                     "lf": LF_CODES.get(f008[33:34]) if len(f008) > 33 else None,
                     "gov": f008[28:29] if len(f008) > 28 else None})
    st["eligible"] = len(recs)
    rng.shuffle(recs)
    rows = []

    def row(rec, src, qtype, instr, criteria, gold):
        return {"id": f"{src}-{rec['id']}", "src": src, "teacher": "Harvard Library cataloguers (MARC)",
                "licence_class": "permissive", "state": rec["state"],
                "question": {"type": qtype, "instructions": rng.choice(instr), "criteria": criteria},
                "target": {gold: 1.0}}

    # LC domain: flat prior, neighbour + frequency-weighted distractors
    per = Counter()
    for rec in recs:
        g = rec["lc"]
        if not g or per[g] >= args.per_class:
            continue
        per[g] += 1
        opts = {g} | set(rng.sample([c for c in LC_CLASSES if c != g], args.k - 1))
        order = sorted(opts)
        rng.shuffle(order)
        rows.append(row(rec, "hvm_lc_domain", "choice", LC_INSTR, {c: LC_CLASSES[c] for c in order}, g))
    st["lc_domain"] = dict(per)

    # literary form: hv_lf option set, capped
    per = Counter()
    for rec in recs:
        g = rec["lf"]
        cap = args.lf_not_fiction if g == "not_fiction" else args.lf_per_class
        if not g or per[g] >= cap:
            continue
        per[g] += 1
        rows.append(row(rec, "hvm_lit_form", "choice", LF_INSTR, LF_CRITERIA, g))
    st["lit_form"] = dict(per)

    # government publication: balanced yes/no
    per = Counter()
    for rec in recs:
        g = "true" if rec["gov"] in GOV_POS else "false" if rec["gov"] == " " else None
        if not g or per[g] >= args.gov_per_side:
            continue
        per[g] += 1
        rows.append(row(rec, "hvm_gov_pub", "noul", GOV_INSTR, {"true": "yes", "false": "no"}, g))
    st["gov_pub"] = dict(per)

    rng.shuffle(rows)
    with open(args.out / "harvard_modern_decision.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st["rows"] = len(rows)
    (args.out / "build_stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
