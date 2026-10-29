# Curation decision recipe

How the training data, the model and the evaluation of
[davanstrien/ultrabert-380m-curation](https://huggingface.co/davanstrien/ultrabert-380m-curation) were made. It is
a 380M-parameter decision model for data curation: give it a text and named, typed questions (topic, format,
educational value, difficulty, …) and it returns a probability for every option.

This repo is mostly about the **data**. Small decision models live or die by their training data, and few releases
say how that data was built. This one shows every source, labeller, licence check, audit and screening step that went
into the [training mix](https://huggingface.co/datasets/davanstrien/curation-decision-mix), and the scripts that did
it. If you train decision models, please publish your data recipe too!

Everything ran on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), with sources, mixes,
evals and checkpoints in Hugging Face storage buckets mounted into each job.

## Use the model on your data

The quickest path is one command on Jobs ([`inference/curate.py`](inference/curate.py)):

```bash
hf jobs uv run --flavor l4x1 --timeout 1h -s HF_TOKEN inference/curate.py -- \
    --dataset HuggingFaceFW/fineweb-edu --config sample-10BT --text-column text \
    --limit 10000 --out-repo YOUR_NAME/fineweb-edu-curated
```

For millions of pages, [`inference/examples/datatrove_curation.py`](inference/examples/datatrove_curation.py) splits
the work over several GPU Jobs with [datatrove](https://github.com/huggingface/datatrove): 1,020,000 FineWeb pages in
about 19 minutes for about $5 ([result](https://huggingface.co/datasets/davanstrien/fineweb-curated-1m)). The model
card covers the System One API (`jev_api.py`) and the packed batch scorer (`fast_inference.py`).

## The data recipe

The mix has 535,730 rows from nine source groups; 516,964 are released (18,766 base rows without a verified open
licence are left out). The [dataset card](https://huggingface.co/datasets/davanstrien/curation-decision-mix) explains
each choice and the measured effect; this table maps each step to its script.

| Step | What it does | Script(s) |
|---|---|---|
| 1. Freeze the evals first | Build the curation test sets from public sources and freeze them before any training data exists. Every later mix is screened against them. | `eval/evals_freeze.py` |
| 2. Public base | Convert tasksource classification tasks and Knowledgator's GLiClass data to one row format; drop tasks that overlap a benchmark by name, canary or text. | `data/prep_tasksource.py`, `data/prep_stage1.py`, `data/prep_stage2.py`, `data/canary_fix_r1g.py` |
| 3. Audit sources | Text-blind baselines per source (majority label, longest option, options-only ranker) to catch shortcut leaks. | `data/c1_source_audit.py` |
| 4. Base sampling | Register the cleaned, audited base as a frozen source with its sampling rule. | `data/cur_base_source.py` |
| 5. Curation slices | Web and prompt rows that reuse public curation labels (WebOrganizer topic and format, FineWeb-Edu, NuminaMath-1.5, DeepMath-103K, WebInstruct-verified, LMArena tags). | `data/cur_build.py`, `data/prompt_build.py`, `data/panel_targets.py` (row selection: see note), `data/sg_screen.py` |
| 6. Human and program labels | Library catalogue records (National Library of Scotland, Harvard), arXiv author categories, program-computed reasoning and code tasks. | `data/cat_nls_build.py`, `data/cat_harvard_build.py`, `data/fh_arxiv_build.py`, `data/g_procpile.py`, `data/tool_build.py` |
| 7. DecisionBank | Licence-filtered DecisionBank rows; held-out test families dropped. | `data/db_build.py` |
| 8. Freeze the mix | Combine the sources by the spec, decontaminate against every eval text, and write a manifest with hashes. | `data/mix_build.py`, `mixes/spec_v1.yaml`, `tests/test_mix_build.py` |
| 9. Release | Licence filter per row, remove web text, write one config per source. | `data/release_build.py` |

**Note on row selection.** The web and prompt rows in the mix are the rows that an earlier labelling experiment
(a panel of open models) also covered; the release keeps the **original** labels on those rows, not the panel's.
Rerunning that panel costs money, so the dataset repo includes the frozen list of all 535,730 training row ids
(`manifest/training_mix_manifest.parquet`) instead.

## Training

| Step | What it does | Script |
|---|---|---|
| Train | GLiClass uni-encoder head on UltraBERT-380M; 12,000 steps; seeds 0, 1, 2 | `train/gliclass_ub_train.py`, launched by `train/cur_train.sh` |
| Average | Average the three **encoders** only (seed 0 first, so its head is kept) | `train/soup.py --average-prefix model.encoder_model OUT RUN0 RUN1 RUN2` |
| Head tune | 500 steps from the averaged model with the encoder frozen | trainer with `--base-model OUT --encoder-lr 0 --others-lr 4e-5 --max-steps 500 --seed 0` |

Settings for the three seed runs: `--single-label-loss softmax_ce --encoder-lr 2e-5 --others-lr 4e-5 --max-steps 12000`,
token budget 32,768 per batch, training length 4,096, evaluation length 2,048. One run takes about an hour on one
RTX PRO 6000 (peak GPU memory about 55 GB). `image/Dockerfile` is the training image (it also contains evaluation
dependencies).

The release runs used an earlier revision of `gliclass_ub_train.py` (sha256 `a041428e…`). The file here adds two
opt-in options, `--save-every` (periodic checkpoints) and `--dev-holdout` (a dev split); with the defaults, training is
unchanged (checked at the time: identical training losses). GPU training is not bit-for-bit deterministic.

## Evaluation

`eval/cur_score.py` scores prediction files per test set; `eval/cur_gate.py` computes the curation score (16 units:
a set asked with label descriptions is averaged with its plain twin, and the ten LMArena questions count as one unit)
with bootstrap intervals, split into same-labeller and independent gold. `eval/cur_edge_eval.py` and
`eval/d1_eval.py` run the comparison models on the same inputs.

## How reproducible is this?

| Part | Status |
|---|---|
| Using the released model (`curate.py`, datatrove example, API, fast path) | Runs as published |
| Released data (516,964 rows) | Download from the Hub; `web` rows need their text rebuilt from the source datasets (snippet in the dataset card) |
| Data builders (`data/`) | The code that ran, as it ran. Paths assume a bucket mounted at `/b` or `/c` and outputs under it; adapt the mount and the bucket names (`YOUR_NAME/...`). Not packaged as a one-command pipeline. |
| Exact training mix | Not fully public: 18,766 base rows lack a verified open licence and web text is not redistributed. The manifest lists every row id and text hash. |
| Training | Runs from the released data plus rebuilt web text, but the released data is not a direct trainer input yet (a converter from the release format to the trainer's JSONL is not included) |
| Evaluation | Builders included; the eval rows themselves are not redistributed. The Decision Index suite is not included. |

## What is not here

- Evaluation rows, predictions and the Decision Index suite (some contexts cannot be redistributed).
- Experiments that did not make the release (other training arms, labelling panels, synthetic data, other base
  models). Their results are summarised in the dataset card where they informed a choice.
- Rebuilt web page text.

## Licence and credits

Code: Apache-2.0 (see `LICENSE`). Data and labels keep the licences of their sources (per row in the dataset).

Built on [UltraBERT](https://huggingface.co/dmis-lab/UltraBERT-380M) (DMIS Lab),
[GLiClass](https://github.com/Knowledgator/GLiClass) (Knowledgator),
[DecisionBank](https://huggingface.co/datasets/jgeuter/DecisionBank),
[tasksource](https://huggingface.co/tasksource), the
[Decision Index](https://github.com/apolinario/decision-index), [datatrove](https://github.com/huggingface/datatrove),
and the open datasets and labels listed in the dataset card.

```bibtex
@misc{vanstrien2026curationrecipe,
  author = {van Strien, Daniel},
  title = {Curation decision recipe: data, training and evaluation for a small curation decision model},
  year = {2026},
  url = {https://github.com/davanstrien/curation-decision-recipe}
}
```
