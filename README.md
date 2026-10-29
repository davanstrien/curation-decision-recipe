# Curation decision recipe

How the training data, the model and the evaluation of
[davanstrien/ultrabert-380m-curation](https://huggingface.co/davanstrien/ultrabert-380m-curation) were made. It is
a 380M-parameter decision model for data curation: give it a text and named, typed questions (topic, format,
educational value, difficulty, …) and it returns a probability for every option.

This repo is mostly about the **data**. Small decision models live or die by their training data, and few releases
say how that data was built. This one shows every source, labeller, licence check, audit and screening step that went
into the [training mix](https://huggingface.co/datasets/davanstrien/curation-decision-mix), and the scripts that did
it. If you train decision models, please publish your data recipe too!

Everything ran on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs): data building,
labelling experiments, training, evaluation and inference. Sources, mixes, evals and checkpoints lived in Hugging Face
storage buckets mounted into each job, and a laptop only launched the work. Agents: start with [`AGENTS.md`](AGENTS.md).

## Run it on Hugging Face Jobs

Three commands, from one page of text to a trained model. You need a Hugging Face account with Jobs enabled (a paid
plan or credits) and the `hf` CLI logged in; replace `YOUR_NAME` with your Hub username.

### 1. Curate a dataset in one command

[`inference/curate.py`](inference/curate.py) reads a text column from any Hub dataset, asks four curation questions
about every row, and pushes a dataset with the answers (`scored`) and an example filter (`subset`):

```bash
hf jobs uv run inference/curate.py -- --dataset HuggingFaceFW/fineweb-edu --config sample-10BT --text-column text \
    --limit 10000 --out-repo YOUR_NAME/fineweb-edu-curated
```

The script's PEP 723 header carries the Job settings (`[tool.hf-jobs]`: `a10g-small` GPU, 1-hour timeout, your `HF_TOKEN`
to push the result), so the command needs only the script's own arguments. Flags such as `--flavor rtx-pro-6000`
override the header.

About 20 rows/s on `l4x1` after a minute or two of start-up; use `rtx-pro-6000` (about 150 rows/s) for 100,000
rows or more.

### 2. Curate millions of pages over several Jobs

[`inference/examples/datatrove_curation.py`](inference/examples/datatrove_curation.py) is a
[datatrove](https://github.com/huggingface/datatrove) pipeline. You run it on your machine; it launches one GPU Job per
task and writes to a [storage bucket](https://huggingface.co/docs/huggingface_hub/guides/buckets):

```bash
OUT="hf://buckets/YOUR_NAME/YOUR_BUCKET/fineweb-1m"
uv run --python 3.12 --with "datatrove[io,processing]==0.10.1" --with "huggingface_hub>=1.32" \
    inference/examples/datatrove_curation.py --out "$OUT" --source hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2025-26 \
    --glob "000_0000*.parquet" --limit 170000 --tasks 6 --timeout 40m
```

That command scored 1,020,000 FineWeb pages in about 19 minutes for about $5
([result](https://huggingface.co/datasets/davanstrien/fineweb-curated-1m)). The model card covers the System One API
(`jev_api.py`) and the packed batch scorer (`fast_inference.py`).

### 3. Train the model from the released data

[`data/release_to_jsonl.py`](data/release_to_jsonl.py) converts the
[released dataset](https://huggingface.co/datasets/davanstrien/curation-decision-mix) to the trainer's input, and
[`train/gliclass_ub_train.py`](train/gliclass_ub_train.py) trains on one GPU Job. Both are uv scripts whose
`[tool.hf-jobs]` headers set the hardware, so each step is one command. A bucket mounted at `/b` carries the data
from the first Job to the second, and the trainer syncs its outputs to `--out-bucket`:

```bash
hf buckets create YOUR_NAME/curation-train
hf jobs uv run -v hf://buckets/YOUR_NAME/curation-train:/b data/release_to_jsonl.py -- --out /b/mix.jsonl
hf jobs uv run -v hf://buckets/YOUR_NAME/curation-train:/b train/gliclass_ub_train.py -- \
    --run-name seed0 --seed 0 --encoder dmis-lab/UltraBERT-380M --train /b/mix.jsonl \
    --single-label-loss softmax_ce --encoder-lr 2e-5 --others-lr 4e-5 --max-steps 12000 --save-model --no-trackio \
    --out-bucket hf://buckets/YOUR_NAME/curation-train/runs
```

One run takes about an hour (about $3). Run with `--max-steps 50` first to check the setup (tested on `rtx-pro-6000`:
the Job installs its packages, trains and saves in a few minutes). This trains on the 475,074 released rows that carry their text; the release model also saw
the `web` rows (rebuild their text, see the dataset card) and 18,766 base rows that are not released. For the
release model, train seeds 0, 1 and 2 and follow the averaging steps under [Training](#training).

## The data recipe

The mix has 535,730 rows from nine source groups; 516,964 are released (18,766 base rows without a verified open
licence are left out). The [dataset card](https://huggingface.co/datasets/davanstrien/curation-decision-mix) explains
each choice and the measured effect; this table maps each step to its script.

| Step                        | What it does                                                                                                                                                              | Script(s)                                                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 1. Freeze the evals first   | Build the curation test sets from public sources and freeze them before any training data exists. Every later mix is screened against them.                               | `eval/evals_freeze.py`                                                                                                     |
| 2. Public base              | Convert tasksource classification tasks and Knowledgator's GLiClass data to one row format; drop tasks that overlap a benchmark by name, canary or text.                  | `data/prep_tasksource.py`, `data/prep_stage1.py`, `data/prep_stage2.py`, `data/canary_fix_r1g.py`                          |
| 3. Audit sources            | Text-blind baselines per source (majority label, longest option, options-only ranker) to catch shortcut leaks.                                                            | `data/c1_source_audit.py`                                                                                                  |
| 4. Base sampling            | Register the cleaned, audited base as a frozen source with its sampling rule.                                                                                             | `data/cur_base_source.py`                                                                                                  |
| 5. Curation slices          | Web and prompt rows that reuse public curation labels (WebOrganizer topic and format, FineWeb-Edu, NuminaMath-1.5, DeepMath-103K, WebInstruct-verified, LMArena tags).    | `data/cur_build.py`, `data/prompt_build.py`, `data/panel_targets.py` (row selection: see note), `data/sg_screen.py`        |
| 6. Human and program labels | Library catalogue records (National Library of Scotland, Harvard), arXiv author categories, program-computed reasoning and code tasks.                                    | `data/cat_nls_build.py`, `data/cat_harvard_build.py`, `data/fh_arxiv_build.py`, `data/g_procpile.py`, `data/tool_build.py` |
| 7. DecisionBank             | Licence-filtered DecisionBank rows; held-out test families dropped.                                                                                                       | `data/db_build.py`                                                                                                         |
| 8. Freeze the mix           | Combine the sources by the spec, decontaminate against every eval text, and write a manifest with hashes.                                                                 | `data/mix_build.py`, `mixes/spec_v1.yaml`, `tests/test_mix_build.py`                                                       |
| 9. Release                  | Licence filter per row, remove web text, write one config per source.                                                                                                     | `data/release_build.py`                                                                                                    |
| 10. Back to trainer input   | Convert the release to the trainer's JSONL. Checked against the private training mix: all 475,074 released rows with text are identical on every field the trainer reads. | `data/release_to_jsonl.py`                                                                                                 |

**Note on row selection.** The web and prompt rows in the mix are the rows that an earlier labelling experiment
(a panel of open models) also covered; the release keeps the **original** labels on those rows, not the panel's.
Rerunning that panel costs money, so the dataset repo includes the frozen list of all 535,730 training row ids
(`manifest/training_mix_manifest.parquet`) instead.

## Training

| Step      | What it does                                                             | Script                                                                                   |
| --------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Train     | GLiClass uni-encoder head on UltraBERT-380M; 12,000 steps; seeds 0, 1, 2 | `train/gliclass_ub_train.py` (the original launcher was `train/cur_train.sh`)            |
| Average   | Average the three **encoders** only (seed 0 first, so its head is kept)  | `train/soup.py --average-prefix model.encoder_model OUT RUN0 RUN1 RUN2`                  |
| Head tune | 500 steps from the averaged model with the encoder frozen                | trainer with `--base-model OUT --encoder-lr 0 --others-lr 4e-5 --max-steps 500 --seed 0` |

Settings for the three seed runs: `--single-label-loss softmax_ce --encoder-lr 2e-5 --others-lr 4e-5 --max-steps 12000`,
token budget 32,768 per batch, training length 4,096, evaluation length 2,048.

The trainer uses the [GLiClass](https://github.com/Knowledgator/GLiClass) library (0.1.21) for the model classes and
data processing, and follows the configuration of Knowledgator's
[gliclass-v4-edge](https://huggingface.co/knowledgator/gliclass-v4-edge). It has its own training loop instead of
GLiClass's `Trainer`; the docstring at the top of the file lists each difference and why. One run takes about an hour on one
RTX PRO 6000 (peak GPU memory about 55 GB). For many runs, `image/Dockerfile` builds an image with the same packages
preinstalled (plus the evaluation dependencies), published as `hf.co/spaces/davanstrien/ub-train-image`; the original
runs used it, through `train/cur_train.sh`. The uv header pins the same versions, and throughput is the same (about
50K tokens/s on `rtx-pro-6000`); the image only saves the few minutes of package installation per Job.

The release runs used an earlier revision of `gliclass_ub_train.py` (sha256 `a041428e…`). The file here adds a uv
script header and two opt-in options, `--save-every` (periodic checkpoints) and `--dev-holdout` (a dev split); with the defaults, training is
unchanged (checked at the time: identical training losses). GPU training is not bit-for-bit deterministic.

## Evaluation

`eval/cur_score.py` scores prediction files per test set; `eval/cur_gate.py` computes the curation score (16 units:
a set asked with label descriptions is averaged with its plain twin, and the ten LMArena questions count as one unit)
with bootstrap intervals, split into same-labeller and independent gold. `eval/cur_edge_eval.py` and
`eval/d1_eval.py` run the comparison models on the same inputs.

## How reproducible is this?

| Part                                                                      | Status                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Using the released model (`curate.py`, datatrove example, API, fast path) | Runs as published                                                                                                                                                                                                                                      |
| Released data (516,964 rows)                                              | Download from the Hub; `web` rows need their text rebuilt from the source datasets (snippet in the dataset card)                                                                                                                                       |
| Data builders (`data/`)                                                   | The code that ran, as it ran. Paths assume a bucket mounted at `/b` or `/c` and outputs under it; adapt the mount and the bucket names (`YOUR_NAME/...`). Comments refer to the original layout (`scripts/…`). Not packaged as a one-command pipeline. |
| Exact training mix                                                        | Not fully public: 18,766 base rows lack a verified open licence and web text is not redistributed. The manifest lists every row id and text hash.                                                                                                      |
| Training                                                                  | Runs from the released data on one GPU Job (step 3 above; tested). Not the exact mix: web rows need rebuilt text, and 18,766 base rows are not released                                                                                                |
| Evaluation                                                                | Builders included; the eval rows themselves are not redistributed. The Decision Index suite is not included.                                                                                                                                           |

## Ideas drawn on

| Idea                                               | Source                                                                                                                                                                  | Where it is applied                                 |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Average the weights of several fine-tuned runs     | *Model soups* ([Wortsman et al., 2022](https://arxiv.org/abs/2203.05482))                                                                                               | `train/soup.py` (encoders only)                     |
| Text-blind baselines to find shortcuts in a source | hypothesis-only baselines ([Poliak et al., 2018](https://arxiv.org/abs/1805.01042)); annotation artifacts ([Gururangan et al., 2018](https://arxiv.org/abs/1803.02324)) | `data/c1_source_audit.py`                           |
| Topic and format taxonomies for web pages          | *Organize the Web* ([Wettig et al., 2025](https://arxiv.org/abs/2502.10341))                                                                                            | `data/cur_build.py`                                 |
| Educational-value labels                           | FineWeb-Edu ([Penedo et al., 2024](https://arxiv.org/abs/2406.17557))                                                                                                   | `data/cur_build.py`                                 |
| Independent gold from a separate taxonomy          | Essential-Web ([Essential AI, 2025](https://arxiv.org/abs/2506.14111))                                                                                                  | `eval/evals_freeze.py`                              |
| Base mix and model configuration                   | GLiClass v4 (Knowledgator)                                                                                                                                              | `data/prep_stage1.py`, `train/gliclass_ub_train.py` |

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
