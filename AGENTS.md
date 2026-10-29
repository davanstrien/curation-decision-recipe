# AGENTS.md

Notes for coding agents (and people) working with this repo. The README explains the recipe; this file says how to
run things and what to watch for.

## What this repo is

The scripts behind [davanstrien/ultrabert-380m-curation](https://huggingface.co/davanstrien/ultrabert-380m-curation)
(a 380M-parameter decision model for data curation) and its training data,
[davanstrien/curation-decision-mix](https://huggingface.co/datasets/davanstrien/curation-decision-mix). Most scripts
are the code that ran, kept as a record; the scripts under `inference/`, `data/release_to_jsonl.py` and
`train/gliclass_ub_train.py` are the ones meant to be run by others.

| Folder       | Contents                                                                                                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `inference/` | `curate.py` (one-command curation of a Hub dataset), `examples/datatrove_curation.py` (multi-Job curation), `fast_inference.py` (packed GPU scorer), `jev_api.py` (System One decision API) |
| `data/`      | data builders, in recipe order (see the README table); `release_to_jsonl.py` converts the released dataset to trainer input                                                                 |
| `train/`     | `gliclass_ub_train.py` (trainer), `soup.py` (encoder averaging), `cur_train.sh` (the original launcher)                                                                                     |
| `eval/`      | scoring and the curation-score gate; the eval rows themselves are not published                                                                                                             |
| `image/`     | the training image (published as `hf.co/spaces/davanstrien/ub-train-image`)                                                                                                                 |

## Run everything on Hugging Face Jobs

Compute runs on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs); the laptop only
launches. Inputs and outputs live in Hub datasets or storage buckets, mounted into the Job with `-v`.

- Scripts with a `# /// script` header run with `hf jobs uv run script.py -- <args>`. A `[tool.hf-jobs]` table in
  the header (as in `inference/curate.py`) sets the flavor, timeout and secrets; otherwise pass them as flags
  (`--flavor l4x1 -s HF_TOKEN`). Put `--` before the script's arguments, or `hf` may take flags such as `--dry-run`
  for itself.
- Scripts without that header run in the training image:
  `hf jobs run --flavor rtx-pro-6000 -s HF_TOKEN -v hf://buckets/<you>/<bucket>:/b hf.co/spaces/davanstrien/ub-train-image -- python /b/<script>.py ...`
- The datatrove example is the exception: run it on your machine with `uv run --python 3.12 ...`; it launches the
  Jobs itself.
- GPU scripts need Ampere or newer (`l4x1`, `a10g-small`, `a100-large`, `rtx-pro-6000`), not `t4`.
- Start small (`--limit`, `--max-steps`) and check the output before a large run.

## Conventions and gotchas

- `YOUR_NAME` stands for your Hub user or organisation name. Paths such as `/b` and `/c` are bucket mount points.
- Comments and usage lines in the record scripts refer to the original layout (`scripts/…`, `runs/…`); here the
  files are in `data/`, `train/`, `eval/` and `inference/`.
- Web text (`web` config) is not redistributed: rebuild it from the source datasets for your own use only.
- Filter on `licence_class` / `licence` before you redistribute any part of the training data.
- The topic label "Electronics & Hardare" is spelled as in the WebOrganizer training labels. Keep that spelling in
  the options you send to the model; rename it in your output.
- Graded questions (educational value, difficulty) are ordered options. Use the expected level or a rank
  correlation, not top-1 accuracy alone.
- Tests: `uv run --with pytest --with polars --with pyyaml --with torch --with numpy --with "gliclass==0.1.21" --with "transformers==5.20.2" pytest tests/`.
