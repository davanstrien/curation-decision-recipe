#!/usr/bin/env bash
# Curation training run on a frozen mix: calls the trainer directly (not arm_run.sh,
# which builds its own mix inside the job). Publishes the trainer to the cur-v1 artifact dir and passes its sha256
# (the job stops if the deployed copy differs), adds every set of the eval version, records provenance (mix and eval
# hashes, train ids) in the run dir, logs to Trackio project dm-curation, and appends the launch to runs/launches.tsv.
# usage: scripts/cur_train.sh RUN SEED ENCODER MIX EVAL_VERSION [trainer args ...]
#   MIX: path under dm-curation, e.g. mixes/cur-v1/v1 ; EVAL_VERSION: e.g. v1
#   env: MAX_STEPS (6000) FLAVOR (rtx-pro-6000) TIMEOUT (3h) DRY_RUN=1 (print only)
set -euo pipefail
run=$1; seed=$2; enc=$3; mix=$4; ev=$5; shift 5
art=hf://buckets/YOUR_NAME/jobs-artifacts/cur-v1
inv=scratch/evals/$ev/inventory.json
[[ -f $inv ]] || { echo "missing $inv (local copy of evals/$ev)"; exit 1; }
evals=$(python3 -c "
import json,sys
inv=json.load(open('$inv'))
print(' '.join(f'--eval cur_{s}=/c/evals/$ev/{v[\"file\"]}' for s,v in sorted(inv['sets'].items())))")
sha=$(shasum -a 256 scripts/gliclass_ub_train.py | cut -d' ' -f1)
cmd="python /s/gliclass_ub_train.py --run-name $run --seed $seed --encoder $enc --train /c/$mix/mix.jsonl \
--mix /c/$mix --eval-version /c/evals/$ev --expect-sha256 $sha \
--single-label-loss softmax_ce --encoder-lr 2e-5 --others-lr 4e-5 --max-steps ${MAX_STEPS:-6000} \
--eval-max-length 2048 --eval-steps 3000 --save-model \
--trackio-project dm-curation --trackio-space YOUR_NAME/ub-gliclass-trackio --trackio-bucket YOUR_NAME/ub-gliclass-trackio \
--out-bucket hf://buckets/YOUR_NAME/dm-curation/runs $evals $*"
if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "$cmd"; exit 0; fi
hf buckets ls hf://buckets/YOUR_NAME/dm-curation/$mix/mix.jsonl >/dev/null 2>&1 || { echo "no mix at dm-curation/$mix"; exit 1; }
hf buckets ls hf://buckets/YOUR_NAME/dm-curation/evals/$ev/inventory.json >/dev/null 2>&1 || { echo "no evals/$ev"; exit 1; }
image_sha=$(hf spaces info YOUR_NAME/ub-train-image --format json 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('sha',''))" || true)
hf buckets cp scripts/gliclass_ub_train.py $art/gliclass_ub_train.py >/dev/null
remote=$(hf buckets cp $art/gliclass_ub_train.py - 2>/dev/null | shasum -a 256 | cut -d' ' -f1 || true)
[[ -z "$remote" || "$remote" == "$sha" ]] || { echo "artifact sha $remote != local $sha"; exit 1; }
out=$(hf jobs run --detach --flavor "${FLAVOR:-rtx-pro-6000}" --timeout "${TIMEOUT:-3h}" -s HF_TOKEN \
  -e MAX_STEPS="${MAX_STEPS:-6000}" -e IMAGE=hf.co/spaces/YOUR_NAME/ub-train-image -e SPACE_REPO_SHA="$image_sha" \
  -e ARTIFACT_URI="$art" -e WRAPPER_SHA256="$(shasum -a 256 "$0" | cut -d' ' -f1)" \
  -v hf://buckets/YOUR_NAME/dm-curation:/c -v $art:/s:ro \
  hf.co/spaces/YOUR_NAME/ub-train-image -- bash -c "$cmd" 2>&1)
job=$(echo "$out" | grep -oE "[A-Za-z0-9_-]+/[0-9a-f]{24}" | head -1)
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$run" "$job" "$sha" "$mix" "$ev" >> runs/launches.tsv
echo "$run $job (trainer $sha)"
