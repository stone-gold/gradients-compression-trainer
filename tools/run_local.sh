#!/bin/bash
# Run the trainer image on a local task the way trainer/runtime.py does (no network, /cache ro, /app/checkpoints rw).
# usage: tools/run_local.sh <work dir name e.g. q05> <task_id> <hf model id> <W2|W3> <hours> <out repo name> [extra docker -e args...]
set -euo pipefail
W=$1; TASK=$2; MODEL=$3; FMT=$4; HOURS=$5; REPO=$6; shift 6
GPU=${GPU:-2}
IMAGE=${IMAGE:-ours-quant:dev}
ROOT=/mnt/sn56/quant
case $FMT in
  W2) SPEC='{"weight_bits":2,"activation":"bf16","group_size":64,"symmetric":false,"scale_dtype":"bfloat16","kv_cache_dtype":"bfloat16","act_order":false}';;
  W3) SPEC='{"weight_bits":3,"activation":"bf16","group_size":128,"symmetric":false,"scale_dtype":"bfloat16","kv_cache_dtype":"bfloat16","act_order":false}';;
esac
REF=$(python3 -c "import json;print(json.load(open('$ROOT/$W/calib.json'))['reference_loss'])")
TARGET="{\"reference_model\":\"$MODEL\",\"reference_loss\":$REF,\"reference_sku\":\"local\"}"
DT=$(cat $ROOT/$W/dataset_type.json)
MOUNT_SRC=()
if [ "${DEV:-1}" = 1 ]; then MOUNT_SRC=(-v /mnt/sn56/ours-quant/quant:/workspace/quant:ro); fi
docker run --rm --network none --gpus "\"device=$GPU\"" --shm-size 8g --cap-drop ALL --security-opt no-new-privileges \
  -e QUANT_SPEC="$SPEC" -e QUANT_TARGET="$TARGET" -e QUANT_CALIBRATION_PATH=/cache/calib_$TASK.json "$@" \
  -v $ROOT/$W/calib.json:/cache/calib_$TASK.json:ro \
  -v /mnt/sn56/cache/models:/cache/models:ro -v $ROOT/cache/datasets:/cache/datasets:ro \
  -v $ROOT/out:/app/checkpoints "${MOUNT_SRC[@]}" \
  $IMAGE --task-id $TASK --model $MODEL --dataset "https://example.invalid/train.json" --dataset-type "$DT" \
  --task-type QuantFixedTask --file-format s3 --expected-repo-name $REPO --hours-to-complete $HOURS --quant-spec "$SPEC"
