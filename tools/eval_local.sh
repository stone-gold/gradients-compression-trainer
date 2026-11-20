#!/bin/bash
# Run a command inside the locally built G.O.D quant evaluator image (vLLM 0.31) on one GPU, no network.
# usage: tools/eval_local.sh 'python /tools/local_eval.py eval --base /models/X --test /work/q05/test.jsonl --spec ... --models /out/...'
set -euo pipefail
GPU=${GPU:-2}
docker run --rm --network none --gpus "\"device=$GPU\"" --shm-size 8g \
  -e LOCAL_VLLM_GPU_UTIL=${LOCAL_VLLM_GPU_UTIL:-} -e HUMMING_CACHE_DIR=/hcache -e HUMMING_TMP_DIR=/htmp \
  -v /mnt/sn56/quant/humming-cache:/hcache -v /mnt/sn56/quant/humming-tmp:/htmp \
  -v /mnt/sn56/cache/models:/models:ro -v /mnt/sn56/quant:/work -v /mnt/sn56/quant/out:/out:ro \
  -v /mnt/sn56/ours-quant/tools:/tools:ro \
  --entrypoint bash local/quant-evaluator:godq -c "$1"
