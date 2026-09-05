#!/usr/bin/env bash
# Driver for tests/test_symm_mem.py — standalone torch symmetric-memory
# allreduce save/load probe (no vLLM init; needs ~4 GB free per GPU, NOT a
# fully empty GPU).
#
#   two_shot   in-place two-shot allreduce (what vLLM uses at TP=2 on sm90)
#   multimem   multimem/NVLS allreduce (what vLLM uses at ws 4/6/8; needs MC)
#
# Usage: run_symm_mem.sh <two_shot|multimem> <save|load|both>
set -euo pipefail

ROOT=/data/liuxs/workarea/foundry-org
PY=$ROOT/foundry_env/bin/python
TEST=$ROOT/foundry/tests/test_symm_mem.py
HOOK_SO=$($PY -c "import foundry.ops, pathlib; print(pathlib.Path(foundry.ops.__file__).parent / 'libcuda_hook.so')")

ALGO=${1:?usage: run_symm_mem.sh <two_shot|multimem|one_shot> <save|load|both>}
PHASE=${2:?usage: run_symm_mem.sh <two_shot|multimem|one_shot> <save|load|both>}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TEST_SYMM_ALGO=$ALGO
# Keep NCCL off the VMM fast paths (same setting as the DeepEP matrix runs).
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

case "$ALGO" in
  two_shot) PORT=29711 ;;
  multimem) PORT=29721 ;;
  one_shot) PORT=29731 ;;
  *) echo "unknown algo: $ALGO"; exit 2 ;;
esac

# GPUs are shared with other users — refuse to run without headroom.
for i in ${CUDA_VISIBLE_DEVICES//,/ }; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $i)
  if [ "$free" -lt 6000 ]; then
    echo "GPU $i has only ${free} MiB free — aborting (need ~6 GB headroom)"; exit 3
  fi
done

WORK=$ROOT/foundry/tests/symm_mem_work/$ALGO
LOGDIR=$ROOT/logs/symm_mem
mkdir -p "$WORK" "$LOGDIR"
cd "$WORK"   # symm_mem_archive/ is CWD-relative

export LD_PRELOAD=$HOOK_SO${LD_PRELOAD:+:$LD_PRELOAD}

run_phase() {
  local p=$1
  if [ "$p" = save ]; then export MASTER_PORT=$PORT; else export MASTER_PORT=$((PORT + 1)); fi
  echo "=== algo=$ALGO phase=$p $(date '+%F %T') ==="
  if [ "$p" = save ]; then rm -rf symm_mem_archive; fi
  $PY "$TEST" --$p --num-processes=2 2>&1 | tee "$LOGDIR/${ALGO}_${p}.log"
}

if [ "$PHASE" = both ]; then run_phase save; run_phase load; else run_phase "$PHASE"; fi
