#!/usr/bin/env bash
# Matrix driver for tests/test_deepep_fabric.py — standalone DeepEP save/load probes
# (no vLLM/sglang init; needs ~4 GB free on each of GPUs 0,1, NOT a fully empty GPU).
#
#   ll_nofabric   pure low-latency (NVSHMEM heap, FD handles), NCCL fast paths off
#                 -> the "EP without fabric" baseline; expected PASS
#   ll_ncclcumem  same, but NCCL_CUMEM_ENABLE=1   -> probe: can NCCL cumem coexist?
#   ll_ncclnvls   same, but CUMEM=1 + NVLS=1      -> probe: can NCCL NVLS coexist?
#   nvl_ipc       + 64 MB NVL buffer, use_fabric=0 -> legacy cudaIpc path through the
#                 hook's VMM-IPC translation (SCM_RIGHTS fd transport; peer mappings
#                 relocate under shared region bases); expected PASS
#   nvl_fabric    + 64 MB NVL buffer, use_fabric=1 -> fabric cuMemCreate on a no-IMEX
#                 machine; documents the fabric dependency; expected FAIL
#
# Usage: run_deepep_matrix.sh <case> <save|load|both>
set -euo pipefail

ROOT=/data/liuxs/workarea/foundry-org
PY=$ROOT/foundry_env/bin/python
TEST=$ROOT/foundry/tests/test_deepep_fabric.py
NVSHMEM_SO=$ROOT/vllm/tools/ep_kernels/ep_kernels_workspace/nvshmem/lib/libnvshmem_host.so
HOOK_SO=$($PY -c "import foundry.ops, pathlib; print(pathlib.Path(foundry.ops.__file__).parent / 'libcuda_hook.so')")

CASE=${1:?usage: run_deepep_matrix.sh <case> <save|load|both>}
PHASE=${2:?usage: run_deepep_matrix.sh <case> <save|load|both>}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# Pin NVSHMEM heap chunks to POSIX FD handles — explicit "no fabric handles anywhere".
# (Auto-detect picks FD on non-MNNVL machines anyway; pinning removes the variable.)
export NVSHMEM_CUMEM_HANDLE_TYPE=FILE_DESCRIPTOR

case "$CASE" in
  ll_nofabric)  export TEST_USE_FABRIC=0 TEST_NVL_BYTES_MB=0  NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0; PORT=29611 ;;
  ll_ncclcumem) export TEST_USE_FABRIC=0 TEST_NVL_BYTES_MB=0  NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0; PORT=29621 ;;
  ll_ncclcumem_preinit) export TEST_USE_FABRIC=0 TEST_NVL_BYTES_MB=0 NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=0 TEST_NCCL_WARMUP_PRE_REGION=1; PORT=29661 ;;
  ll_ncclnvls)  export TEST_USE_FABRIC=0 TEST_NVL_BYTES_MB=0  NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1; PORT=29631 ;;
  ll_ncclnvls_preinit) export TEST_USE_FABRIC=0 TEST_NVL_BYTES_MB=0 NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1 TEST_NCCL_WARMUP_PRE_REGION=1; PORT=29671 ;;
  nvl_ipc)      export TEST_USE_FABRIC=0 TEST_NVL_BYTES_MB=64 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0; PORT=29641 ;;
  nvl_fabric)   export TEST_USE_FABRIC=1 TEST_NVL_BYTES_MB=64 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0; PORT=29651 ;;
  nvl_ipc_prealloc) export TEST_USE_FABRIC=0 TEST_NVL_BYTES_MB=64 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 TEST_LOAD_PREALLOC_MB=1024; PORT=29681 ;;
  *) echo "unknown case: $CASE"; exit 2 ;;
esac

# GPUs are shared with other users — refuse to run without headroom.
# Check exactly the GPUs we'll run on (CUDA_VISIBLE_DEVICES), not a hardcoded 0,1.
for i in ${CUDA_VISIBLE_DEVICES//,/ }; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $i)
  if [ "$free" -lt 6000 ]; then
    echo "GPU $i has only ${free} MiB free — aborting (need ~6 GB headroom)"; exit 3
  fi
done

# main() overwrites TEST_USE_FABRIC from the CLI flag, so pass the flag too.
FABRIC_ARG=""
[ "$TEST_USE_FABRIC" = "0" ] && FABRIC_ARG="--no-fabric"

WORK=$ROOT/foundry/tests/deepep_matrix_work/$CASE
LOGDIR=$ROOT/logs/deepep_matrix
mkdir -p "$WORK" "$LOGDIR"
cd "$WORK"   # deepep_fabric_archive/ is CWD-relative

export LD_PRELOAD=$NVSHMEM_SO:$HOOK_SO${LD_PRELOAD:+:$LD_PRELOAD}

run_phase() {
  local p=$1
  # distinct rendezvous port per phase to dodge TIME_WAIT
  if [ "$p" = save ]; then export MASTER_PORT=$PORT; else export MASTER_PORT=$((PORT + 1)); fi
  echo "=== case=$CASE phase=$p $(date '+%F %T') ==="
  echo "env: TEST_USE_FABRIC=$TEST_USE_FABRIC TEST_NVL_BYTES_MB=$TEST_NVL_BYTES_MB" \
       "NCCL_CUMEM_ENABLE=$NCCL_CUMEM_ENABLE NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE" \
       "NVSHMEM_CUMEM_HANDLE_TYPE=$NVSHMEM_CUMEM_HANDLE_TYPE MASTER_PORT=$MASTER_PORT"
  if [ "$p" = save ]; then rm -rf deepep_fabric_archive; fi
  $PY "$TEST" --$p $FABRIC_ARG --num-processes=2 2>&1 | tee "$LOGDIR/${CASE}_${p}.log"
}

if [ "$PHASE" = both ]; then run_phase save; run_phase load; else run_phase "$PHASE"; fi
