#!/bin/bash
# Usage: bash serve_qwen3-30ba3bfp8_ipc_ep.sh <ep_size> [--save|--load]
#
# EXPERIMENTAL: FP8 Qwen3-30B-A3B with DeepGEMM MoE + DeepEP expert-parallel,
# keeping the legacy CUDA-IPC NVLink buffer ON under foundry (FOUNDRY_DEEPEP_NVL_IPC=1).
# Same as serve_qwen3-30ba3b_ipc_ep.sh but: FP8 model + VLLM_USE_DEEP_GEMM=1
# (blockscale FP8 MoE), and its own archive (foundry_archive_ipc_fp8).
#
# Run every phase from one consistent path (or absolute path) so SAVE pass 1/2
# and LOAD pass the identical graph_extension_config_path — see README.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EP_SIZE=${1:?Usage: $0 <ep_size> [--save|--load]}
MODEL_NAME="Qwen/Qwen3-30B-A3B-FP8"
HOST="0.0.0.0"
PORT=12000
GPU_MEMORY_UTILIZATION=0.8

FOUNDRY_ARGS=()
if [[ "$2" == "--save" ]]; then
    FOUNDRY_ARGS+=( --compilation-config.graph_extension_config_path "${SCRIPT_DIR}/foundry_save_fp8.toml" )
    export NCCL_CUMEM_ENABLE=0
    export NCCL_NVLS_ENABLE=0
    export VLLM_USE_V2_MODEL_RUNNER=0
    export FOUNDRY_DEEPEP_NVL_IPC=1
    echo "Using foundry SAVE (FP8/DeepGEMM, DeepEP NVL/IPC): ${SCRIPT_DIR}/foundry_save_fp8.toml"
elif [[ "$2" == "--load" ]]; then
    FOUNDRY_ARGS+=( --compilation-config.graph_extension_config_path "${SCRIPT_DIR}/foundry_load_fp8.toml" )
    export NCCL_CUMEM_ENABLE=0
    export NCCL_NVLS_ENABLE=0
    export VLLM_USE_V2_MODEL_RUNNER=0
    export FOUNDRY_DEEPEP_NVL_IPC=1
    echo "Using foundry LOAD (FP8/DeepGEMM, DeepEP NVL/IPC): ${SCRIPT_DIR}/foundry_load_fp8.toml"
elif [[ -n "$2" ]]; then
    echo "Usage: $0 <ep_size> [--save|--load]"
    exit 1
else
    echo "Running without foundry (baseline vLLM)"
fi

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
# FP8 blockscale MoE via DeepGEMM.
export VLLM_USE_DEEP_GEMM=1

CUDAGRAPH_CAPTURE_SIZES=($(seq 1 256))

ARGS=(
    --trust-remote-code
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size 1
    --data-parallel-size "$EP_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --distributed-executor-backend uni
    --enable-expert-parallel
    --all2all-backend deepep_low_latency
    --no-enable-prefix-caching
    --max-num-batched-tokens 256
    --max-num-seqs 256
    --attention-config.backend FLASH_ATTN
    --compilation-config.cudagraph_mode FULL_DECODE_ONLY
    --compilation-config.cudagraph_num_of_warmups 0
    --chat-template-content-format string
    --cudagraph-capture-sizes ${CUDAGRAPH_CAPTURE_SIZES[@]}
)

vllm serve "$MODEL_NAME" "${ARGS[@]}" "${FOUNDRY_ARGS[@]}"
