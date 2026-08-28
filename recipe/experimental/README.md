# Foundry recipe — experimental (DeepEP legacy CUDA-IPC NVL buffer)

End-to-end SAVE / LOAD recipe for **Qwen3-30B-A3B** expert-parallel where DeepEP's
intranode NVLink buffer stays on the **legacy CUDA-IPC** path
(`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`) under foundry, instead of the
default fabric/NVSHMEM-only path used by `recipe/vllm/serve_qwen3-30ba3b_ep.sh`.

This exercises foundry's **VMM-IPC translation layer**: DeepEP's NVL buffer is a
foundry VMM (`cuMemCreate`) allocation that legacy IPC can't share, so the hook
exports it as a POSIX fd and transports that fd to the peer rank via
`SCM_RIGHTS` over a per-process unix socket, then maps it into the importer's
own VA range. This is the path that lets foundry SAVE/LOAD work on machines
**without fabric / IMEX**, where DeepEP would otherwise fail at `Buffer.sync`
with `cuMemImportFromShareableHandle ... error 999`.

```
recipe/experimental/
├── README.md                          # this file
├── foundry_save.toml                  # SAVE config (workspace_root = "foundry_archive_ipc")
├── foundry_load.toml                  # LOAD config (same workspace_root)
└── serve_qwen3-30ba3b_ipc_ep.sh       # Qwen3-30B-A3B (MoE)  EP, DeepEP NVL/IPC
```

It is the standard `recipe/vllm` EP recipe plus one switch — `FOUNDRY_DEEPEP_NVL_IPC=1`
— so read [`../vllm/README.md`](../vllm/README.md) first for installation, the
two-pass SAVE workflow, the archive layout, and the shared EP flags. Only the
IPC-specific deltas are documented here.

## Required code

The IPC path needs only **Foundry** changes beyond the standard install — no
vLLM edit:

- **Foundry** — `foundry/csrc/hook.cpp`: the SCM_RIGHTS VMM-IPC fd transport +
  whole-chunk peer mapping (handles LOAD-mode buffers carved from the
  preallocated chunk). **C++ change → rebuild required:**
  `uv pip install -e . --no-build-isolation` in `foundry/`.
- **Foundry** — `foundry/python/foundry/integration/vllm/hooks.py`: the
  `FOUNDRY_DEEPEP_NVL_IPC` env knob in `_patch_deepep` (Python, no rebuild).

### Run all phases from one consistent path (no vLLM edit needed)

vLLM folds the foundry TOML path (`graph_extension_config_path`) into its
torch.compile cache key even though the path never affects codegen. If SAVE
pass 1 and pass 2 see that path with *different spellings* — e.g. two mount
aliases of the same dir (`/data/...` vs `/home/...`), or you move the TOML
between passes — pass 2 misses pass 1's warm compile cache and inductor
recompiles **inside** the cuda-graph capture window, where its combo-kernel
benchmark does an illegal `torch.randn` → `cudaErrorStreamCaptureInvalidated`.

The fix is operational, not code: **invoke the script the same way for every
phase** (same shell / same `cd`, or always an absolute path), so SAVE pass 1,
SAVE pass 2, and LOAD all pass the identical path string → identical cache hash
→ pass 2 reuses pass 1's compiled kernels. Run from the workspace root, e.g.:

```bash
cd <workspace>          # one canonical path for all three phases
bash foundry/recipe/experimental/serve_qwen3-30ba3b_ipc_ep.sh 2 --save
bash foundry/recipe/experimental/serve_qwen3-30ba3b_ipc_ep.sh 2 --save
bash foundry/recipe/experimental/serve_qwen3-30ba3b_ipc_ep.sh 2 --load
```

## Workflow

Same two-pass SAVE → LOAD as the base recipe; `<ep_size>` is the first arg:

```bash
# 0. Fresh start (distinct workspace from the base recipe)
rm -rf foundry_archive_ipc

# 1. SAVE pass 1 — memory profile + capture
bash serve_qwen3-30ba3b_ipc_ep.sh 2 --save
# wait for "Application startup complete", then Ctrl-C

# 2. SAVE pass 2 — deterministic re-capture
bash serve_qwen3-30ba3b_ipc_ep.sh 2 --save
# wait for "Application startup complete", then Ctrl-C

# 3. LOAD — preallocate, re-import IPC buffers, replay graphs
bash serve_qwen3-30ba3b_ipc_ep.sh 2 --load
# leave running

# 4. Query (separate shell)
bash ../../../experimental/query.sh 12000 Qwen/Qwen3-30B-A3B
```

Uncomment `nvshmem_host_path` in both TOMLs first (EP still needs NVSHMEM for the
DeepEP RDMA buffer; the IPC path only changes the NVL buffer).

## What `FOUNDRY_DEEPEP_NVL_IPC=1` does

`_patch_deepep` (`foundry/python/foundry/integration/vllm/hooks.py`) normally
forces `use_fabric=True, num_nvl_bytes=0` so the only cross-GPU buffer is the
NVSHMEM symmetric heap. With `FOUNDRY_DEEPEP_NVL_IPC=1` it instead keeps
upstream's nonzero `num_nvl_bytes` with `use_fabric=False`, so the DeepEP Buffer
allocates the NVLink buffer via `cudaMalloc` + `cudaIpcGetMemHandle` on **both**
SAVE and LOAD. The hook then:

- **SAVE**: exports each VMM-backed NVL buffer as a POSIX fd, served to peers
  over `\0foundry-vmm-ipc.<pid>` via `SCM_RIGHTS` (same-uid `SO_PEERCRED` check,
  per-process token vs PID reuse).
- **LOAD**: buffers are carved from the preallocated chunk (no individual
  handle), so the hook exports the **whole chunk** fd + offset; the peer maps the
  entire chunk once and returns an interior pointer.
- Peer mappings land at a **relocated** VA (logged `[HOOK] INFO: VMM-IPC import
  relocated`) — correct, because DeepEP resolves peers through its device-side
  `buffer_ptrs_gpu` table (refreshed by `Buffer.sync`), never through addresses
  baked into captured graphs.

A healthy run logs two `VMM-IPC import relocated` lines per phase (one per peer)
and **no** `error 999`.

## Validation status

SAVE → SAVE → LOAD → query verified on Qwen3-30B-A3B EP=2 (2× H200, no IMEX):
per-rank `final_alloc_offset` identical across passes, LOAD replays at the saved
offset, query returns coherent completions. LOAD reaches a serving server in
~27 s; the IPC import itself is sub-second (inside the ~7.5 s `load_model`) and
has no steady-state serving cost — peer addresses are resolved once at init via
the device pointer table, not per token.

## IPC-specific troubleshooting

| Symptom | Likely cause |
|---|---|
| `[HOOK] ERROR: cuMemImportFromShareableHandle failed with error 999` at `deep_ep.cpp` `Buffer::sync` | Foundry hook not rebuilt with the SCM_RIGHTS transport — it's still packing a raw fd. Rebuild `foundry/` (`uv pip install -e . --no-build-isolation`). |
| `operation failed due to a previous error during capture` on SAVE **pass 2** | vLLM compile-cache over-keying not applied (`graph_extension_config_path` still hashed) → recompile-in-capture. Apply the `compilation.py` `ignored_factors` edit, or set `FOUNDRY_DISABLE_COMBO_KERNELS=1` (opt-in belt-and-suspenders, wired in the experimental serve script under `experimental/expert-parallel/`). |
| LOAD `illegal memory access` at replay with an NVL buffer present | Relocated peer import collided with the NVSHMEM heap hint — ensure the hook build includes the dedicated import-VA zone (`0x300000000000`). |
| `error 999` only with `--deepep-mode auto`/`normal` | That's expected for non-LL modes here; this recipe pins `deepep_low_latency`. |
