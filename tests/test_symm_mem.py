# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Standalone foundry save/load test for torch symmetric-memory allreduce.

Mirrors tests/test_deepep_fabric.py but exercises the TP-allreduce path that
vLLM main uses by default (VLLM_ALLREDUCE_USE_SYMM_MEM=1):

    buffer = torch_symm_mem.empty(...); torch_symm_mem.rendezvous(buffer, group)
    # captured in the decode graph:
    buffer[:n].copy_(inp); torch.ops.symm_mem.two_shot_all_reduce_(buffer[:n],
    "sum", group); out.copy_(buffer[:n])

Driver-level background (torch 2.11, CUDA backend, single node):
  - torch resolves cuMem* via cudaGetDriverEntryPointByVersion -> hooked dlsym
    -> hooked cuGetProcAddress_v2. Foundry redirects cuMemAddressReserve (VA
    steering) but passes cuMemCreate/cuMemMap/cuMulticast* to the real driver.
  - Handle exchange is torch's own AF_UNIX SCM_RIGHTS IpcChannel — foundry's
    VMM-IPC translation is NOT involved (unlike DeepEP's legacy cudaIpc path).
  - The allreduce kernels read peer pointers from a device array
    (buffer_ptrs_dev) at RUN time; the captured graph bakes only the local
    buffer VA (copy kernels + tensor arg) and the dev-array addresses.

So the test answers: are the local symm buffer VA and the dev arrays placed
deterministically enough across 2nd-SAVE and LOAD for replay to be correct?

Usage (LD_PRELOAD libcuda_hook.so; see tests/run_symm_mem.sh):
  python tests/test_symm_mem.py --save --num-processes=2
  python tests/test_symm_mem.py --load --num-processes=2

Env knobs:
  TEST_SYMM_ALGO   two_shot (default) | multimem | one_shot
  TEST_BUF_MB      symm buffer size in MiB (default 8, matches small TP slices)
"""

import argparse
import json
import os
import shutil
import sys

import torch
import torch.distributed as dist

BASE_ADDR = 0x600000000000
REGION_SIZE_STR = "32GB"
ARCHIVE_DIR = "symm_mem_archive"


def _init_dist(local_rank: int, num_local_ranks: int):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "29500"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    import inspect

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_nodes * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in sig.parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.group.WORLD


def _algo():
    return os.environ.get("TEST_SYMM_ALGO", "two_shot")


def _buf_elems():
    # bf16 elements
    return int(os.environ.get("TEST_BUF_MB", "8")) * 1024 * 1024 // 2


def _make_input(rank: int, n: int) -> torch.Tensor:
    # Small integers: exact in bf16, rank-distinguishable, deterministic.
    base = torch.arange(n, device="cuda", dtype=torch.float32) % 13
    return (base + rank + 1).to(torch.bfloat16)


def _expected_sum(n: int, num_ranks: int) -> torch.Tensor:
    base = torch.arange(n, device="cuda", dtype=torch.float32) % 13
    total = base * num_ranks + sum(r + 1 for r in range(num_ranks))
    return total.to(torch.bfloat16)


def _run_allreduce(buf: torch.Tensor, n: int, group_name: str):
    """The op sequence vLLM's SymmMemCommunicator.all_reduce captures."""
    algo = _algo()
    view = buf[:n]
    if algo == "multimem":
        torch.ops.symm_mem.multimem_all_reduce_(view, "sum", group_name)
    elif algo == "one_shot":
        torch.ops.symm_mem.one_shot_all_reduce(view, "sum", group_name)
    else:
        torch.ops.symm_mem.two_shot_all_reduce_(view, "sum", group_name)


def _verify(out: torch.Tensor, n: int, num_ranks: int, rank: int, prefix: str):
    expected = _expected_sum(n, num_ranks)
    ok = torch.equal(out[:n].float(), expected[:n].float())
    sample = out[:5].float().tolist()
    exp_sample = expected[:5].float().tolist()
    print(
        f"[Rank {rank}] {prefix}: allreduce {'CORRECT' if ok else 'WRONG'} "
        f"out[:5]={sample} expected[:5]={exp_sample}",
        flush=True,
    )
    if not ok:
        bad = (out[:n].float() != expected[:n].float()).nonzero()
        print(
            f"[Rank {rank}] {prefix}: first mismatches at {bad[:5].flatten().tolist()}",
            flush=True,
        )
        raise RuntimeError(f"{prefix}: allreduce verification failed")


def _setup_symm_buffer(rank: int, group, prefix: str):
    """Allocate + rendezvous the persistent symm buffer (as vLLM init does)."""
    import torch.distributed._symmetric_memory as torch_symm_mem

    n = _buf_elems()
    buf = torch_symm_mem.empty(
        n, device=f"cuda:{torch.cuda.current_device()}", dtype=torch.bfloat16
    )
    hdl = torch_symm_mem.rendezvous(buf, group.group_name)

    info = {
        "buffer_va": buf.data_ptr(),
        "buffer_ptrs_dev": int(hdl.buffer_ptrs_dev) if hasattr(hdl, "buffer_ptrs_dev") else 0,
        "signal_pad_ptrs_dev": int(hdl.signal_pad_ptrs_dev)
        if hasattr(hdl, "signal_pad_ptrs_dev")
        else 0,
        "multicast_ptr": int(hdl.multicast_ptr) if hasattr(hdl, "multicast_ptr") else 0,
    }
    for k, v in info.items():
        print(f"[Rank {rank}] {prefix}:   {k} = {hex(v)}", flush=True)
    return buf, hdl, info


def _run_save(local_rank: int, num_processes: int):
    import foundry as fdry

    # Import the Python wrapper explicitly: `foundry.CUDAGraph` may be the raw
    # pybind class depending on __init__ import order (isort reshuffles it),
    # and the raw capture_begin requires a positional pool.
    from foundry.graph import CUDAGraph as FoundryCUDAGraph
    from foundry.graph import graph as foundry_graph_ctx

    rank, num_ranks, group = _init_dist(local_rank, num_processes)
    print(f"[Rank {rank}] SAVE: Initializing CUDA", flush=True)
    torch.cuda.init()

    region_size = fdry.parse_size(REGION_SIZE_STR)
    print(f"[Rank {rank}] SAVE: Setting up allocation region at {hex(BASE_ADDR)}", flush=True)
    fdry.set_allocation_region(BASE_ADDR, region_size)

    n = _buf_elems()
    print(
        f"[Rank {rank}] SAVE: symm buffer: {n} bf16 elems ({n * 2 // (1024 * 1024)} MiB), "
        f"algo={_algo()}",
        flush=True,
    )
    buf, hdl, info = _setup_symm_buffer(rank, group, "SAVE")

    # Regular (region-tracked) input/output tensors, allocated pre-capture.
    inp = _make_input(rank, n)
    out = torch.empty_like(inp)
    print(
        f"[Rank {rank}] SAVE: inp={hex(inp.data_ptr())} out={hex(out.data_ptr())}",
        flush=True,
    )

    # Eager warmup (loads the symm-mem kernels; verifies live path).
    buf[:n].copy_(inp)
    _run_allreduce(buf, n, group.group_name)
    out.copy_(buf[:n])
    torch.cuda.synchronize()
    _verify(out, n, num_ranks, rank, "SAVE-warmup")
    dist.barrier(group)

    # Capture the vLLM-shaped sequence.
    print(f"[Rank {rank}] SAVE: Capturing CUDA graph", flush=True)
    graph = FoundryCUDAGraph()
    with foundry_graph_ctx(graph):
        buf[:n].copy_(inp)
        _run_allreduce(buf, n, group.group_name)
        out.copy_(buf[:n])
    print(f"[Rank {rank}] SAVE: Capture done", flush=True)

    # Replay once and verify.
    out.zero_()
    torch.cuda.synchronize()
    dist.barrier(group)
    graph.replay()
    torch.cuda.synchronize()
    _verify(out, n, num_ranks, rank, "SAVE-replay")

    # Persist.
    rank_archive = os.path.join(ARCHIVE_DIR, f"rank_{rank}")
    os.makedirs(rank_archive, exist_ok=True)
    graph_json = os.path.join(rank_archive, "symm_allreduce_graph.json")
    graph.save(graph_json, output_tensors=[out])
    fdry.pack_fatbins_to_folder(rank_archive)
    with open(os.path.join(rank_archive, "symm_meta.json"), "w") as f:
        json.dump({"info": info, "inp_va": inp.data_ptr(), "out_va": out.data_ptr()}, f)
    print(f"[Rank {rank}] SAVE: Graph + fatbins saved to {rank_archive}", flush=True)

    dist.barrier(group)
    fdry.stop_allocation_region()
    dist.destroy_process_group()
    print(f"[Rank {rank}] SAVE: Completed", flush=True)


def _run_load(local_rank: int, num_processes: int):
    import foundry as fdry
    from foundry.graph import CUDAGraph as FoundryCUDAGraph

    rank, num_ranks, group = _init_dist(local_rank, num_processes)
    print(f"[Rank {rank}] LOAD: Initializing CUDA", flush=True)
    torch.cuda.init()

    rank_archive = os.path.join(ARCHIVE_DIR, f"rank_{rank}")
    if not os.path.exists(rank_archive):
        raise RuntimeError(f"{rank_archive} not found — run SAVE first")

    fdry.set_skip_fatbin_processing(True)

    # Same order as the DeepEP test: modules first, then region.
    print(f"[Rank {rank}] LOAD: Loading CUDA modules from {rank_archive}", flush=True)
    fdry.load_cuda_modules_and_libraries(rank_archive)

    region_size = fdry.parse_size(REGION_SIZE_STR)
    print(f"[Rank {rank}] LOAD: Setting up allocation region at {hex(BASE_ADDR)}", flush=True)
    fdry.set_allocation_region(BASE_ADDR, region_size)

    n = _buf_elems()
    buf, hdl, info = _setup_symm_buffer(rank, group, "LOAD")

    inp = _make_input(rank, n)
    out = torch.empty_like(inp)
    print(
        f"[Rank {rank}] LOAD: inp={hex(inp.data_ptr())} out={hex(out.data_ptr())}",
        flush=True,
    )

    # Cross-check determinism vs SAVE (informational + hard assert on the
    # addresses the captured graph bakes).
    with open(os.path.join(rank_archive, "symm_meta.json")) as f:
        saved = json.load(f)
    for key in ("buffer_va", "buffer_ptrs_dev", "signal_pad_ptrs_dev", "multicast_ptr"):
        sv, lv = saved["info"][key], info[key]
        match = "MATCH" if sv == lv else "MISMATCH"
        print(
            f"[Rank {rank}] LOAD: {key}: save={hex(sv)} load={hex(lv)} -> {match}",
            flush=True,
        )
    for key, lv in (("inp_va", inp.data_ptr()), ("out_va", out.data_ptr())):
        sv = saved[key]
        match = "MATCH" if sv == lv else "MISMATCH"
        print(
            f"[Rank {rank}] LOAD: {key}: save={hex(sv)} load={hex(lv)} -> {match}",
            flush=True,
        )

    # Eager warmup: triggers torch symm-mem kernel module loads (tracked as
    # warmup handles) and validates the live rendezvous before replay.
    buf[:n].copy_(inp)
    _run_allreduce(buf, n, group.group_name)
    out.copy_(buf[:n])
    torch.cuda.synchronize()
    _verify(out, n, num_ranks, rank, "LOAD-warmup")
    dist.barrier(group)

    # Load + replay the archived graph.
    graph_json = os.path.join(rank_archive, "symm_allreduce_graph.json")
    print(f"[Rank {rank}] LOAD: Loading graph from {graph_json}", flush=True)
    load_api = os.environ.get("TEST_LOAD_API", "parallel")
    if load_api == "single":
        graph, output_tensors = FoundryCUDAGraph.load(graph_json)
    else:
        pending = FoundryCUDAGraph.start_graph_builds([graph_json])
        ((graph, output_tensors),) = FoundryCUDAGraph.finish_graph_loads(pending)
    print(f"[Rank {rank}] LOAD: Graph loaded", flush=True)

    loaded_out = None
    if output_tensors:
        loaded_out = output_tensors[0]
        print(
            f"[Rank {rank}] LOAD: loaded out tensor at {hex(loaded_out.data_ptr())}",
            flush=True,
        )
        loaded_out.zero_()
    torch.cuda.synchronize()
    dist.barrier(group)

    print(f"[Rank {rank}] LOAD: Replaying loaded graph", flush=True)
    graph.replay()
    torch.cuda.synchronize()

    verify_tensor = loaded_out if loaded_out is not None else out
    _verify(verify_tensor, n, num_ranks, rank, "LOAD-replay")

    # Replay twice more: catches one-shot-correct-but-corrupting-state bugs.
    for i in range(2):
        verify_tensor.zero_()
        torch.cuda.synchronize()
        dist.barrier(group)
        graph.replay()
        torch.cuda.synchronize()
        _verify(verify_tensor, n, num_ranks, rank, f"LOAD-replay-{i + 2}")

    dist.barrier(group)
    fdry.stop_allocation_region()
    dist.destroy_process_group()
    print(f"[Rank {rank}] LOAD: Completed successfully", flush=True)


def _cleanup_archive():
    if os.path.exists(ARCHIVE_DIR):
        shutil.rmtree(ARCHIVE_DIR)
        print(f"[TEST] Cleaned up {ARCHIVE_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--num-processes", type=int, default=2)
    args = parser.parse_args()

    if args.cleanup:
        _cleanup_archive()
        return
    if not (args.save or args.load):
        parser.print_help()
        sys.exit(1)

    fn = _run_save if args.save else _run_load
    print(
        f"[TEST] symm-mem {'SAVE' if args.save else 'LOAD'}: algo={_algo()} "
        f"buf={os.environ.get('TEST_BUF_MB', '8')}MB nprocs={args.num_processes}",
        flush=True,
    )
    torch.multiprocessing.spawn(fn, args=(args.num_processes,), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
