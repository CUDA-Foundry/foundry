import os
import sys
import subprocess
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn


BASE_ADDR = 0x2F0000000000
REGION_SIZE_STR = "1GB"
WORKSPACE_DIR = "test_workspace"
HOOK_WORKSPACE_DIR = "hook_workspace"


def _get_hook_so_path():
    import importlib.util
    spec = importlib.util.find_spec('foundry.ops')
    if not spec or not spec.origin:
        raise RuntimeError('foundry.ops not found; ensure setup.py develop/pip install completed')

    ops_so_path = Path(spec.origin).resolve()
    hook_so_path = ops_so_path.parent / 'libcuda_hook.so'

    if not hook_so_path.exists():
        raise RuntimeError(f'libcuda_hook.so not found at {hook_so_path}')

    return str(hook_so_path)


def init_cublas():
    _ = torch._C._cuda_getCurrentBlasHandle()


def _run_saving_run():
    import foundry as fdry

    print("[SAVING] Starting saving run")

    torch.cuda.init()
    device = torch.device('cuda:0')
    torch.set_default_device(device)

    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    region_size = fdry.parse_size(REGION_SIZE_STR)
    fdry.set_allocation_region(BASE_ADDR, region_size)
    print(f"[SAVING] Allocation region set: base=0x{BASE_ADDR:x}, size={REGION_SIZE_STR}")

    input_tensor_a = torch.full((128, 64), 2.0, device=device)
    input_tensor_b = torch.full((64, 32), 3.0, device=device)
    print(f"[SAVING] input_a address: 0x{input_tensor_a.data_ptr():x}")
    print(f"[SAVING] input_b address: 0x{input_tensor_b.data_ptr():x}")

    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        init_cublas()
    print("[SAVING] cuBLAS initialized")

    graph = fdry.CUDAGraph()
    with fdry.graph(graph, stream=stream):
        result = torch.matmul(input_tensor_a, input_tensor_b)
    print("[SAVING] Graph captured")

    graph_json = os.path.join(WORKSPACE_DIR, "graph.json")
    graph.save(graph_json, output_tensors=result)
    print(f"[SAVING] Graph saved to {graph_json}")

    graph.replay()
    torch.cuda.synchronize()

    expected = torch.matmul(input_tensor_a, input_tensor_b)
    expected = expected.cpu()
    result = result.cpu()
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("[SAVING] Output verification PASSED")

    fdry.stop_allocation_region()
    print("[SAVING] Saving run completed")


def _run_loading_run():
    import foundry as fdry

    print("[LOADING] Starting loading run")

    torch.cuda.init()
    device = torch.device('cuda:0')
    torch.set_default_device(device)

    if not os.path.exists(WORKSPACE_DIR):
        raise RuntimeError(f"Workspace directory {WORKSPACE_DIR} not found")

    fdry.load_cuda_modules_and_libraries(HOOK_WORKSPACE_DIR)
    print(f"[LOADING] CUDA modules loaded from {HOOK_WORKSPACE_DIR}")

    region_size = fdry.parse_size(REGION_SIZE_STR)
    fdry.set_allocation_region(BASE_ADDR, region_size)
    print(f"[LOADING] Allocation region set: base=0x{BASE_ADDR:x}, size={REGION_SIZE_STR}")

    input_tensor_a = torch.full((128, 64), 5.0, device=device)
    input_tensor_b = torch.full((64, 32), 3.0, device=device)
    print(f"[LOADING] input_a address: 0x{input_tensor_a.data_ptr():x}")
    print(f"[LOADING] input_b address: 0x{input_tensor_b.data_ptr():x}")

    fdry.graph.default_capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(fdry.graph.default_capture_stream):
        init_cublas()
    print("[LOADING] cuBLAS initialized")

    graph_json = os.path.join(WORKSPACE_DIR, "graph.json")
    graph, output_tensor = fdry.CUDAGraph.load(graph_json)
    print(f"[LOADING] Graph loaded from {graph_json}")

    graph.replay()
    torch.cuda.synchronize()
    print("[LOADING] Graph replayed")

    expected = torch.matmul(input_tensor_a, input_tensor_b)
    expected = expected.cpu()
    output_tensor = output_tensor.cpu()
    torch.testing.assert_close(output_tensor, expected, rtol=1e-5, atol=1e-5)
    print("[LOADING] Output verification PASSED")

    fdry.stop_allocation_region()
    print("[LOADING] Loading run completed")


def _cleanup_workspace():
    if os.path.exists(WORKSPACE_DIR):
        shutil.rmtree(WORKSPACE_DIR)
    if os.path.exists(HOOK_WORKSPACE_DIR):
        shutil.rmtree(HOOK_WORKSPACE_DIR)


def _spawn_with_preload(launch_mode):
    so_path = _get_hook_so_path()
    env = os.environ.copy()
    if env.get('LD_PRELOAD'):
        env['LD_PRELOAD'] = f"{so_path}:{env['LD_PRELOAD']}"
    else:
        env['LD_PRELOAD'] = so_path
    env["TORCH_CUBLASLT_UNIFIED_WORKSPACE"] = "1"

    cmd = [sys.executable, str(Path(__file__).resolve()), f'--{launch_mode}']
    subprocess.check_call(cmd, env=env)


@pytest.mark.filterwarnings('ignore:TORCH_CUDA_ARCH_LIST is not set')
def test_cublas_ws():
    _cleanup_workspace()
    print("[TEST] Running saving run (capture and save)")
    _spawn_with_preload('saving-run')
    print("[TEST] Running loading run (load and replay)")
    _spawn_with_preload('loading-run')
    _cleanup_workspace()
    print("[TEST] test_cublas_ws PASSED")


if __name__ == '__main__':
    if '--saving-run' in sys.argv:
        _run_saving_run()
    elif '--loading-run' in sys.argv:
        _run_loading_run()
    elif '--cleanup' in sys.argv:
        _cleanup_workspace()
    else:
        test_cublas_ws()
