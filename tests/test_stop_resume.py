import os
import sys
import subprocess
from pathlib import Path

import pytest
import torch

BASE_ADDR = 0x7f0000000000
REGION_SIZE_STR = "1GB"


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


def _run_core():
    import foundry as fdry

    torch.cuda.init()
    device = torch.device('cuda:0')
    torch.set_default_device(device)

    region_size = fdry.parse_size(REGION_SIZE_STR)
    region_end = BASE_ADDR + region_size

    print("[TEST] Setting up allocation region")
    fdry.set_allocation_region(BASE_ADDR, region_size)

    print("[TEST] Allocation 1 (Inside Region)")
    t1 = torch.full((2*1024*1024,), 1.0, device=device)
    addr1 = t1.data_ptr()
    print(f"[TEST] t1 size {t1.element_size() * t1.numel() / (1024*1024)} MB, address: {hex(addr1)}")

    # Verify t1 is within allocation region
    assert addr1 >= BASE_ADDR, f"t1 address {hex(addr1)} is below region base {hex(BASE_ADDR)}"
    assert addr1 < region_end, f"t1 address {hex(addr1)} is above region end {hex(region_end)}"
    print(f"[TEST] ✓ t1 address is within region [{hex(BASE_ADDR)}, {hex(region_end)})")

    print("[TEST] Stopping allocation region")
    fdry.stop_allocation_region()

    # Create a brand new pool (separate from default)
    pool = torch.cuda.memory.MemPool()

    # Allocate in the *new* pool
    with torch.cuda.use_mem_pool(pool, device=device):
        print("[TEST] Allocation 2 (Stopped Region)")
        t2 = torch.full((2*1024*1024,), 2.0, device=device)
        addr2 = t2.data_ptr()
        print(f"[TEST] t2 size {t2.element_size() * t2.numel() / (1024*1024)} MB, address: {hex(addr2)}")

        # Verify t2 is outside allocation region
        is_outside = addr2 < BASE_ADDR or addr2 >= region_end
        assert is_outside, f"t2 address {hex(addr2)} should be outside region [{hex(BASE_ADDR)}, {hex(region_end)})"
        print(f"[TEST] ✓ t2 address is outside region [{hex(BASE_ADDR)}, {hex(region_end)})")

    print("[TEST] Resuming allocation region")
    fdry.resume_allocation_region()

    print("[TEST] Allocation 3 (Resumed Region)")
    t3 = torch.full((2*1024*1024,), 3.0, device=device)
    addr3 = t3.data_ptr()
    print(f"[TEST] t3 size {t3.element_size() * t3.numel() / (1024*1024)} MB, address: {hex(addr3)}")

    # Verify t3 is within allocation region
    assert addr3 >= BASE_ADDR, f"t3 address {hex(addr3)} is below region base {hex(BASE_ADDR)}"
    assert addr3 < region_end, f"t3 address {hex(addr3)} is above region end {hex(region_end)}"
    print(f"[TEST] ✓ t3 address is within region [{hex(BASE_ADDR)}, {hex(region_end)})")

    print("[TEST] test_stop_resume PASSED")
    return t1, t2, t3


def _spawn_with_preload(test_mode):
    so_path = _get_hook_so_path()
    env = os.environ.copy()
    if env.get('LD_PRELOAD'):
        env['LD_PRELOAD'] = f"{so_path}:{env['LD_PRELOAD']}"
    else:
        env['LD_PRELOAD'] = so_path

    cmd = [sys.executable, str(Path(__file__).resolve()), f'--{test_mode}']
    subprocess.check_call(cmd, env=env)


@pytest.mark.filterwarnings('ignore:TORCH_CUDA_ARCH_LIST is not set')
def test_stop_resume():
    """Test stopping and resuming allocation region"""
    _spawn_with_preload('core')


if __name__ == '__main__':
    if '--core' in sys.argv:
        _run_core()
    else:
        test_stop_resume()
