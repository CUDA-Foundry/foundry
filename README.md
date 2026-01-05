# Foundry

Foundry provides APIs to capture, save, and reload workloads that requires CUDA graphs. It intercepts CUDA driver calls to achieve deterministic memory addresses and dump used module/library. The CUDA graph is saved into a JSON file along with a image of used modules/libraries. By reloading the saved graph, the program can skip the warmup process and start in a second.

## Requirements

- PyTorch 2.9.0+
- CUDA Driver 12.0+
- Boost 1.83.0+

If you are using a conda environment, you can install the requirements with the following command:
```bash
conda install -c conda-forge boost-cpp boost

pip3 install torch torchvision
```

## Installation

```bash
python setup.py develop
```

## Quick Start

### Graph Capture and Save

Foundry requires LD_PRELOAD to intercept CUDA driver calls. The graph capture and save must run in a subprocess with the hook library preloaded.

```python
import foundry as fdry
import torch

torch.cuda.init()
device = torch.device('cuda:0')
torch.set_default_device(device)

# Set up VMM allocation region for deterministic memory addresses
BASE_ADDR = 0x7f0000000000
region_size = fdry.parse_size('1GB')
fdry.set_allocation_region(BASE_ADDR, region_size)

# Allocate input tensors
input_a = torch.full((100, 100), 2.0, device=device)
input_b = torch.full((100, 100), 3.0, device=device)

# Warm up the model
model = MyModel()
model(input_a, input_b)
torch.cuda.synchronize()

# Capture CUDA graph
graph = fdry.CUDAGraph()
with fdry.graph(graph):
    result = model(input_a, input_b)

# Replay and verify
graph.replay()
torch.cuda.synchronize()

# Save graph with output tensors
graph.save('graph.json', output_tensors=result)

fdry.stop_allocation_region()
```

### Graph Load and Replay

Loading a saved graph also requires LD_PRELOAD and must use the same allocation region base address.

```python
import foundry as fdry
import torch

torch.cuda.init()
device = torch.device('cuda:0')
torch.set_default_device(device)

# Load CUDA modules and libraries from workspace
fdry.load_cuda_modules_and_libraries('hook_workspace')

# Set up the same allocation region as capture
BASE_ADDR = 0x7f0000000000
region_size = fdry.parse_size('1GB')
fdry.set_allocation_region(BASE_ADDR, region_size)

# Allocate input tensors (can have different values)
input_a = torch.full((100, 100), 5.0, device=device)
input_b = torch.full((100, 100), 3.0, device=device)

# Load and replay the graph
graph, output_tensor = fdry.CUDAGraph.load('graph.json')
graph.replay()
torch.cuda.synchronize()

# output_tensor now contains the result
fdry.stop_allocation_region()
```

### Memory Preallocation for Fast Graph Reload

The preallocation API physically allocates memory upfront, enabling subsequent allocations to use a fast path (pointer bump only, no VMM driver calls).

```python
import foundry as fdry

# With preallocation - allocations within 8GB use fast path
with fdry.allocation_region(0x500000000000, '16GB', prealloc_size='8GB'):
    graph, outputs = fdry.CUDAGraph.load('model.json')
    graph.replay()
```

| Function | Description |
|----------|-------------|
| `set_allocation_region(base, size)` | Set VMM allocation region for deterministic memory addresses |
| `stop_allocation_region()` | Stop the allocation region |
| `allocation_region(base, size, prealloc_size=None)` | Context manager to set up VMM allocation region with optional preallocation |
| `preallocate_region(size)` | Manually preallocate memory inside an allocation region |
| `free_preallocated_region()` | Free manually preallocated memory |
| `load_cuda_modules_and_libraries(workspace_dir)` | Load CUDA modules and libraries for graph loading |

## Testing

Run the test suite:

```bash
pytest tests/
```

## Documentation


## Environment Variables

### Build Configuration
- `FOUNDRY_DEBUG` - Enable debug logging

## Setting up clangd
```
conda install -c conda-forge libstdcxx-ng libgcc-ng
conda install -c conda-forge bear
bear -- python setup.py build_ext --inplace
```