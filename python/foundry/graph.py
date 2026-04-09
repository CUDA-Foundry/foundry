from __future__ import annotations

import gc
import json
import os
import torch
from typing import Optional, Union, List, Tuple, TYPE_CHECKING
from . import ops

if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE

OutputTensors = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]


class CUDAGraph(ops.CUDAGraph):
    def __new__(cls, keep_graph: bool = False):
        return super().__new__(cls, keep_graph)

    def capture_begin(
        self, pool: Optional[_POOL_HANDLE] = None, capture_error_mode: str = "global"
    ) -> None:
        super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)

    def capture_end(self) -> None:
        super().capture_end()

    def instantiate(self) -> None:
        super().instantiate()

    def replay(self) -> None:
        super().replay()

    def reset(self) -> None:
        super().reset()

    def pool(self) -> _POOL_HANDLE:
        return super().pool()

    def enable_debug_mode(self) -> None:
        return super().enable_debug_mode()

    def debug_dump(self, debug_path: str) -> None:
        return super().debug_dump(debug_path)

    def raw_cuda_graph(self) -> int:
        return super().raw_cuda_graph()

    def raw_cuda_graph_exec(self) -> int:
        return super().raw_cuda_graph_exec()

    def save(
        self,
        json_path: str,
        output_tensors: Optional[OutputTensors] = None
    ) -> None:
        return super().save(json_path, output_tensors)

    @staticmethod
    def load(
        json_path: str,
        pool: Optional[tuple[int, int]] = None,
    ) -> Union["CUDAGraph", Tuple["CUDAGraph", OutputTensors]]:
        if graph.default_capture_stream is None:
            graph.default_capture_stream = torch.cuda.Stream()

        torch.cuda.synchronize()

        stream_ctx = torch.cuda.stream(graph.default_capture_stream)
        with stream_ctx:
            cuda_graph, output_tensors = ops.CUDAGraph.load(json_path, pool)

        if output_tensors is None:
            return cuda_graph
        return cuda_graph, output_tensors

    @staticmethod
    def start_graph_builds(
        json_paths: List[str],
        pool: Optional[tuple[int, int]] = None,
        num_threads: int = 4,
    ):
        """Parse graph JSONs and start template building in background.

        Returns immediately with an opaque PendingGraphLoads handle.
        Template graphs are built on a background thread while the caller
        continues with subsequent initialization (KV cache, etc.).

        Call finish_graph_loads() later to wait for builds to complete
        and perform generator registration, allocator replay, and output
        tensor reconstruction.
        """
        if graph.default_capture_stream is None:
            graph.default_capture_stream = torch.cuda.Stream()

        torch.cuda.synchronize()

        stream_ctx = torch.cuda.stream(graph.default_capture_stream)
        with stream_ctx:
            return ops.CUDAGraph.start_graph_builds(json_paths, pool, num_threads)

    @staticmethod
    def finish_graph_loads(
        pending,
    ) -> List[Tuple["CUDAGraph", Optional[OutputTensors]]]:
        """Finish graph loads: replay allocator events and reconstruct output tensors.

        Takes the PendingGraphLoads handle from start_graph_builds().
        """
        return ops.CUDAGraph.finish_graph_loads(pending)


class graph:
    default_capture_stream: Optional[torch.cuda.Stream] = None

    def __init__(
        self,
        cuda_graph: CUDAGraph,
        pool: Optional[_POOL_HANDLE] = None,
        stream: Optional[torch.cuda.Stream] = None,
        capture_error_mode: str = "global",
    ):
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.cuda.Stream()

        self.pool = () if pool is None else (pool,)
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        assert self.capture_stream is not None
        self.stream_ctx = torch.cuda.stream(self.capture_stream)
        self.cuda_graph = cuda_graph
        self.capture_error_mode = capture_error_mode

    def __enter__(self) -> None:
        torch.cuda.synchronize()

        if torch.compiler.config.force_cudagraph_gc:
            gc.collect()

        torch.cuda.empty_cache()

        self.stream_ctx.__enter__()
        self.cuda_graph.capture_begin(
            *self.pool,
            capture_error_mode=self.capture_error_mode,
        )

    def __exit__(self, *args) -> None:
        self.cuda_graph.capture_end()
        self.stream_ctx.__exit__(*args)


def save_graph_manifest(archive_dir: str) -> None:
    """Write graph_manifest.json with topology groups and template assignments.

    Reads topology_key from each saved graph JSON, groups graphs by topology,
    picks the first graph in each group as the template, strips the
    dependencies array from on-demand (non-template) graphs to reduce file
    size, and writes the manifest.

    Should be called after all graphs are captured and saved.
    """
    graph_files = sorted(
        (f for f in os.listdir(archive_dir)
         if f.startswith("graph_") and f.endswith(".json")),
        key=lambda f: int(f.split("_")[1]),
    )
    if not graph_files:
        return

    # Read topology_key from each graph JSON
    topology_keys: dict[str, list[str]] = {}
    for filename in graph_files:
        json_path = os.path.join(archive_dir, filename)
        with open(json_path) as f:
            data = json.load(f)
        topo_key = data.get("topology_key", "")
        if not topo_key:
            return
        topology_keys.setdefault(topo_key, []).append(filename)

    # Build manifest
    groups = []
    templates = set()
    for topo_key, filenames in topology_keys.items():
        groups.append({
            "topology_key": topo_key,
            "template": filenames[0],
            "members": filenames,
        })
        templates.add(filenames[0])

    manifest = {"topology_groups": groups}
    manifest_path = os.path.join(archive_dir, "graph_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Strip dependencies from on-demand graph files.
    # On-demand graphs only need nodes + common_kernel_node_attrs for
    # prepare_on_demand_graph; dependencies are only used by
    # build_graph_from_parsed (template builds).
    num_stripped = 0
    for filename in graph_files:
        if filename in templates:
            continue
        json_path = os.path.join(archive_dir, filename)
        with open(json_path) as f:
            data = json.load(f)
        if "dependencies" in data:
            del data["dependencies"]
            with open(json_path, "w") as f:
                json.dump(data, f)
            num_stripped += 1

    num_templates = len(groups)
    num_on_demand = sum(len(g["members"]) - 1 for g in groups)
    import sys
    print(f"[foundry] Saved graph_manifest.json: {num_templates} topology groups "
          f"({num_templates} templates, {num_on_demand} on-demand, "
          f"{num_stripped} stripped)", file=sys.stderr)
