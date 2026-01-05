from __future__ import annotations

import gc
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
