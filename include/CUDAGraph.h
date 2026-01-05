#pragma once

#include <ATen/Tensor.h>
#include <boost/unordered/concurrent_flat_map_fwd.hpp>
#include <boost/unordered/concurrent_flat_map.hpp>
#include <boost/json.hpp>
#include <c10/core/Device.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/intrusive_ptr.h>
#include "metadata.h"

#include <vector>
#include <memory>
#include <atomic>
#include <variant>

namespace at {

struct Generator;
struct CUDAGeneratorImpl;
struct CUDAGeneratorState;

}

namespace foundry {

using MempoolId_t = c10::MempoolId_t;
using CaptureId_t = c10::CaptureId_t;

enum class OutputTensorType {
  None,
  Single,
  List,
  Tuple
};

using OutputTensors = std::variant<
    std::monostate,
    at::Tensor,
    std::vector<at::Tensor>
>;

struct CUDAGraph;

struct GraphLoadResult {
  std::shared_ptr<CUDAGraph> graph;
  OutputTensors outputs;
  OutputTensorType output_type;
};

MempoolId_t graph_pool_handle();

void preallocate_cublas_workspaces();

struct CUDAGeneratorStateRegistry {
  uint64_t query_state_id(at::CUDAGeneratorState* state);
  c10::intrusive_ptr<at::CUDAGeneratorState> get_state_from_id(uint64_t id, uint64_t seed);

 protected:
  std::atomic<uint64_t> id_counter{0};
  boost::concurrent_flat_map<at::CUDAGeneratorState*, uint64_t> id_map_;
  boost::concurrent_flat_map<uint64_t, c10::intrusive_ptr<at::CUDAGeneratorState>> state_pool_;
};

struct CUDAGraph {
  CUDAGraph(bool keep_graph = false);
  ~CUDAGraph();

  void register_generator_state(c10::intrusive_ptr<at::CUDAGeneratorState> state);
  void register_generator_state(const at::Generator& generator);
  void capture_begin(
      MempoolId_t pool = {0, 0},
      cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal);
  void capture_end();
  void instantiate();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);
  cudaGraph_t raw_cuda_graph();
  cudaGraphExec_t raw_cuda_graph_exec();

  void analyze_captured_graph();
  void save(const std::string& json_path,
            const OutputTensors& output_tensors = std::monostate{},
            OutputTensorType output_type = OutputTensorType::None);
  static GraphLoadResult load(const std::string& json_path, MempoolId_t pool = {0, 0});

 protected:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;

  bool has_graph_ = false;
  bool capture_ended_ = false;
  bool has_graph_exec_ = false;

  CaptureId_t capture_id_ = -1;

  MempoolId_t mempool_id_;

  at::cuda::CUDAStream capture_stream_;

  ska::flat_hash_map<c10::intrusive_ptr<at::CUDAGeneratorState>, uint64_t>
      captured_generator_states_;

  static constexpr c10::DeviceIndex UNDEFINED_DEVICE = -1;
  c10::DeviceIndex capture_dev_{UNDEFINED_DEVICE};

  bool keep_graph_ = false;

  std::vector<GraphNode> graph_nodes;
  std::vector<GraphDependency> graph_dependencies;

  boost::json::object allocator_events_;

  struct LoadedGraphResources {
    std::vector<CUevent> created_events;
  };
  std::unique_ptr<LoadedGraphResources> loaded_graph_resources_;
};

} // namespace foundry
