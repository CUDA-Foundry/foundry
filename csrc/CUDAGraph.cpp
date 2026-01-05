#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/Functions.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/driver_api.h>
#include "CUDAGraph.h"
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <cuda.h>
#include <fstream>
#include <boost/json.hpp>
#include <iomanip>
#include <sstream>
#include "hook.h"


namespace at {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

  __attribute__((visibility("hidden")))
  void CUDAGeneratorState::register_graph(cuda::CUDAGraph* graph) {
    at::cuda::assertNotCapturing(
        "Cannot register the state during capturing stage.");
  
    if (registered_graphs_.empty()) {
      auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
      seed_extragraph_ = at::empty({1}, options);
      offset_extragraph_ = at::empty({1}, options);
    }
  
    if (registered_graphs_.find(graph) == registered_graphs_.end()) {
      registered_graphs_.insert(graph);
    }
  }

  __attribute__((visibility("hidden")))
  void CUDAGeneratorState::unregister_graph(cuda::CUDAGraph* graph) {
    TORCH_CHECK(
        registered_graphs_.find(graph) != registered_graphs_.end(),
        "The graph should be registered to the state");
  
    registered_graphs_.erase(graph);
  
    if (registered_graphs_.empty()) {
      seed_extragraph_.reset();
      offset_extragraph_.reset();
    }
  }

  __attribute__((visibility("hidden")))
  void CUDAGeneratorState::increase(uint64_t increment) {
    increment = ((increment + 3) / 4) * 4;
    if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
      TORCH_CHECK(
          capturing_,
          "Attempt to increase offset for a CUDA generator not in capture mode.");
      TORCH_INTERNAL_ASSERT(
          offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
      TORCH_INTERNAL_ASSERT(
          offset_intragraph_ <= std::numeric_limits<uint32_t>::max() - increment,
          "Increment causes overflow in the offset value.");
      offset_intragraph_ += increment;
    } else {
      TORCH_CHECK(
          !capturing_,
          "Offset increment outside graph capture encountered unexpectedly.");
      TORCH_INTERNAL_ASSERT(
          philox_offset_per_thread_ % 4 == 0,
          "RNG offset must be a multiple of 4.");
      philox_offset_per_thread_ += increment;
    }
  }

  __attribute__((visibility("hidden")))
  void CUDAGeneratorState::capture_prologue() {
    capturing_ = true;
    offset_intragraph_ = 0;
    seed_extragraph_.fill_(int64_t(seed_));
    offset_extragraph_.fill_(int64_t(0));
  }

  __attribute__((visibility("hidden")))
  uint64_t CUDAGeneratorState::capture_epilogue() {
    capturing_ = false;
    return offset_intragraph_;
  }
  
  __attribute__((visibility("hidden")))
  void CUDAGeneratorState::replay_prologue(uint64_t wholegraph_increment) {
    at::cuda::assertNotCapturing(
        "Cannot prepare for replay during capturing stage.");
    if (wholegraph_increment) {
        seed_extragraph_.fill_(int64_t(seed_));
        offset_extragraph_.fill_(int64_t(philox_offset_per_thread_));
        increase(wholegraph_increment);
    }
  }

  __attribute__((visibility("hidden")))
  void CUDAGeneratorImpl::register_graph(cuda::CUDAGraph* graph) {
    auto graph_ = reinterpret_cast<foundry::CUDAGraph*>(graph);
    graph_->register_generator_state(state_);
    state_->register_graph(graph);
  }
  
  __attribute__((visibility("hidden")))
  void CUDAGeneratorImpl::unregister_graph(cuda::CUDAGraph* graph) {
    state_->unregister_graph(graph);
  }
  
#pragma GCC diagnostic pop
}

namespace foundry {

static bool _cuda_graphs_debug = false;

static CUDAGeneratorStateRegistry global_generator_state_registry;

uint64_t CUDAGeneratorStateRegistry::query_state_id(at::CUDAGeneratorState* state) {
  uint64_t result_id = 0;

  auto visit_fn = [&](const auto& value) {
    result_id = value.second;
  };

  if (!id_map_.cvisit(state, visit_fn)) {
    result_id = id_counter.fetch_add(1, std::memory_order_relaxed);
    id_map_.emplace(state, result_id);
  }

  return result_id;
}

c10::intrusive_ptr<at::CUDAGeneratorState> CUDAGeneratorStateRegistry::get_state_from_id(uint64_t id, uint64_t seed) {
  c10::intrusive_ptr<at::CUDAGeneratorState> result_state;

  auto visit_fn = [&](const auto& value) {
    result_state = value.second;
  };

  if (!state_pool_.cvisit(id, visit_fn)) {
    result_state = c10::make_intrusive<at::CUDAGeneratorState>(seed, 0, 0);
    state_pool_.emplace(id, result_state);
  }

  return result_state;
}

MempoolId_t graph_pool_handle() {
  return c10::cuda::MemPool::graph_pool_handle();
}

void preallocate_cublas_workspaces() {
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CHECK(cublas_handle != nullptr, "Failed to get cuBLAS handle");

  cublasLtHandle_t cublaslt_handle = at::cuda::getCurrentCUDABlasLtHandle();
  TORCH_CHECK(cublaslt_handle != nullptr, "Failed to get cuBLASLt handle");

  void* cublaslt_workspace = at::cuda::getCUDABlasLtWorkspace();
  TORCH_CHECK(cublaslt_workspace != nullptr, "Failed to allocate cuBLASLt workspace");
}

CUDAGraph::CUDAGraph(bool keep_graph)
  : capture_stream_(at::cuda::getCurrentCUDAStream()),
    keep_graph_(keep_graph) {
}

void CUDAGraph::register_generator_state(
    c10::intrusive_ptr<at::CUDAGeneratorState> state) {
  global_generator_state_registry.query_state_id(state.get());
  captured_generator_states_[std::move(state)] = 0;
}

void CUDAGraph::register_generator_state(const at::Generator& generator) {
  c10::intrusive_ptr<at::CUDAGeneratorImpl> cuda_gen =
      c10::dynamic_intrusive_pointer_cast<at::CUDAGeneratorImpl>(
          generator.getIntrusivePtr());
  cuda_gen->register_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));
}

void CUDAGraph::capture_begin(MempoolId_t pool, cudaStreamCaptureMode capture_mode) {
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  gen->register_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->capture_prologue();
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::cuda::current_device();

  if (pool.first != 0 || pool.second != 0) {
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    mempool_id_ = c10::cuda::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  c10::cuda::CUDACachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](cudaStream_t stream) {
      cudaStreamCaptureStatus status{};
      CaptureId_t stream_capture_id = 0;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
      return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive && stream_capture_id == capture_id_;
  });
  foundry::resume_allocation_region();
  foundry::start_hook_record();

  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));

  cudaStreamCaptureStatus status{};
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);
}

void CUDAGraph::capture_end() {
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  foundry::end_hook_record();
  foundry::stop_allocation_region();
  allocator_events_ = foundry::save_hook_events_to_json();
  foundry::clear_hook_events();

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
      TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  capture_ended_ = true;
  has_graph_ = true;

  analyze_captured_graph();

  if (!keep_graph_) {
    instantiate();
    if (!_cuda_graphs_debug) {
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    }
    has_graph_ = false;
  }
}

void CUDAGraph::instantiate() {
  TORCH_CHECK(capture_ended_, "capture_end() must have been called before calling instantiate");

  if (has_graph_exec_) {
    TORCH_CHECK(keep_graph_, "instantiate() is intended to be called by the user only when keep_graph=true");
    AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
  }
#if !defined(USE_ROCM) || ROCM_VERSION >= 60200
  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
#else
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#endif
#if !defined(USE_ROCM) || ROCM_VERSION >= 60200
  } else {
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
  }
#endif
  has_graph_exec_ = true;
}

void CUDAGraph::replay() {
  TORCH_CHECK(capture_ended_ || has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture or load.");

  if (!has_graph_exec_) {
    TORCH_INTERNAL_ASSERT(keep_graph_);
    instantiate();
  }

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void CUDAGraph::enable_debug_mode() {
  _cuda_graphs_debug = true;
}

void CUDAGraph::debug_dump(const std::string& debug_path) {
#if defined(CUDA_VERSION) || defined(USE_ROCM)
  if (_cuda_graphs_debug || keep_graph_) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      C10_CUDA_CHECK_WARN(cudaGraphDebugDotPrint(graph_, debug_path.c_str(), cudaGraphDebugDotFlagsVerbose));
      if (!keep_graph_) {
        AT_CUDA_CHECK(cudaGraphDestroy(graph_));
        has_graph_ = false;
      }
    }
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with [graph].enable_debug_mode()");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

cudaGraph_t CUDAGraph::raw_cuda_graph() {
  TORCH_CHECK(keep_graph_, "You cannot access the raw cudaGraph_t instance unless CUDAGraph was initialized with keep_graph=true");
  TORCH_CHECK(has_graph_, "You cannot access the raw cudaGraph_t instance until capture_end() has been called");
  return graph_;
}

cudaGraphExec_t CUDAGraph::raw_cuda_graph_exec() {
  TORCH_CHECK(
      has_graph_exec_,
      "You cannot access the raw cudaGraphExec_t instance until instantiate() has been called");
  return graph_exec_;
}

void CUDAGraph::reset() {
  if (capture_ended_) {
    c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
    capture_ended_ = false;
  }
  if (has_graph_) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

MempoolId_t CUDAGraph::pool() {
TORCH_CHECK(capture_ended_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

CUDAGraph::~CUDAGraph() {
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->unregister_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));
  }
  reset();

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
  if (capture_dev_ != UNDEFINED_DEVICE)
  {
    AT_CUDA_CHECK(cudaSetDevice(capture_dev_));
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif
}


void CUDAGraph::analyze_captured_graph() {
  TORCH_CHECK(has_graph_, "analyze_captured_graph() called before the graph is captured");

  graph_nodes.clear();
  graph_dependencies.clear();

  CUgraph cuGraph = reinterpret_cast<CUgraph>(graph_);

  size_t numNodes = 0;
  C10_CUDA_DRIVER_CHECK(cuGraphGetNodes(cuGraph, nullptr, &numNodes));

  TORCH_CHECK(numNodes > 0, "Graph contains no nodes");

  std::vector<CUgraphNode> nodes(numNodes);
  C10_CUDA_DRIVER_CHECK(cuGraphGetNodes(cuGraph, nodes.data(), &numNodes));

  for (size_t i = 0; i < numNodes; ++i) {
    CUgraphNodeType nodeType;
    C10_CUDA_DRIVER_CHECK(cuGraphNodeGetType(nodes[i], &nodeType));

    GraphNode graphNode;
    graphNode.index = i;
    graphNode.node = nodes[i];

    if (nodeType == CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS params;
      memset(&params, 0, sizeof(params));
      C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeGetParams(nodes[i], &params));

      KernelNodeMetadata metadata;
      metadata.blockDimX = params.blockDimX;
      metadata.blockDimY = params.blockDimY;
      metadata.blockDimZ = params.blockDimZ;
      metadata.gridDimX = params.gridDimX;
      metadata.gridDimY = params.gridDimY;
      metadata.gridDimZ = params.gridDimZ;
      metadata.func = params.func;
      metadata.kern = params.kern;
      metadata.ctx = params.ctx;
      metadata.sharedMemBytes = params.sharedMemBytes;

      metadata.num_params = 0;
      size_t offset, size;
      while (cuFuncGetParamInfo(params.func, metadata.num_params, &offset, &size) == CUDA_SUCCESS) {
        metadata.offset_and_sizes.emplace_back(offset, size);
        metadata.num_params++;
      }

      if (params.kernelParams) {
        size_t totalSize = 0;
        for (const auto& [offset, size] : metadata.offset_and_sizes) {
          totalSize += size;
        }

        if (totalSize > 0) {
          metadata.kernelParams = static_cast<void**>(malloc(metadata.num_params * sizeof(void*)));
          for (int j = 0; j < metadata.num_params; ++j) {
            metadata.kernelParams[j] = malloc(std::get<1>(metadata.offset_and_sizes[j]));
            memcpy(metadata.kernelParams[j], params.kernelParams[j], std::get<1>(metadata.offset_and_sizes[j]));
          }
        }
      }

      if (params.extra) {
        void** config = params.extra;
        size_t argBufferSize = 0;
        void* argBufferPtr = nullptr;

        int idx;
        for (idx = 0; config[idx] != CU_LAUNCH_PARAM_END; idx++) {
          if (config[idx] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            argBufferPtr = config[idx + 1];
            idx++;
          } else if (config[idx] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            size_t* sizePtr = static_cast<size_t*>(config[idx + 1]);
            argBufferSize = *sizePtr;
            idx++;
          }
        }
        int configItems = idx + 1;

        metadata.extraSize = configItems * sizeof(void*);
        metadata.extra = malloc(metadata.extraSize);
        void** newConfig = static_cast<void**>(metadata.extra);

        if (argBufferSize > 0) {
          metadata.argBufferSize = static_cast<size_t*>(malloc(sizeof(size_t)));
          *metadata.argBufferSize = argBufferSize;
          metadata.argBuffer = malloc(argBufferSize);
          TORCH_INTERNAL_ASSERT(argBufferPtr != nullptr);
          memcpy(metadata.argBuffer, argBufferPtr, argBufferSize);
        }

        for (int idx = 0; idx < configItems; idx++) {
          if (idx < configItems - 1 && config[idx] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            newConfig[idx] = config[idx];
            newConfig[idx + 1] = metadata.argBuffer;
            idx++;
          } else if (idx < configItems - 1 && config[idx] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            newConfig[idx] = config[idx];
            newConfig[idx + 1] = metadata.argBufferSize;
            idx++;
          } else if (idx == configItems - 1) {
            newConfig[idx] = CU_LAUNCH_PARAM_END;
          } else {
            newConfig[idx] = config[idx];
          }
        }
      }

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_MEMSET) {
      CUDA_MEMSET_NODE_PARAMS params;
      memset(&params, 0, sizeof(params));
      C10_CUDA_DRIVER_CHECK(cuGraphMemsetNodeGetParams(nodes[i], &params));

      MemsetNodeMetadata metadata;
      metadata.dst = params.dst;
      metadata.elementSize = params.elementSize;
      metadata.height = params.height;
      metadata.pitch = params.pitch;
      metadata.value = params.value;
      metadata.width = params.width;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_MEMCPY) {
      CUDA_MEMCPY3D params;
      memset(&params, 0, sizeof(params));
      C10_CUDA_DRIVER_CHECK(cuGraphMemcpyNodeGetParams(nodes[i], &params));

      MemcpyNodeMetadata metadata;
      metadata.Depth = params.Depth;
      metadata.Height = params.Height;
      metadata.WidthInBytes = params.WidthInBytes;
      metadata.dstArray = params.dstArray;
      metadata.dstDevice = params.dstDevice;
      metadata.dstHeight = params.dstHeight;
      metadata.dstHost = params.dstHost;
      metadata.dstLOD = params.dstLOD;
      metadata.dstMemoryType = params.dstMemoryType;
      metadata.dstPitch = params.dstPitch;
      metadata.dstXInBytes = params.dstXInBytes;
      metadata.dstY = params.dstY;
      metadata.dstZ = params.dstZ;
      metadata.reserved0 = params.reserved0;
      metadata.reserved1 = params.reserved1;
      metadata.srcArray = params.srcArray;
      metadata.srcDevice = params.srcDevice;
      metadata.srcHeight = params.srcHeight;
      metadata.srcHost = params.srcHost;
      metadata.srcLOD = params.srcLOD;
      metadata.srcMemoryType = params.srcMemoryType;
      metadata.srcPitch = params.srcPitch;
      metadata.srcXInBytes = params.srcXInBytes;
      metadata.srcY = params.srcY;
      metadata.srcZ = params.srcZ;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
      CUevent event;
      C10_CUDA_DRIVER_CHECK(cuGraphEventRecordNodeGetEvent(nodes[i], &event));

      EventRecordNodeMetadata metadata;
      metadata.event = event;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_WAIT_EVENT) {
      CUevent event;
      C10_CUDA_DRIVER_CHECK(cuGraphEventWaitNodeGetEvent(nodes[i], &event));

      EventWaitNodeMetadata metadata;
      metadata.event = event;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_EMPTY) {
      EmptyNodeMetadata metadata;

      graphNode.metadata = std::move(metadata);
    } else {
      TORCH_CHECK(false, "Graph contains unsupported node type!");
    }

    graph_nodes.push_back(std::move(graphNode));
  }
  
  size_t numEdges = 0;
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)  
  C10_CUDA_DRIVER_CHECK(cuGraphGetEdges(cuGraph, nullptr, nullptr, nullptr, &numEdges));
#else
  C10_CUDA_DRIVER_CHECK(cuGraphGetEdges_v2(cuGraph, nullptr, nullptr, nullptr, &numEdges));
#endif
  
  if (numEdges > 0) {
    std::vector<CUgraphNode> from_nodes(numEdges);
    std::vector<CUgraphNode> to_nodes(numEdges);
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)  
    C10_CUDA_DRIVER_CHECK(cuGraphGetEdges(cuGraph, from_nodes.data(), to_nodes.data(), nullptr, &numEdges));
#else
    C10_CUDA_DRIVER_CHECK(cuGraphGetEdges_v2(cuGraph, from_nodes.data(), to_nodes.data(), nullptr, &numEdges));
#endif
    std::unordered_map<CUgraphNode, int> node_to_index;
    for (size_t i = 0; i < nodes.size(); i++) {
      node_to_index[nodes[i]] = i;
    }
    
    for (size_t i = 0; i < numEdges; i++) {
      auto from_it = node_to_index.find(from_nodes[i]);
      auto to_it = node_to_index.find(to_nodes[i]);
      
      if (from_it != node_to_index.end() && to_it != node_to_index.end()) {
        GraphDependency dep;
        dep.from_index = from_it->second;
        dep.to_index = to_it->second;
        graph_dependencies.push_back(dep);
      }
    }
  }
}


static boost::json::object serialize_tensor_metadata(const at::Tensor& tensor) {
  namespace json = boost::json;
  json::object obj;

  obj["data_ptr"] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.data_ptr()));

  json::array sizes_arr;
  for (auto s : tensor.sizes()) {
    sizes_arr.push_back(static_cast<int64_t>(s));
  }
  obj["sizes"] = sizes_arr;

  json::array strides_arr;
  for (auto s : tensor.strides()) {
    strides_arr.push_back(static_cast<int64_t>(s));
  }
  obj["strides"] = strides_arr;

  obj["dtype"] = static_cast<int>(tensor.scalar_type());
  obj["device_type"] = static_cast<int>(tensor.device().type());
  obj["device_index"] = tensor.device().index();
  obj["requires_grad"] = tensor.requires_grad();
  obj["numel"] = static_cast<int64_t>(tensor.numel());
  obj["element_size"] = static_cast<int64_t>(tensor.element_size());

  return obj;
}

static at::Tensor reconstruct_tensor_from_metadata(const boost::json::object& obj) {
  uintptr_t data_ptr = static_cast<uintptr_t>(obj.at("data_ptr").to_number<uint64_t>());

  const auto& sizes_arr = obj.at("sizes").as_array();
  std::vector<int64_t> sizes;
  for (const auto& s : sizes_arr) {
    sizes.push_back(s.to_number<int64_t>());
  }

  const auto& strides_arr = obj.at("strides").as_array();
  std::vector<int64_t> strides;
  for (const auto& s : strides_arr) {
    strides.push_back(s.to_number<int64_t>());
  }

  auto dtype = static_cast<c10::ScalarType>(obj.at("dtype").to_number<int>());
  auto device_type = static_cast<c10::DeviceType>(obj.at("device_type").to_number<int>());
  auto device_index = static_cast<c10::DeviceIndex>(obj.at("device_index").to_number<int>());
  bool requires_grad = obj.at("requires_grad").as_bool();

  auto options = at::TensorOptions()
      .dtype(dtype)
      .device(c10::Device(device_type, device_index))
      .requires_grad(requires_grad);

  auto tensor = at::from_blob(
      reinterpret_cast<void*>(data_ptr),
      sizes,
      strides,
      options
  );

  return tensor;
}

void CUDAGraph::save(const std::string& json_path,
                     const OutputTensors& output_tensors,
                     OutputTensorType output_type) {
  TORCH_CHECK(!graph_nodes.empty(), "Graph hasn't been captured yet or has no nodes");

  set_pack_fatbins_on_exit(true);

  namespace json = boost::json;
  json::object root;
  json::array nodes_array;

  std::unordered_map<CUevent, int> event_to_id;
  int next_event_id = 0;

  for (const auto& graphNode : graph_nodes) {
    json::object node_obj;
    node_obj["id"] = graphNode.index;

    if (std::holds_alternative<KernelNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<KernelNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "KernelNode";

      json::object params;
      params["blockDimX"] = metadata.blockDimX;
      params["blockDimY"] = metadata.blockDimY;
      params["blockDimZ"] = metadata.blockDimZ;
      params["gridDimX"] = metadata.gridDimX;
      params["gridDimY"] = metadata.gridDimY;
      params["gridDimZ"] = metadata.gridDimZ;
      params["sharedMemBytes"] = metadata.sharedMemBytes;

      json::array kernel_params_array;
      if (metadata.kernelParams) {
        for (int i = 0; i < metadata.num_params; ++i) {
          TORCH_CHECK(metadata.kernelParams[i], "kernelParams[", i, "] is null");
          json::object param_obj;
          param_obj["index"] = i;
          param_obj["offset"] = std::get<0>(metadata.offset_and_sizes[i]);
          param_obj["size"] = std::get<1>(metadata.offset_and_sizes[i]);

          std::ostringstream hex_stream;
          const auto* bytes = static_cast<const unsigned char*>(metadata.kernelParams[i]);
          for (size_t j = 0; j < std::get<1>(metadata.offset_and_sizes[i]); ++j) {
            hex_stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[j]);
          }
          param_obj["value_hex"] = hex_stream.str();

          kernel_params_array.push_back(param_obj);
        }
      } else {
        for (int i = 0; i < metadata.num_params; ++i) {
          json::object param_obj;
          param_obj["index"] = i;
          param_obj["offset"] = std::get<0>(metadata.offset_and_sizes[i]);
          param_obj["size"] = std::get<1>(metadata.offset_and_sizes[i]);
          kernel_params_array.push_back(param_obj);
        }
      }
      params["kernelParams"] = kernel_params_array;

      if (metadata.extra) {
        void** config = static_cast<void**>(metadata.extra);
        json::array extra_array;
        size_t argBufferSize = 0;

        int idx = 0;
        while (config[idx] != CU_LAUNCH_PARAM_END) {
          if (config[idx] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            extra_array.push_back("CU_LAUNCH_PARAM_BUFFER_POINTER");
            extra_array.push_back("null");
            idx += 2;
          } else if (config[idx] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            extra_array.push_back("CU_LAUNCH_PARAM_BUFFER_SIZE");
            size_t* sizePtr = static_cast<size_t*>(config[idx + 1]);
            argBufferSize = *sizePtr;
            extra_array.push_back(static_cast<uint64_t>(argBufferSize));
            idx += 2;
          } else {
            extra_array.push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(config[idx])));
            idx++;
          }
        }
        extra_array.push_back("CU_LAUNCH_PARAM_END");

        params["extra"] = extra_array;

        if (metadata.argBuffer && argBufferSize > 0) {
          std::ostringstream hex_stream;
          const auto* bytes = static_cast<const unsigned char*>(metadata.argBuffer);
          for (size_t j = 0; j < argBufferSize; ++j) {
            hex_stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[j]);
          }
          params["extra_argBuffer_hex"] = hex_stream.str();
        } else {
          params["extra_argBuffer_hex"] = "";
        }
      } else {
        params["extra"] = json::array{};
        params["extra_argBuffer_hex"] = "";
      }

      CUkernel kern = metadata.kern;
      CUfunction func = metadata.func;
      std::string function_name;
      uint64_t binary_hash = 0;

      if (kern != nullptr) {
        const char* name = nullptr;
        if (cuKernelGetName(&name, kern) == CUDA_SUCCESS && name) {
          function_name = name;
        }

        CUlibrary lib = nullptr;
        if (cuKernelGetLibrary(&lib, kern) == CUDA_SUCCESS && lib != nullptr) {
          binary_hash = query_binary_hash(lib);
          mark_binary_used(binary_hash);
        }
      } else if (func != nullptr) {
        CUkernel temp_kern = reinterpret_cast<CUkernel>(func);
        CUlibrary lib = nullptr;
        if (cuKernelGetLibrary(&lib, temp_kern) == CUDA_SUCCESS && lib != nullptr) {
          const char* name = nullptr;
          if (cuKernelGetName(&name, temp_kern) == CUDA_SUCCESS && name) {
            function_name = name;
          }
          binary_hash = query_binary_hash(lib);
          mark_binary_used(binary_hash);
        } else {
          const char* name = nullptr;
          if (cuFuncGetName(&name, func) == CUDA_SUCCESS && name) {
            function_name = name;
          }

          CUmodule mod = nullptr;
          if (cuFuncGetModule(&mod, func) == CUDA_SUCCESS && mod != nullptr) {
            binary_hash = query_binary_hash(mod);
            mark_binary_used(binary_hash);
          }
        }
      }

      params["function_name"] = function_name;
      params["kernel_source_binary_hash"] = binary_hash;

      json::object func_attrs;
      func_attrs["max_dynamic_shared_size_bytes"] = static_cast<int>(metadata.sharedMemBytes);

      int preferred_carveout = 0;
      int cluster_scheduling = 0;
      if (kern != nullptr) {
        cuKernelGetAttribute(&preferred_carveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, kern, capture_dev_);
        cuKernelGetAttribute(&cluster_scheduling, CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, kern, capture_dev_);
      } else if (func != nullptr) {
        cuFuncGetAttribute(&preferred_carveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, func);
        cuFuncGetAttribute(&cluster_scheduling, CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, func);
      }
      func_attrs["preferred_shared_memory_carveout"] = preferred_carveout;
      func_attrs["cluster_scheduling_policy_preference"] = cluster_scheduling;
      params["func_attrs"] = func_attrs;
      // NOTE: We do not save CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,
      // CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,
      // and CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED because these are
      // kernel properties may not be changed at runtime.

      node_obj["params"] = params;

    } else if (std::holds_alternative<MemcpyNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<MemcpyNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "MemcpyNode";

      json::object params;
      params["Depth"] = metadata.Depth;
      params["Height"] = metadata.Height;
      params["WidthInBytes"] = metadata.WidthInBytes;

      TORCH_CHECK(metadata.dstArray == nullptr, "dstArray must be null");
      TORCH_CHECK(metadata.srcArray == nullptr, "srcArray must be null");
      TORCH_CHECK(metadata.dstHost == nullptr, "dstHost must be null");
      TORCH_CHECK(metadata.srcHost == nullptr, "srcHost must be null");
      TORCH_CHECK(metadata.reserved0 == nullptr, "reserved0 must be null");
      TORCH_CHECK(metadata.reserved1 == nullptr, "reserved1 must be null");

      params["dstDevice"] = static_cast<uint64_t>(metadata.dstDevice);
      params["dstHeight"] = metadata.dstHeight;
      params["dstLOD"] = metadata.dstLOD;
      params["dstMemoryType"] = static_cast<int>(metadata.dstMemoryType);
      params["dstPitch"] = metadata.dstPitch;
      params["dstXInBytes"] = metadata.dstXInBytes;
      params["dstY"] = metadata.dstY;
      params["dstZ"] = metadata.dstZ;

      params["srcDevice"] = static_cast<uint64_t>(metadata.srcDevice);
      params["srcHeight"] = metadata.srcHeight;
      params["srcLOD"] = metadata.srcLOD;
      params["srcMemoryType"] = static_cast<int>(metadata.srcMemoryType);
      params["srcPitch"] = metadata.srcPitch;
      params["srcXInBytes"] = metadata.srcXInBytes;
      params["srcY"] = metadata.srcY;
      params["srcZ"] = metadata.srcZ;

      node_obj["params"] = params;

    } else if (std::holds_alternative<MemsetNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<MemsetNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "MemsetNode";

      json::object params;
      params["dst"] = static_cast<uint64_t>(metadata.dst);
      params["elementSize"] = metadata.elementSize;
      params["height"] = metadata.height;
      params["pitch"] = metadata.pitch;
      params["value"] = metadata.value;
      params["width"] = metadata.width;

      node_obj["params"] = params;

    } else if (std::holds_alternative<EventRecordNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<EventRecordNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "EventRecordNode";

      if (event_to_id.find(metadata.event) == event_to_id.end()) {
        event_to_id[metadata.event] = next_event_id++;
      }

      json::object params;
      params["event_id"] = event_to_id[metadata.event];
      node_obj["params"] = params;

    } else if (std::holds_alternative<EventWaitNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<EventWaitNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "EventWaitNode";

      if (event_to_id.find(metadata.event) == event_to_id.end()) {
        event_to_id[metadata.event] = next_event_id++;
      }

      json::object params;
      params["event_id"] = event_to_id[metadata.event];
      node_obj["params"] = params;

    } else if (std::holds_alternative<EmptyNodeMetadata>(graphNode.metadata)) {
      node_obj["type"] = "EmptyNode";
      node_obj["params"] = json::object{};
    }

    nodes_array.push_back(node_obj);
  }

  root["nodes"] = nodes_array;

  json::array deps_array;
  for (const auto& dep : graph_dependencies) {
    json::object dep_obj;
    dep_obj["from"] = dep.from_index;
    dep_obj["to"] = dep.to_index;
    deps_array.push_back(dep_obj);
  }
  root["dependencies"] = deps_array;

  std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sorted_generators;
  for (const auto& [state, wholegraph_increment] : captured_generator_states_) {
    uint64_t state_id = global_generator_state_registry.query_state_id(state.get());
    sorted_generators.emplace_back(state_id, state->seed_, wholegraph_increment);
  }

  std::sort(sorted_generators.begin(), sorted_generators.end(),
            [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

  json::array generators_array;
  for (const auto& [state_id, seed, wholegraph_increment] : sorted_generators) {
    json::object gen_obj;
    gen_obj["id"] = state_id;
    gen_obj["seed"] = seed;
    gen_obj["wholegraph_increment"] = wholegraph_increment;
    generators_array.push_back(gen_obj);
  }
  root["generators"] = generators_array;

  root["allocator_events"] = allocator_events_;

  json::object output_tensors_obj;
  output_tensors_obj["type"] = static_cast<int>(output_type);

  json::array tensors_array;
  if (std::holds_alternative<at::Tensor>(output_tensors)) {
    tensors_array.push_back(serialize_tensor_metadata(std::get<at::Tensor>(output_tensors)));
  } else if (std::holds_alternative<std::vector<at::Tensor>>(output_tensors)) {
    for (const auto& t : std::get<std::vector<at::Tensor>>(output_tensors)) {
      tensors_array.push_back(serialize_tensor_metadata(t));
    }
  }
  output_tensors_obj["tensors"] = tensors_array;
  root["output_tensors"] = output_tensors_obj;

  std::ofstream file(json_path);
  TORCH_CHECK(file.is_open(), "Failed to open file for writing: ", json_path);
  file << json::serialize(root);
  file.close();
}

GraphLoadResult CUDAGraph::load(const std::string& json_path, MempoolId_t pool) {
  std::ifstream file(json_path);
  TORCH_CHECK(file.is_open(), "Failed to open file for reading: ", json_path);

  std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  namespace json = boost::json;
  json::value root_val = json::parse(json_str);
  const json::object& root = root_val.as_object();

  auto graph = std::make_shared<CUDAGraph>(true);
  graph->loaded_graph_resources_ = std::make_unique<LoadedGraphResources>();

  graph->capture_dev_ = c10::cuda::current_device();

  if (pool.first != 0 || pool.second != 0) {
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    graph->mempool_id_ = pool;
  } else {
    graph->mempool_id_ = c10::cuda::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(graph->mempool_id_.first > 0);
  }

  const json::array& generators_array = root.at("generators").as_array();
  for (const auto& gen_val : generators_array) {
    const json::object& gen_obj = gen_val.as_object();
    uint64_t state_id = gen_obj.at("id").to_number<uint64_t>();
    uint64_t seed = gen_obj.at("seed").to_number<uint64_t>();
    uint64_t wholegraph_increment = gen_obj.at("wholegraph_increment").to_number<uint64_t>();

    auto state = global_generator_state_registry.get_state_from_id(state_id, seed);
    state->register_graph(reinterpret_cast<at::cuda::CUDAGraph*>(graph.get()));
    graph->captured_generator_states_[state] = wholegraph_increment;
  }

  const json::object& allocator_events = root.at("allocator_events").as_object();
  foundry::replay_hook_events_from_json(allocator_events);

  CUgraph cuGraph;
  C10_CUDA_DRIVER_CHECK(cuGraphCreate(&cuGraph, 0));
  graph->graph_ = reinterpret_cast<cudaGraph_t>(cuGraph);
  graph->has_graph_ = true;

  const json::array& nodes_array = root.at("nodes").as_array();

  std::unordered_map<int, CUgraphNode> id_to_node;
  std::unordered_map<int, CUevent> event_id_to_event;

  std::vector<std::vector<std::vector<uint8_t>>> all_kernel_params;
  std::vector<std::vector<void*>> all_param_ptrs;
  std::vector<std::vector<void*>> all_extra_configs;
  std::vector<std::vector<uint8_t>> all_arg_buffers;
  std::vector<size_t> all_arg_buffer_sizes;

  CUcontext current_ctx;
  C10_CUDA_DRIVER_CHECK(cuCtxGetCurrent(&current_ctx));

  for (const auto& node_val : nodes_array) {
    const json::object& node_obj = node_val.as_object();
    int node_id = node_obj.at("id").to_number<int>();
    std::string node_type = node_obj.at("type").as_string().c_str();
    const json::object& params = node_obj.at("params").as_object();

    CUgraphNode cuNode = nullptr;

    if (node_type == "KernelNode") {
      CUDA_KERNEL_NODE_PARAMS node_params;
      memset(&node_params, 0, sizeof(node_params));

      node_params.blockDimX = params.at("blockDimX").to_number<unsigned int>();
      node_params.blockDimY = params.at("blockDimY").to_number<unsigned int>();
      node_params.blockDimZ = params.at("blockDimZ").to_number<unsigned int>();
      node_params.gridDimX = params.at("gridDimX").to_number<unsigned int>();
      node_params.gridDimY = params.at("gridDimY").to_number<unsigned int>();
      node_params.gridDimZ = params.at("gridDimZ").to_number<unsigned int>();
      node_params.sharedMemBytes = params.at("sharedMemBytes").to_number<unsigned int>();
      node_params.ctx = current_ctx;

      std::string function_name = params.at("function_name").as_string().c_str();
      uint64_t binary_hash = params.at("kernel_source_binary_hash").to_number<uint64_t>();

      auto func_handle_variant = query_function_handle(binary_hash, function_name);
      if (std::holds_alternative<CUkernel>(func_handle_variant)) {
        CUkernel kern = std::get<CUkernel>(func_handle_variant);
        node_params.kern = kern;
      } else {
        CUfunction func = std::get<CUfunction>(func_handle_variant);
        node_params.func = func;
      }

      if (params.contains("func_attrs")) {
        const json::object& func_attrs = params.at("func_attrs").as_object();
        int max_shared = func_attrs.at("max_dynamic_shared_size_bytes").to_number<int>();
        int preferred_carveout = func_attrs.at("preferred_shared_memory_carveout").to_number<int>();
        int cluster_scheduling = func_attrs.at("cluster_scheduling_policy_preference").to_number<int>();

        if (std::holds_alternative<CUkernel>(func_handle_variant)) {
          CUkernel kern = std::get<CUkernel>(func_handle_variant);
          if (max_shared > 0) {
            C10_CUDA_DRIVER_CHECK(cuKernelSetAttribute(
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, max_shared, kern, graph->capture_dev_));
          }
          if (preferred_carveout >= 0) {
            C10_CUDA_DRIVER_CHECK(cuKernelSetAttribute(
                CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, preferred_carveout, kern, graph->capture_dev_));
          }
          if (cluster_scheduling > 0) {
            C10_CUDA_DRIVER_CHECK(cuKernelSetAttribute(
                CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, cluster_scheduling, kern, graph->capture_dev_));
          }
        } else {
          CUfunction func = std::get<CUfunction>(func_handle_variant);
          if (max_shared > 0) {
            C10_CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, max_shared));
          }
          if (preferred_carveout >= 0) {
            C10_CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, preferred_carveout));
          }
          if (cluster_scheduling > 0) {
            C10_CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                func, CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, cluster_scheduling));
          }
        }
        // NOTE: We do not set CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,
        // CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,
        // and CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED because these are
        // kernel properties may not be changed at runtime.
      }

      const json::array& kernel_params_array = params.at("kernelParams").as_array();
      int num_params = kernel_params_array.size();

      bool has_kernel_params = false;
      if (num_params > 0) {
        const json::object& first_param = kernel_params_array[0].as_object();
        has_kernel_params = first_param.contains("value_hex");
      }

      if (has_kernel_params) {
        all_kernel_params.emplace_back(num_params);
        all_param_ptrs.emplace_back(num_params);
        auto& param_data = all_kernel_params.back();
        auto& param_ptrs = all_param_ptrs.back();

        for (size_t i = 0; i < kernel_params_array.size(); ++i) {
          const json::object& param_obj = kernel_params_array[i].as_object();
          size_t param_size = param_obj.at("size").to_number<size_t>();
          std::string value_hex = param_obj.at("value_hex").as_string().c_str();

          param_data[i].resize(param_size);
          for (size_t j = 0; j < param_size; ++j) {
            std::string byte_str = value_hex.substr(j * 2, 2);
            param_data[i][j] = std::stoul(byte_str, nullptr, 16);
          }
          param_ptrs[i] = param_data[i].data();
        }
        node_params.kernelParams = param_ptrs.data();
      }

      const json::array& extra_array = params.at("extra").as_array();

      if (!extra_array.empty()) {
        all_extra_configs.emplace_back();
        auto& extra_config = all_extra_configs.back();

        size_t arg_buffer_idx = all_arg_buffers.size();
        bool has_arg_buffer = false;

        for (size_t i = 0; i < extra_array.size(); ++i) {
          if (extra_array[i].is_string()) {
            std::string str_val = extra_array[i].as_string().c_str();
            if (str_val == "CU_LAUNCH_PARAM_BUFFER_POINTER") {
              extra_config.push_back(CU_LAUNCH_PARAM_BUFFER_POINTER);
            } else if (str_val == "CU_LAUNCH_PARAM_BUFFER_SIZE") {
              extra_config.push_back(CU_LAUNCH_PARAM_BUFFER_SIZE);
            } else if (str_val == "CU_LAUNCH_PARAM_END") {
              extra_config.push_back(CU_LAUNCH_PARAM_END);
            } else if (str_val == "null") {
              TORCH_CHECK(!has_arg_buffer, "Encountered multiple 'null' entries in extra field, but only one is expected for CU_LAUNCH_PARAM_BUFFER_POINTER");
              std::string argBuffer_hex = params.at("extra_argBuffer_hex").as_string().c_str();
              if (!argBuffer_hex.empty()) {
                all_arg_buffers.emplace_back();
                auto& arg_buffer = all_arg_buffers.back();
                arg_buffer.resize(argBuffer_hex.length() / 2);
                for (size_t j = 0; j < arg_buffer.size(); ++j) {
                  std::string byte_str = argBuffer_hex.substr(j * 2, 2);
                  arg_buffer[j] = std::stoul(byte_str, nullptr, 16);
                }
                has_arg_buffer = true;
                extra_config.push_back(all_arg_buffers[arg_buffer_idx].data());
              } else {
                extra_config.push_back(nullptr);
              }
            }
          } else if (extra_array[i].is_uint64() || extra_array[i].is_int64()) {
            uint64_t val = extra_array[i].to_number<uint64_t>();
            if (i > 0 && extra_config.back() == CU_LAUNCH_PARAM_BUFFER_SIZE) {
              all_arg_buffer_sizes.push_back(val);
              extra_config.push_back(&all_arg_buffer_sizes.back());
            } else {
              extra_config.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(val)));
            }
          }
        }

        node_params.extra = extra_config.data();
      }

      C10_CUDA_DRIVER_CHECK(cuGraphAddKernelNode(&cuNode, cuGraph, nullptr, 0, &node_params));
    } else if (node_type == "MemcpyNode") {
      CUDA_MEMCPY3D copy_params;
      memset(&copy_params, 0, sizeof(copy_params));

      copy_params.Depth = params.at("Depth").to_number<size_t>();
      copy_params.Height = params.at("Height").to_number<size_t>();
      copy_params.WidthInBytes = params.at("WidthInBytes").to_number<size_t>();
      copy_params.dstDevice = params.at("dstDevice").to_number<CUdeviceptr>();
      copy_params.dstHeight = params.at("dstHeight").to_number<size_t>();
      copy_params.dstLOD = params.at("dstLOD").to_number<size_t>();
      copy_params.dstMemoryType = static_cast<CUmemorytype>(params.at("dstMemoryType").to_number<int>());
      copy_params.dstPitch = params.at("dstPitch").to_number<size_t>();
      copy_params.dstXInBytes = params.at("dstXInBytes").to_number<size_t>();
      copy_params.dstY = params.at("dstY").to_number<size_t>();
      copy_params.dstZ = params.at("dstZ").to_number<size_t>();
      copy_params.srcDevice = params.at("srcDevice").to_number<CUdeviceptr>();
      copy_params.srcHeight = params.at("srcHeight").to_number<size_t>();
      copy_params.srcLOD = params.at("srcLOD").to_number<size_t>();
      copy_params.srcMemoryType = static_cast<CUmemorytype>(params.at("srcMemoryType").to_number<int>());
      copy_params.srcPitch = params.at("srcPitch").to_number<size_t>();
      copy_params.srcXInBytes = params.at("srcXInBytes").to_number<size_t>();
      copy_params.srcY = params.at("srcY").to_number<size_t>();
      copy_params.srcZ = params.at("srcZ").to_number<size_t>();

      C10_CUDA_DRIVER_CHECK(cuGraphAddMemcpyNode(&cuNode, cuGraph, nullptr, 0, &copy_params, current_ctx));
    } else if (node_type == "MemsetNode") {
      CUDA_MEMSET_NODE_PARAMS memset_params;
      memset(&memset_params, 0, sizeof(memset_params));

      memset_params.dst = params.at("dst").to_number<CUdeviceptr>();
      memset_params.elementSize = params.at("elementSize").to_number<unsigned int>();
      memset_params.height = params.at("height").to_number<size_t>();
      memset_params.pitch = params.at("pitch").to_number<size_t>();
      memset_params.value = params.at("value").to_number<unsigned int>();
      memset_params.width = params.at("width").to_number<size_t>();

      C10_CUDA_DRIVER_CHECK(cuGraphAddMemsetNode(&cuNode, cuGraph, nullptr, 0, &memset_params, current_ctx));
    } else if (node_type == "EventRecordNode") {
      int event_id = params.at("event_id").to_number<int>();

      CUevent event;
      if (event_id_to_event.find(event_id) == event_id_to_event.end()) {
        C10_CUDA_DRIVER_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        event_id_to_event[event_id] = event;
        graph->loaded_graph_resources_->created_events.push_back(event);
      } else {
        event = event_id_to_event[event_id];
      }

      C10_CUDA_DRIVER_CHECK(cuGraphAddEventRecordNode(&cuNode, cuGraph, nullptr, 0, event));

    } else if (node_type == "EventWaitNode") {
      int event_id = params.at("event_id").to_number<int>();

      CUevent event;
      if (event_id_to_event.find(event_id) == event_id_to_event.end()) {
        C10_CUDA_DRIVER_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        event_id_to_event[event_id] = event;
        graph->loaded_graph_resources_->created_events.push_back(event);
      } else {
        event = event_id_to_event[event_id];
      }

      C10_CUDA_DRIVER_CHECK(cuGraphAddEventWaitNode(&cuNode, cuGraph, nullptr, 0, event));

    } else if (node_type == "EmptyNode") {
      C10_CUDA_DRIVER_CHECK(cuGraphAddEmptyNode(&cuNode, cuGraph, nullptr, 0));
    }

    if (cuNode) {
      id_to_node[node_id] = cuNode;
    }
  }

  const json::array& deps_array = root.at("dependencies").as_array();
  if (!deps_array.empty()) {
    std::vector<CUgraphNode> from_nodes;
    std::vector<CUgraphNode> to_nodes;
    from_nodes.reserve(deps_array.size());
    to_nodes.reserve(deps_array.size());

    for (const auto& dep_val : deps_array) {
      const json::object& dep_obj = dep_val.as_object();
      int from_id = dep_obj.at("from").to_number<int>();
      int to_id = dep_obj.at("to").to_number<int>();

      from_nodes.push_back(id_to_node[from_id]);
      to_nodes.push_back(id_to_node[to_id]);
    }

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)  
    C10_CUDA_DRIVER_CHECK(cuGraphAddDependencies(cuGraph, from_nodes.data(), to_nodes.data(), nullptr, deps_array.size()));
#else
    C10_CUDA_DRIVER_CHECK(cuGraphAddDependencies_v2(cuGraph, from_nodes.data(), to_nodes.data(), nullptr, deps_array.size()));
#endif
  }

  graph->capture_ended_ = true;
  graph->instantiate();
  // NOTE(yongji): bypass destructor's release memory pool call
  graph->capture_ended_ = false;

  GraphLoadResult result;
  result.graph = graph;
  result.output_type = OutputTensorType::None;
  result.outputs = std::monostate{};

  if (root.contains("output_tensors")) {
    const json::object& output_tensors_obj = root.at("output_tensors").as_object();
    result.output_type = static_cast<OutputTensorType>(output_tensors_obj.at("type").to_number<int>());

    const json::array& tensors_array = output_tensors_obj.at("tensors").as_array();

    if (result.output_type == OutputTensorType::Single && tensors_array.size() == 1) {
      result.outputs = reconstruct_tensor_from_metadata(tensors_array[0].as_object());
    } else if (result.output_type == OutputTensorType::List || result.output_type == OutputTensorType::Tuple) {
      std::vector<at::Tensor> tensors;
      tensors.reserve(tensors_array.size());
      for (const auto& t_val : tensors_array) {
        tensors.push_back(reconstruct_tensor_from_metadata(t_val.as_object()));
      }
      result.outputs = std::move(tensors);
    }
  }

  return result;
}

} // namespace foundry
