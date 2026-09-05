#pragma once

// Dependency insertion for rebuilt graphs (serial and parallel LOAD paths).

#include <cuda.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

namespace foundry {

// cuGraphAddDependencies_v2 documents one CUgraphEdgeData per edge, but libcuda
// 580.126.20 through 610.57.04 apply edge_data[0] to every edge of the call. A
// rebuilt graph normally leads with a default edge, so one bulk call silently
// turned every programmatic (PDL) edge into a full-completion barrier and cost
// 1-7% TPOT (docs/pdl-edge-batching.md). Insert one homogeneous batch per
// distinct edge record; each batch still passes a full array, so this is also
// correct on drivers that honour the documented contract.
inline CUresult add_graph_dependencies(CUgraph graph, const CUgraphNode* from,
                                       const CUgraphNode* to, const CUgraphEdgeData* edge_data,
                                       size_t count) {
  CUresult result = CUDA_SUCCESS;
  if (!edge_data || count < 2) {
    result = cuGraphAddDependencies_v2(graph, from, to, edge_data, count);
  } else {
    struct Batch {
      std::vector<CUgraphNode> from, to;
      std::vector<CUgraphEdgeData> data;
    };
    using Key = std::array<unsigned char, sizeof(CUgraphEdgeData)>;
    std::map<Key, Batch> batches;
    for (size_t i = 0; i < count; ++i) {
      Key key;
      std::memcpy(key.data(), &edge_data[i], sizeof(key));
      Batch& b = batches[key];
      b.from.push_back(from[i]);
      b.to.push_back(to[i]);
      b.data.push_back(edge_data[i]);
    }
    for (const auto& [key, b] : batches) {
      result = cuGraphAddDependencies_v2(graph, b.from.data(), b.to.data(), b.data.data(),
                                         b.from.size());
      if (result != CUDA_SUCCESS)
        break;
    }
  }

#ifdef FOUNDRY_DEBUG
  // Verify the driver kept the edge records: programmatic count in == count out.
  if (result == CUDA_SUCCESS && edge_data) {
    size_t expected = 0;
    for (size_t i = 0; i < count; ++i) {
      expected += edge_data[i].type == CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
    }
    size_t n = 0;
    cuGraphGetEdges_v2(graph, nullptr, nullptr, nullptr, &n);
    std::vector<CUgraphNode> f(n), t(n);
    std::vector<CUgraphEdgeData> e(n);
    if (n)
      cuGraphGetEdges_v2(graph, f.data(), t.data(), e.data(), &n);
    size_t actual = 0;
    for (size_t i = 0; i < n; ++i) {
      actual += e[i].type == CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
    }
    fprintf(stderr,
            "[foundry DEBUG] add_graph_dependencies: %zu edges, programmatic %zu in / %zu out%s\n",
            n, expected, actual, expected == actual ? "" : "  <-- EDGE DATA LOST");
  }
#endif
  return result;
}

}  // namespace foundry
