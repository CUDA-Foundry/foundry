# Restored-vs-captured graph slowdown — root causes & fixes (2026-07-27)

Report: restored (LOADed) graphs stably ~10% slower than captured ones.
Reproduced on Qwen3-1.7B, 1 GPU (H200): bs16 −19%, bs32 −12%, shrinking with
batch size (captured 4641/7187 tok/s vs restored 3761/6302 at bs16/32).

## Contributor 1 — PDL edges discarded (fixed)

FA3 decode on Hopper launches with Programmatic Dependent Launch: the
`prepare_varlen_num_blocks` → fwd → combine overlap is captured as
`CUgraphEdgeData` programmatic ports (28 edges/graph on Qwen3-1.7B:
`from_port=1, edge_type=1`). Foundry stored only `{from,to}` per edge and
rebuilt with NULL edge data → every PDL edge became a full-completion
dependency. Fixed end-to-end: capture keeps ports, JSON gains optional
`from_port/to_port/edge_type`, both rebuild sites pass reconstructed
`CUgraphEdgeData`, and `PROGRAMMATIC_STREAM_SERIALIZATION` is captured/
restored as a node attr (FA3 nodes carry none in practice — the edges are
the substance).

## Contributor 2 — shared-exec param switching (fixed)

THE dominant cost. On-demand (non-template) graphs shared one exec per
topology group; every batch-size change did `cudaDeviceSynchronize` (drains
the async decode pipeline) + ~374 µs param rewrite + `cuGraphExecUpdate` +
an unconditional stderr fprintf. One bench sweep hit **606 switches**
(~227 ms host cost + the sync stalls). Captured mode has one exec per size
and zero switch cost.

Fix: each on-demand graph lazily instantiates a **private exec** from the
shared template graph on first replay (host-side writes only — no sync) and
launches it directly thereafter. Memory matches captured mode (one exec per
graph). `FOUNDRY_SHARED_EXEC_REPLAY=1` restores the legacy low-memory
switching mode. Binary format bumped to v2 (BinDependency carries edge
data; v1 archives readable).

**Parity proof**: fixed-size bs=1 decode (512 tok, no size churn):
restored TPOT 2.03/2.04/2.04 ms vs captured 2.02/2.05/2.02 ms — identical.

## execUpdate penalty — MECHANISM ISOLATED (2026-07-31)

Design decision: the shared-template + cuGraphExecUpdate flow STAYS (the
one-step-ahead scheduler overlaps the update with the in-flight forward);
private per-graph execs were reverted.

Microbench (`tests/bench_exec_update.py`, 365-node linear graph, penalty
measured against a fresh exec of the SAME mutated graph, interleaved
medians, H200/CUDA 13):

- Whole-graph update with byte-identical params: **0%** — the driver diffs
  internally; rewriting unchanged nodes is free (foundry-side diffing buys
  nothing).
- Nodes whose params TRULY change are **permanently demoted** off the
  pre-baked launch path: ~+0.1 µs/node/replay, linear in dirty-node count
  (1/365 → +0.1%, 180/365 → +6.5%, 365/365 → +13%). Saturating, not
  cumulative across updates.
- **Nothing re-bakes**: cuGraphUpload after update → still +13%; later
  no-op update → still +13%; per-node cuGraphExecKernelNodeSetParams → +8%
  (same demotion class). Only fresh cuGraphInstantiate restores speed.

Production reading: pure graph forward time of a REBUILT graph is at parity
with captured (bs1 TPOT 2.03 vs 2.02 ms) when its exec is pristine — the
PDL fix closed the true rebuild defect. The residual ~6% appears after the
exec's first real switch and is the dirty-node tax (about half the decode
graph's nodes change grid/scalars across sizes; tensor pointers are shared).

Mitigation compatible with the design: **async background re-instantiate**
after a switch (~16-28 ms per graph off the critical path) — switch stays
instant via execUpdate; the exec is swapped for a pristine one when the
re-instantiate completes; steady state at any size converges to parity.
Memory stays one exec per template. Not yet implemented.

Docs corroboration (full sweep of CUDA PG/API refs + NVIDIA blogs + public
trackers): the penalty is UNDOCUMENTED and publicly unreported. NVIDIA's
model says an update dirties device work descriptors and the next launch
partially re-pays a one-time upload cost — steady-state replay should be
identical. No perf caveat exists for param/grid/function updates; upload is
described as mechanical. Our measurement (persistent per-replay GPU-timeline
cost, linear in dirty nodes, unaffected by cuGraphUpload) contradicts that
model → candidate driver bug report.
Related watch-item: MLX PR #2813 — cudaGraphExecUpdate mis-updates
thread-block-CLUSTER dimensions (correctness bug; MLX re-instantiates when
clusters are present). Foundry MoE archives carry clusterDim attrs, so the
update path may also be exposed there.

## Round 2 — re-verification + workaround search (2026-08-02)

Context: async re-instantiate REJECTED (user: instantiate cost unacceptable;
their tests show cuGraphInstantiate does not overlap graph forward, even
across different graphs). Re-checked everything on the box's current driver
and swept for alternatives. Driver/runtime unchanged per constraint.

### Finding re-verified on driver 595.58.03 / CUDA 13.1 (GPU 1, quiet)

- Sanity fresh-vs-fresh: −0.0%. Full original table reproduces byte-for-byte:
  noop 0%, args +13.2%, grid +12.5%, func-swap +13.2%, upload-after +13.2%,
  no-op-re-update +13.1%, per-node exec update +7.9%, fraction sweep linear
  (1/365 +0.1%, 36/365 +1.2%, 180/365 +6.5%).
- **Attribution: 100% GPU-timeline.** GPU event delta +38.3 µs/replay;
  CPU enqueue cost identical (2.15 µs/launch both). No CPU-side scheduling
  trick can hide it.
- **Node-count threshold (new):** all-dirty penalty is tiered by TOTAL exec
  node count: ≤128 nodes → 0 ns/node; 160–256 → ~15–19 ns/node; ≥320 →
  ~103–109 ns/node (same 102.9 ns/node at 365 and 1024). Production decode
  graphs (365+ nodes) sit in the full-penalty regime.
- **Kernel duration irrelevant:** with real ~29 µs kernels (runtime-bound
  spin loop; earlier attempts got constant-folded by ptxas) the absolute tax
  is unchanged (+38.5 µs/replay) — it's serial GPU front-end work per dirty
  node, never hidden behind kernel execution. Production impact ≈ dirty
  nodes × ~0.1 µs added to every decode step.
- Child-graph split (≤128-node chunks inside one exec): still +13.2% — the
  driver flattens children at instantiation; threshold is per-exec. Dead.
- cuGraphNodeSetEnabled off→on round-trip (params untouched): +13.2% —
  **demotion tracks touched nodes, not changed bytes**. Disabled nodes also
  cost ~+400 ns each at replay. Union-graph/enable-muxing designs dead.
- Mechanism color: libcuda strings show QMD (hardware work descriptor)
  chaining machinery (`CUDA_ENABLE_DYNAMIC_QMD_CHAINING_PREHOPPER`) —
  consistent with "dirty nodes fall off a pre-built QMD chain and take a
  slow re-dispatch path per replay". Exec device memory ≈ 2.5 KB/node
  (~896 KiB per 365-node exec) ≈ QMD array.

### Driver fix status (research, no install)

- Latest available as of Aug 2026: datacenter R595 = 595.71.05 (Apr 2026),
  Unix production = 595.84 (Jun 2026), New Feature Branch r610 = 610.43.03
  (Jul 2026, changelog literally "minor bug fixes and improvements"); CUDA
  13.2u2 + 13.3u1. **Zero graph-related items in any Fixed/Known Issues**
  across all of them; our 595.58.03 is superseded but nothing suggests a fix.
- Behavior remains undocumented; the "constant time launch" blog explicitly
  states update cost is a PARTIAL ONE-TIME repayment — NVIDIA's own
  published model contradicts our persistent measurement (bug-report ammo).
- No public report exists (verified again, adversarially). sglang merged an
  off-by-default shared-exec-update feature (2026-06-29) architecturally
  identical to foundry's — they will hit this too.
- User's non-overlap observation is NVIDIA-confirmed as expected behavior:
  the driver holds process-wide rwlocks; kernel launches take a read lock,
  heavyweight ops (instantiate) take the write lock and stall launch
  ENQUEUE; instantiate also blocks behind in-flight GPU work (forum threads
  347783/362467, R. Crovella). Our microbench: GPU-side replay throughput
  is unaffected by background plain instantiates (deep queue hides enqueue
  stalls), but instantiate+UPLOAD steals GPU cycles (+26%). With shallow
  one-step-ahead queues in production, the rwlock stall lands on the
  critical path — the user's constraint is durable.

### Workarounds measured/assessed (driver fixed, no runtime instantiate)

1. **Per-size pristine exec cache (RECOMMENDED).** Instantiate an exec per
   (hot) batch size at LOAD time from the same template — params set per
   size BEFORE its instantiate, so every exec is born pristine; template +
   execUpdate stays for tail/uncached sizes. Zero replay overhead, zero
   runtime instantiate, ~0.9 MB device mem per 365-node exec (~35 sizes ≈
   ~32 MB/topology), load cost ≈ instantiate × cached sizes (backgroundable).
   Prior art: sglang pre-dedup model, TRT-LLM per-size graphs + padding.
2. **Conditional SWITCH dispatcher (viable, elegant, more work).** One exec;
   SWITCH node with one body per batch size; tiny selector kernel reads the
   size index from device memory (host writes one int per switch — no exec
   update ever). Measured on H200/595.58.03: IF body +6.9 µs/replay,
   SWITCH-of-8 +7.2 µs (+2.4% on this microbench; ~+0.35% on a 2 ms decode
   step). The ~50 µs/conditional A30 forum report does NOT reproduce here.
   Caveats: bodies allow only kernel/empty/child/memset/memcpy/conditional
   nodes (no event/host/ext-sem nodes — must audit decode graphs), selector
   needs device-runtime linkage, save/load format work. Memory/instantiate
   cost same as (1).
3. Device-updatable kernel nodes: pristine cost only +2.3%, but the attr is
   capture-time-only (returned devNode handle per launch; our rebuild path
   uses cuGraphAddKernelNode → can't opt in), host cuGraphExecUpdate on such
   execs returns 801 NOT_SUPPORTED, single instantiation, updates only
   params/gridDim/enabled from device kernels. Dominated by (1).
4. Byte-identical-params indirection (llama.cpp PR #9017, OpenXLA
   NEVER_UPDATE VA-remap, PyGraph): exploits "noop updates are free", but
   requires kernels to take indirect params / fixed grids — vLLM kernel
   changes, out of foundry's scope. Note foundry's VMM already stabilizes
   POINTER args; grid dims + scalars are what stay dirty.
5. Child-split, enable-muxing: measured dead (above).

Bench: `foundry/tests/bench_exec_update.py --cases ...`.

## Round 3 — the complete demotion law + the piecewise fix (2026-08-08)

Refined by ordering/class experiments (scripts in scratchpad; GPU 0+1,
595.58.03). The full law:

1. **Ordering gate**: updates applied BETWEEN instantiate and the exec's
   first upload demote every byte-changed node (this is what all round-1/2
   demotion cases measured — and exactly what production hits: replay()
   updates the shared exec before its very first launch). After the exec is
   uploaded (one launch, or plain cuGraphUpload), **kernel-argument byte
   changes become FREE** (0.0% at 365 and 1024 nodes).
2. **Launch-config changes always demote**, even post-upload: gridDim
   (+12.7%), blockDim (+13.3%), sharedMemBytes (+13.7%), function (+13.7%).
   Per-node and linear (grid on 112/365 nodes → +4.2%, ~110 ns/node).
3. **Permanent, no recovery**: post-upload demotion also latches — restoring
   the uploaded baseline values, no-op updates, and re-upload all keep the
   penalty. Only a fresh instantiate produces fast nodes.
4. **Exec-size exemption is complete**: execs ≤128 nodes never demote — any
   class (args/grid/func), any ordering (pre/post-upload). 160 nodes →
   +2.1% (middle tier), ≥320 → full tier. Threshold is per-EXEC (child
   graphs are flattened and don't help; separate execs do).
5. Functional correctness verified: updates genuinely apply (accumulator
   readback: all-nodes bump doubles output; 1-node bump shifts by exactly
   one node's contribution).

Archive reality (Qwen3-1.7B FULL, 512 sizes, rank_0): between ANY two sizes
100% of nodes differ (pointer args with token-scaled offsets, num_token
scalars, packed cutlass argbufs, grids). Grid signatures are UNIQUE per size
(361/361 distinct) and 309/337 nodes take ≥2 grid values across the range —
so upload-after-instantiate alone saturates at ~= today's tax, and
grid-in-the-key ≡ per-size groups. blockDim/shmem/func/attrs do NOT vary
across sizes.

**The fix — piecewise execs (user's proposal, validated)**: split each
template into sequential pieces of ≤128 nodes (own CUgraph + shared exec per
piece), launched back-to-back on one stream. Measured: 3×122 pieces replay
at the same speed as the 366-node mono exec (boundary cost −3 µs/step ≈
free), and all-dirty updates on the pieces cost 0.0%. Every cross-piece
dependency is satisfied by stream ordering (capture emits nodes in
topological issue order; pieces serialize). Costs nothing in memory or load
time; keeps the template+execUpdate design unchanged. Caveats: cut points
should avoid splitting PDL triplets (cross-boundary PDL edges degrade to
full barriers); the 128 threshold is empirical on H200/r595 → make piece
size configurable (FOUNDRY_PIECE_NODES) and re-bench per platform (A100!).
NOT YET IMPLEMENTED — next step: piecewise template splitting in
CUDAGraphParallel.cpp + replay().

## Residual (superseded by the section above)

Post-fix sweep still showed ~8-12% at bs16/32 with the standard bench
(short 128-tok outputs, mixed sizes). Not per-graph replay speed (bs1 TPOT
parity). Leading hypotheses, unresolved because the final control run was
contaminated by another job landing on the GPU mid-measurement:

1. LOAD-mode prefill runs EAGER (compile disabled on LOAD) while the
   captured server prefills through inductor-compiled code — bench
   throughput includes prefill; short outputs magnify it. A prefill-diluted
   bs16/512-out A/B was inconclusive due to GPU contention — rerun on a
   quiet GPU.
2. First-touch private-exec instantiation absorbed during measured runs
   (once per size; run-2 numbers should be clean — partially observed).
Next instrument: nsys on both servers at fixed bs, diff kernel durations
vs inter-kernel gaps; and a per-phase (prefill vs decode) timing split.

Archives must be re-SAVEd to carry PDL edge data (old archives load fine at
old performance). Logs: logs/slowdown_*.log, results in
qwen_30ba3_{captured,restored_*}_1p7b_results/.
