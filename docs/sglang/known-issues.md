# Known issues — SGLang integration

## LOAD-mode first-token corruption on the EP / dp-attention path (FIXED)

**Symptom.** On a foundry-LOADed EP server (Qwen3-30B-A3B, EP=2,
fa3 + dp-attention + DeepEP low-latency), some prompts deterministically get
a wrong *first* generated token at temperature 0 — e.g. `" zeroes"` before
`<think>` — after which generation continues coherently and correctly.
Different prompts get different (but per-prompt deterministic) stray tokens;
many prompts are unaffected. The very first request served by a fresh LOAD
process is clean; corruption appears on later requests as a function of the
per-rank request history. Logprobs show this is not a near-tie flip: the
stray token has logprob ≈ −0.0001 with the correct token ~9 nats down and a
garbage tail — the last-position prefill logits are confidently *wrong*, not
perturbed.

**Ruled out (all verified experimentally, 2026-09-01, H200 ×2, GPUs 0/2):**

- *Two-pass vs single-pass save*: byte-identical archives and byte-identical
  (mis)behavior. Single-pass save is sufficient for sglang — the SAVE path
  has no pass-conditional logic (warmup_state.json stores only the memory
  pool config + final alloc offset, both consumed by LOAD only).
- *Foundry-mode environment* (NCCL_CUMEM/NVLS pins, kernel_warmup
  suppression, capture machinery): a foundry SAVE-mode server answers the
  probe prompts byte-identically to a no-foundry baseline.
- *VMM layout divergence*: per-rank labeled alloc offsets and
  final_alloc_offset match SAVE exactly on LOAD (DP0=137308930048,
  DP1=123247525888 on both sides).
- *The restored decode graphs themselves*: decode continues coherently and
  correctly after the corrupted first token; a 6-request run of one prompt
  replays consistently.
- *The server startup warmup request*: with `--skip-server-warmup` the first
  request is clean, but the corruption then appears from the second request
  on — the warmup was merely the "prior request" in default runs.

**Established mechanism (partially pinned).** Some early forward on the
LOAD path (dp-attention idle/companion batches are the prime suspect)
consumes `attn_backend.forward_metadata` *without initializing it first*:
setting it to `None` after the load-time fa3 metadata pre-pass turns the
corruption into a CUDA illegal-memory-access on the first decode batch
(async, surfaces at `copy_done.synchronize()`), proving the leftover is
consumed. On the normal path the leftover is the pre-pass's last-bs decode
metadata, whose contents (built from initial buffer values, no warmup
forward) differ from what upstream capture would have left — wrong
attention state → confidently wrong last-position logits, entangled with
the other rank via the dp-attention gather.

The plain-DP (no dp-attention) flashinfer path shows a much milder analog
(token-trajectory divergence from baseline on 1/4 probe prompts, no stray
tokens) — not yet characterized under the same first-vs-later-request lens.

**Note.** This is not new to the sglang-0.5.12 hook port: the previous
integration also populated fa3 metadata post-load without the capture-time
warmup forwards. Earlier validations measured throughput and coherence,
not token-level parity, so this went unnoticed.

**Deep-dive round 2 (2026-09-01, instrumented; all sglang instrumentation
reverted afterwards).** The earlier metadata theory is REFUTED; the full
localization chain, each step measured:

- The serving rank's prefill runs with CORRECT attention metadata
  (max_seq_len_k == prompt length) and per-layer last-token hidden norms are
  IDENTICAL between a clean and a dirty request through all 48 layers.
- The dp-attention hidden gather is correct: right variant (SUM_LEN /
  all-reduce with pre-zeroed buffer), correct offsets (get_dp_local_info),
  and the all-reduce output norm is identical on both ranks and equal to the
  clean value. Logits-shard assembly offsets are correct too.
- The corruption materializes between the LM-head matmul and the sampled
  token: the logits processor already emits argmax=" zeroes" on dirty
  requests. Serving-rank asymmetry: dp_rank 0 requests are always clean;
  dp_rank 1 requests are corrupted (prompt-dependent visibility).
- NOT an execution-order race: persists under CUDA_LAUNCH_BLOCKING=1.
- NOT torch-cache aliasing alone: `torch.cuda.empty_cache()` after graph
  load does not fix it (kept in hooks as hygiene regardless).
- ADDRESS-DISPLACEMENT SENSITIVE (smoking gun): planting a 64 MB tensor at
  the exact region boundary (final_alloc_offset, 0x601cb2200000 on the dirty
  rank) makes ALL requests clean — and the canary itself records ZERO
  corrupted bytes. So nothing writes into that band; displacing runtime
  eager allocations off their default addresses removes the corruption.
  Conclusion: some component holds a STALE INIT-TIME POINTER to an address
  in the early eager zone; when the runtime allocator reuses that address
  for a live tensor (logits-path tensors by default), the stale reference
  corrupts it. One-sided (NVSHMEM/DeepEP) access explains the
  launch-blocking immunity and the sporadic illegal accesses.

**Prime suspect / exact-divergence hypothesis:** the DeepEP buffer address
differs between SAVE and LOAD. On SAVE it is created mid-warmup-pass
(interleaved with activation allocations); on LOAD, `bootstrap_deepep_buffer`
creates it in isolation → different VMM address for the same object, while
`preallocate_for_load_mode` jumps the cursor to final_alloc_offset and MASKS
the order divergence. Restored graphs and the peer rank's one-sided ops
reference the SAVE address; the runtime eager path uses the LOAD address.

**RESOLUTION (confirmed 2026-09-01).** The exact divergence: the DeepEP
buffer was created at different VMM addresses on SAVE (lazily, mid-warmup
pass, after rank-asymmetric JIT/activation allocations) vs LOAD
(bootstrap_deepep_buffer, in isolation). Cursor logs prove it: with the fix,
both modes create the buffer at 120468799488 -> 122398179328 on both ranks;
before the fix, LOAD's bootstrap landed at the 122398179328 point while
SAVE's warmup-lazy creation landed elsewhere (rank-dependent). One-sided
NVSHMEM traffic and graph-referenced addresses used the SAVE-time address
while the runtime used the LOAD-time one; whichever eager tensor later
reused the stale VA got corrupted (logits-path tensors by default).

**Fix** (hooks.py, no dummy forwards): run `bootstrap_deepep_buffer` BEFORE
the SAVE-side warmup pass, so the buffer is created at the same
allocation-sequence point in both modes and its address matches by
construction. Archives must be RE-SAVED (baked addresses change).
`torch.cuda.empty_cache()` after graph load was also added as allocator
hygiene. Validated: the repro sequence is fully clean, first token
`<think>` at logprob ~0.0, DP2 regression clean.

**Follow-up.** Add a token-parity check (fixed prompt set, temp 0,
logprobs) to the validation recipe so LOAD-vs-baseline divergence is caught
routinely.

Repro: `experimental/expert-parallel/serve_sglang_qwen_30b_a3b.sh 2 --load`,
then two chat requests, temp 0: `"What is 15 - 6? Answer briefly."` — the
second (or the first, if the server warmup ran) starts with `" zeroes"`.
