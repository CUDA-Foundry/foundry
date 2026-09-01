# Known issues — SGLang integration

## LOAD-mode first-token corruption on the EP / dp-attention path (OPEN)

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

**Next steps.**

1. Instrument `init_forward_metadata_out_graph` /
   `_apply_cuda_graph_metadata` on LOAD: log (forward_mode, bs,
   id(metadata), max_seq_len_k, cu_seqlens) per call; diff a dirty vs clean
   request to name the exact consumer that skips init.
2. Fix direction: make the LOAD pre-pass reproduce the *post-capture* state
   exactly — either run the same per-bs dummy forwards upstream capture
   runs (the SAVE-side EP warmup-pass machinery, adapted to not re-enter
   graph_capture on LOAD), or make the consumer initialize its metadata
   instead of reusing the leftover.
3. Add a token-parity check (fixed prompt set, temp 0, logprobs) to the
   validation recipe so LOAD-vs-baseline divergence is caught routinely.

Repro: `experimental/expert-parallel/serve_sglang_qwen_30b_a3b.sh 2 --load`,
then two chat requests, temp 0: `"What is 15 - 6? Answer briefly."` — the
second (or the first, if the server warmup ran) starts with `" zeroes"`.
