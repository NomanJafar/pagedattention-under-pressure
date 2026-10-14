# Concepts: what PagedAttention does when the cache runs out

`pagedattention-under-pressure` shrinks vLLM's KV cache on purpose, drives more load than fits, and records what the block manager does: how blocks are allocated and freed, when requests are preempted, what preemption costs, and how block size changes the picture. kvcalc sized the cache; this project is what happens past its edge.

## 1. Blocks and block tables

The KV cache is carved into fixed-size **blocks** of `block_size` tokens (16 by default). A request owns a **block table**: an ordered list of physical block ids, one per `block_size` tokens of its context. Blocks are allocated on demand as tokens are generated, not reserved up front, so a request that has produced 100 tokens holds 7 blocks, the last one partly full. The attention kernel follows the table, gathering keys and values from wherever the blocks happen to sit.

Two consequences:

- **Internal fragmentation is bounded by `block_size − 1` tokens per sequence.** With 16-token blocks and 64 sequences, at most ~1,000 tokens of the cache are reserved-but-empty: about 0.3% of a 340k-token cache. (Pre-paging systems reserved max-length contiguous regions per request and wasted 60–80%.)
- **There is no external fragmentation.** Any free block serves any request, because a block table does not need contiguous blocks. "Fragmentation" in the paged world means something else: the ragged tail block per sequence, and the inefficiency of large blocks (below).

## 2. Block size: waste vs kernel efficiency

Larger blocks mean more tail waste per sequence and coarser allocation, but fewer block-table entries and better attention-kernel efficiency (longer contiguous runs of keys per gather). vLLM's default of 16 is a compromise tuned for CUDA kernels; 32 trims the table and may speed attention slightly at the cost of twice the tail waste. On a cache of hundreds of thousands of tokens the waste is negligible either way; the kernel effect is what a comparison measures.

## 3. Memory pressure: what the scheduler does when blocks run out

Each iteration the scheduler needs one more block for every running sequence whose last block just filled. When the free list is empty it must evict someone. vLLM's V1 scheduler **preempts** the most recently admitted running request(s): their blocks are freed and the request goes back to the waiting queue at its head, keeping its generated tokens so far.

When the preempted request is readmitted it must get its KV back. Two policies:

- **Recompute** (vLLM V1's approach): re-run prefill over the prompt plus everything generated so far. Costs compute (a prefill of a now-longer sequence) but no memory to park anything. With prefix caching on, blocks that were cached for a common prefix may still be there, which shortens the recompute.
- **Swap** (older V0 option, `--swap-space` GiB of CPU RAM): copy the blocks to host memory and back over PCIe. Costs bandwidth and host RAM but no recompute. Removed in V1 as not worth the complexity on modern hardware.

Either way the preempted request's user sees a **stall**: their stream stops until the request is readmitted and re-prefilled. Everyone else sees throughput dip while the recompute runs. The counter `vllm:num_preemptions_total` is the server's confession that this is happening.

## 4. What pressure looks like in the metrics

A server under KV pressure shows a characteristic signature:

| Metric | Healthy | Under pressure |
|---|---|---|
| `kv_cache_usage_perc` | well below 1.0 | pinned at ~1.0 |
| `num_requests_running` | tracks demand | saw-tooth: drops when requests are preempted, climbs as they are readmitted |
| `num_requests_waiting` | 0 | > 0 even though `max-num-seqs` is not reached |
| `num_preemptions_total` | flat | climbing |
| output tokens/s | steady | oscillating, lower on average: prefill work is being repeated |
| per-request ITL p99 | tight | long stalls (seconds) for the preempted requests |

The experiment's job is to produce exactly that signature on demand and annotate the timelines so the cause of each feature is named.

## 5. How to create pressure on a 16 GB card

The default cache for a 1.5B model is 340k tokens; sixty-four 2k-token requests would use 38% of it, so default settings never show pressure. Two knobs make it appear:

- **`--gpu-memory-utilization` down** (say 0.35): the budget after weights shrinks from ~9 GiB to ~2 GiB, i.e. ~75k tokens, which 64 requests at 2k tokens overflow by 70%.
- **`--max-num-seqs` up** (say 256) so the sequence cap is not what stops admission; the memory must be.

Then an open-loop load with long outputs (so that running sequences keep growing and keep needing blocks) and enough concurrency drives the free list to zero within a minute.

## 6. What preemption costs, quantitatively

For a request with P prompt tokens and G generated tokens when preempted, recompute costs a prefill of P + G tokens: on a T4 with the 1.5B model, roughly (P + G) / 5,000 seconds of GPU time (prefill runs at a few thousand tokens per second here), during which all other decode steps are delayed or slowed. If preemptions happen k times per minute, the GPU spends k × (P+G)/5,000 seconds per minute redoing work. The experiment measures the throughput drop against the preemption count so this cost can be read off rather than estimated.

Prefix caching changes the sum: the prompt part of the recompute hits cached blocks if they survived, and only the generated tail is truly recomputed. Under pressure, cached blocks are the first to be evicted (they have no owner), so this help fades as pressure rises.

## 7. What the experiment will show

1. **Healthy baseline**: default cache, open-loop load, the metrics timeline; preemptions should stay at 0.
2. **Pressure**: same load, utilisation 0.35; timelines for usage, running, waiting, preemptions, tokens/s; annotated to show the saw-tooth and the throughput cost per preemption.
3. **Block size 16 vs 32** under pressure: KV capacity in tokens (vLLM prints it), preemption count, throughput.
4. **Prefix caching on vs off** under pressure: does sharing reduce preemptions for prompts with a common prefix?
5. A short report (`REPORT.md`) with the annotated timelines, for the reader who wants the scheduler's behaviour explained from evidence rather than from the flag list.

## Further reading

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023), sections 4 and 5 (block manager, preemption, recomputation vs swapping).
- vLLM docs: *Design: Automatic Prefix Caching* and the V1 scheduler description; *Engine Arguments* for `block-size`, `gpu-memory-utilization`, `max-num-seqs`.
- Operating systems texts on demand paging: the analogy holds surprisingly far.
