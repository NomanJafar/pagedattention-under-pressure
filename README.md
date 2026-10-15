# pagedattention-under-pressure

Reproducing vLLM PagedAttention block management under memory pressure: fragmentation, preemption and eviction, measured and documented.

Project 9 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

Reproduce vLLM paging behaviour under memory pressure and document fragmentation, preemption and eviction with measurements.

## Why it matters

Understanding the scheduler beats memorizing flags.

## Status

Done. The full write-up with the annotated timelines is [REPORT.md](REPORT.md); raw data in [`results/kaggle_run1/`](results/kaggle_run1/).

## Results in one paragraph

With prefix caching off and a cache shrunk to 52k tokens, vLLM preempted 203 times in 8 bursts: KV usage pinned at 100%, throughput down 42% (1,223 → 714 tok/s), ITL p99 486 ms against a 62 ms median, 32 requests waiting, and a third of requests lost at a 120 s client timeout. With prefix caching on (the default), the same "pressure" configuration never filled the cache, because every request shared one prompt; identical prompts measure the prefix cache, not the block manager. Block size 32 cut the reported capacity by 23% on this stack without any measurable benefit. One client-side artifact (httpx's 100-connection default pool) is named in the report and fixed in tokenbench.

## What is in here

| Module | Purpose |
|---|---|
| `paged_pressure.timeline` | Preemption events from counter deltas, phase annotation (filling / full / preempting / recovering), the pressure summary (time at full cache, steady vs post-preemption throughput), server args |
| `notebooks/kaggle_pagedattention.ipynb` | Four configurations (healthy; pressure; pressure with block 32; pressure with prefix caching off), 0.5 s metrics sampling, annotated timelines, comparison table |

![timelines](results/kaggle_run1/timelines.png)

## Layout

```
CONCEPTS.md     the theory behind the project, written before the code
src/paged_pressure/   implementation
tests/          CPU-only tests, runnable on a laptop
notebooks/      GPU notebooks (Kaggle / Colab) that run the experiments and write results/
results/        measured numbers and plots
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU experiments run from `notebooks/`; each notebook clones this repo, installs it and writes its outputs to `results/`.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels), the README says so rather than papering over it.

## License

MIT
