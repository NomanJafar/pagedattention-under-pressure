# pagedattention-under-pressure

Reproducing vLLM PagedAttention block management under memory pressure: fragmentation, preemption and eviction, measured and documented.

Project 9 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

Reproduce vLLM paging behaviour under memory pressure and document fragmentation, preemption and eviction with measurements.

## Why it matters

Understanding the scheduler beats memorizing flags.

## Status

Scaffolded. Concept notes and the first implementation are in progress; results land in `results/` once the GPU runs are done.

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
