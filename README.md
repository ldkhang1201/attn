# B3 · Transformer Attention Mechanism

**[Full walkthrough → demo.ipynb](demo.ipynb)** — bottleneck analysis, kernel
design, timing methodology, metrics, and charts, with outputs.

Scaled dot-product attention implemented from scratch with numba CUDA and
optimised in three steps. Everything except attention (embedding, projections,
layer norms, FFN) stays in PyTorch on the CPU — only the profiled bottleneck
is replaced.

| Version | File | Key idea |
|---|---|---|
| `CpuPipeline` | `src/cpu_baseline.py` | PyTorch reference + timing baseline |
| `CpuNaivePipeline` | `src/cpu_naive.py` | bare Python triple loops — the naive baseline behind published "1000×" claims |
| `GpuV1` | `src/gpu_v1.py` | naive three-kernel attention, everything through global memory |
| `GpuV2` | `src/gpu_v2.py` | tiled QKᵀ in shared memory + single-pass online softmax |
| `GpuV3` | `src/gpu_v3.py` | FlashAttention-style fused kernel, never materialises the N×N matrix |

V1 is the straight port: correct and massively parallel, but every
intermediate — including the full N×N score matrix — round-trips through
global memory. V2 keeps the same three-kernel structure but fixes *reuse*:
16×16 shared-memory tiles cut redundant global loads, and an online softmax
collapses three passes over each row into one. V3 changes the algorithm
itself, fusing QKᵀ, softmax, and the weighted sum into a single kernel
(FlashAttention-1) so the N×N matrix V2 was economising never gets
materialised at all — trading a per-tile rescale/renormalise for O(N²)
memory traffic that no longer exists.

## Layout

- `src/abstract.py` — shared transformer block; subclasses implement `attention`
- `src/bench.py` — profiling and timing helpers
- `demo.ipynb` — report: bottleneck profile, correctness checks, benchmarks, charts
- `tests/test_correctness.py` — every version vs `torch` SDPA (GPU tests auto-skip without CUDA)

## Run

```sh
uv sync
uv run pytest
```

The notebook needs a CUDA GPU (e.g. Colab T4) for the V1–V3 benchmarks; the
CPU profile chart runs anywhere.
