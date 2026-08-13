# Repository Guidelines

## Project Structure & Module Organization

This repository follows two complementary learning tracks:

- `gpu-kernel-lab/` contains PyTorch, Triton, and CUDA kernel experiments. The current entry point is `00_pytorch_gpu.py`.
- `mini-llm/` is reserved for the decoder-only inference runtime, including model execution, KV-cache management, and scheduling.
- `docs/` contains the 20-week execution plan, profiling protocol, and weekly review template.
- `refs/` stores source learning material. Treat these files as references rather than editable project output.

As implementation grows, place tests beside each track in `gpu-kernel-lab/tests/` and `mini-llm/tests/`. Store reproducible raw measurements and plots under `benchmark-results/`, grouped by run or commit.

## Build, Test, and Development Commands

There is no build system or dependency manifest yet. Run commands from the repository root:

```bash
python gpu-kernel-lab/00_pytorch_gpu.py
python -m compileall gpu-kernel-lab mini-llm
python -m pytest gpu-kernel-lab/tests mini-llm/tests
```

The first command runs the current CUDA matrix-multiplication benchmark and requires CUDA-enabled PyTorch and an NVIDIA GPU. `compileall` provides a lightweight syntax check. Use the pytest command once test directories are introduced. Before GPU work, record `nvidia-smi`, `nvcc --version`, and relevant package versions as required by `docs/benchmark-and-profiling.md`.

## Coding Style & Naming Conventions

Use Python with four-space indentation and PEP 8 conventions. Prefer `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Keep reference implementations simple, then place optimized Triton/CUDA variants in clearly named modules such as `rmsnorm_triton.py`. Add type hints to reusable runtime APIs. Never report timings without warmup and explicit GPU synchronization.

## Testing Guidelines

Use pytest and name files `test_<component>.py`. Compare optimized kernels against a PyTorch reference with `torch.testing.assert_close`, covering multiple shapes, dtypes, boundary sizes, and error tolerances. Correctness must pass before benchmarking. Save workload metadata and raw results so performance claims can be reproduced.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, including `feat: warmup`. Prefer focused Conventional Commit-style messages such as `feat: add paged kv allocator` or `test: cover softmax boundary widths`. Pull requests should explain scope, link an issue when available, list verification commands, and include benchmark tables or profiler screenshots for performance changes. Document hardware, software versions, regressions, and known limitations.
