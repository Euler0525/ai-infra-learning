# AI Infra Learning

一个用于理解 Decoder-only 模型推理主链路的最小工程。仓库当前提供单请求生成基线：使用 Hugging Face Qwen 模型，显式拆分 Prefill 和 Decode，复用 KV Cache，并记录两阶段的 GPU 执行时间。

## 目录结构

```text
.
├── mini_llm/
│   ├── __init__.py
│   ├── config.py                 # 模型、revision、设备与精度配置
│   ├── loading.py                # tokenizer、模型和 prompt 加载
│   ├── engine/
│   │   ├── generation.py         # Prefill、Decode 与生成循环
│   │   └── state.py              # KV 状态和生成结果数据结构
│   └── sampling/
│       └── greedy.py             # greedy token 选择
├── scripts/
│   └── generate.py               # 命令行生成入口
├── docs/                         # 独立项目文档
├── pyproject.toml
└── README.md
```

## 基础知识

```shell
────────────────────────────────────────

             AI Application
                   │
              HTTP Server
                   │
             LLM Engine
                   │
        ┌──────────┴───────────┐
        │                      │
    Scheduler             KV Cache
        │                      │
        └──────────┬───────────┘
                   │
              Model Runner
                   │
────────────────────────────────────────
               [Project]  # GPU 很贵，我怎样让它一直有有用的工作可做？
────────────────────────────────────────
                   │
                PyTorch
                   │
          Triton / CUDA Kernel
                   │
           GPU Hardware
────────────────────────────────────────
               [Project]  # 对于一次具体计算，我怎样让 GPU 更高效地执行？
────────────────────────────────────────
```

- 基础算子参考 [GitHub | Euler0525/leetgpu-challenges](https://github.com/Euler0525/leetgpu-challenges)

### 术语

- 预热 `warmup`：第一次执行可能包括 library initialization, kernel loading, JIT compilation, cache effects…这些不应该算在 steady-state performance 中，所以性能测试通常是 warmup -> benchmark
- 延迟 `latency`：完成一次任务要多久
- 吞吐量 `throughput`：单位时间完成的工作量

```python
import time
import torch


device = "cuda"

def matmul_time_test(n):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # warmup
    for _ in range(10):
        c = a @ b

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(100):
        c = a @ b

    torch.cuda.synchronize()
    end = time.perf_counter()

    print(f"n={n}, average:{(end - start) / 100}")

def main():
    matmul_time_test(1024)
    matmul_time_test(2048)
    matmul_time_test(4096)

if __name__ == "__main__":
    main()

"""
1. warmup 第一次执行可能包括
    - library initialization
    - kernel loading
    - JIT compilation
    - cache effects
上面这些不应该算在 steady-state performance中，所以性能测试通常是 warmup --> benchmark

2. GPU 操作通常是异步的，执行c = a @ b，不代表 GPU 此时已经算完了矩阵乘法，如果不加`torch.cuda.synchronize()`，测到的时间只是 launch kernel 的 CPU 时间，而不是 GPU 真正执行计算的时间.
"""
```

## 环境配置

参考 [Euler0525@Blog | AlamaLinux 安装流程](https://euler0525.github.io/blogs/posts/1dc8999e/)，包括 Linux, Pytorch, NVIDIA Driver CUDA Toolkit 等环境的配置。

- `nvidia-smi`：NVIDIA GPU 的管理工具，用来查看

```shell
GPU utilization
GPU memory
GPU temperature
GPU process
```

- `nvcc`：NVIDIA CUDA Compiler，即 CUDA 编译工具链

## 安装与运行

在仓库根目录安装项目及依赖

```powershell
python -m pip install -e .
```

运行生成入口

```powershell
python -m scripts.generate --prompt "Explain KV cache briefly." --max-new-tokens 32
```

## 参考资料

- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Euler0525@Blog | AI Infra](https://euler0525.github.io/blogs/series/AI-Infra/)
