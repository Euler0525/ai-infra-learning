# AI Infra Learning

一个用于理解 Decoder-only 模型推理主链路的最小工程。仓库当前提供单请求 greedy 生成基线、纯 PyTorch Qwen 参考实现，以及 MHA/MQA 的 KV Cache 与显存带宽实验。

## 目录结构

```text
.
├── mini_llm/
│   ├── __init__.py
│   ├── config/                   # 模型、引擎与采样配置
│   ├── generate.py               # 单请求 greedy 生成入口
│   ├── preflight.py              # CUDA、BF16 与本地模型检查
│   ├── reference.py              # Hugging Face 正确性基线
│   ├── utils/                    # CUDA 同步计时
│   └── engine/
│       ├── prefill_decode.py     # Prefill、KV Cache 复用与单 token Decode
│       └── state.py              # Decode 状态、单步输出与生成结果
├── tests/                        # 配置和环境测试
├── mha_mqa_lab/                  # MHA/MQA 数学、KV Cache 与带宽实验
├── qwen2p5.py                    # 单文件的详细推理观察脚本
├── pyproject.toml
└── README.md
```

## 核心推理链路

`mini_llm.generate` 使用 Hugging Face `AutoModelForCausalLM` 加载本地 Qwen2.5 权重，并显式拆分推理过程：

```text
Prompt -> Chat Template -> Token IDs -> Prefill -> KV Cache
       -> 逐 token Decode -> greedy argmax -> EOS / max_new_tokens
```

当前阶段使用 Hugging Face 模型作为正确性基线；后续自定义模型、算子和 kernel 分别放入 `models/`、`layers/` 和 `kernels/`。

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

### Decoder Layer

整个 Transformer Decoder 会将一个 Decoder Layer 会堆叠多次，每一层结构如下

```plaintext
DecoderLayer
│
├── Attention Block
│   ├── RMSNorm              # 控制数据规模
│   ├── GQA Self-Attention   # 和其他 token 交流提取上下文信息
│   └── Residual Connection  # 保留原信息 + 新上下文信息
│
└── FFN Block
    ├── RMSNorm              # 再次稳定数值
    ├── SwiGLU MLP           # 每个 token 内部做非线性特征变换
    └── Residual Connection  # 保留旧信息 + 新特征
```

先让每个 token看一遍上下文，再让每个 token 自己做一次非线性加工，将结果传给下一层。数学表达式为

$$
\begin{aligned}
x' &= x + \mathrm{Attention}(\mathrm{RMSNorm}(x))\\
y  &= x' + \mathrm{MLP}(\mathrm{RMSNorm}(x'))
\end{aligned}
$$

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
python -m pip install -e ".[test]"
```

运行生成入口

```powershell
python -m mini_llm.generate --prompt "Explain KV cache briefly." --max-new-tokens 32
```

模型默认使用 CUDA、BF16 和 `local_files_only=True`，因此需要 NVIDIA GPU，且指定 revision 的模型权重必须已缓存在本地。运行环境检查：

```powershell
python -m mini_llm.preflight
```

若要观察 token、logits、Top 5 候选、KV Cache 和显存峰值，可运行：

```powershell
python qwen2p5.py
```

## MHA 与 MQA 实验

独立实验目录 `mha_mqa_lab/` 从数学、张量维度、KV Cache 和显存带宽角度比较 MHA 与 MQA。先运行维度与等价关系演示：

```powershell
python -m mha_mqa_lab.demo
```

在 CUDA GPU 上运行带显存保护的单层 decode 微基准：

```powershell
python -m mha_mqa_lab.benchmark
```

推导、论文数据、测量口径和参数说明见 [MHA/MQA 实验文档](mha_mqa_lab/README.md)。

## 参考资料

- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Euler0525@Blog | AI Infra](https://euler0525.github.io/blogs/series/AI-Infra/)
