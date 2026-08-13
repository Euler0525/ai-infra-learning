# AI Infra

## 目录结构

```shell
.
├── gpu-kernel-lab  # GPU Performance Engineering
├── mini-llm        # LLM Serving System
└── README.md
```

## 基础知识

一个 LLM serving 系统大概经历

```mermaid
graph LR
A(HTTP Request) --> B(Tokenizer) --> C(Request Queu) --> D(Scheduler) --> E(Batch)
--> F(Model) --> G(GPU Kernels) --> H(logits) --> I(Sampler) --> J(next token)
```

本项目研究以下内容

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

## 开发环境

参考博客 [AlamaLinux 安装流程](https://euler0525.github.io/blogs/posts/1dc8999e/)，包括 Linux, Pytorch, NVIDIA Driver CUDA Toolkit 等环境的配置。

- `nvidia-smi`：NVIDIA GPU 的管理工具，用来查看

```shell
GPU utilization
GPU memory
GPU temperature
GPU process
```

- `nvcc`：NVIDIA CUDA Compiler，即 CUDA 编译工具链
