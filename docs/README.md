# AI Infra 学习执行文档

本目录把 [`refs/AI Infra 学习路径.mhtml`](../refs/AI%20Infra%20学习路径.mhtml) 中的学习路径整理为一个可执行、可验收、可复现的 20 周项目计划。

## 文档导航

- [20 周执行方案](./execution-plan.md)：范围、阶段依赖、逐周任务、交付物与验收标准。
- [Benchmark 与 Profiling 方案](./benchmark-and-profiling.md)：统一测量协议、指标定义、核心实验和报告模板。
- [每周复盘模板](./weekly-review-template.md)：用于记录完成情况、实测数据、问题和下一周决策。

## 项目目标

最终交付一个单 GPU、Decoder-only、FP16/BF16 的最小 LLM 推理系统，并形成从 Runtime 到 GPU Kernel 的闭环：

```text
Request
  ↓
Scheduler → Continuous Batching
  ↓
Block Manager → Paged KV Cache → Prefix Cache
  ↓
Model Executor
  ↓
PyTorch → Triton/CUDA Kernels
  ↓
Nsight Systems / Nsight Compute
  ↓
Latency / Throughput / Memory / Real Speedup
```

项目沿用仓库现有的两条实现线：

- `gpu-kernel-lab/`：CUDA/Triton、RMSNorm、Softmax、Attention、微基准与 Nsight Compute。
- `mini-llm/`：模型执行、KV Cache、Block Manager、Scheduler、Serving Benchmark 与 Nsight Systems。

两条实现线在第 16–20 周汇合：把自定义 Kernel 接回推理引擎，并以端到端数据判断优化是否有效。

## 固定范围

V1 只支持：

- 单 NVIDIA GPU；
- 一种 Llama/Qwen 类 Decoder-only 架构；
- 一种权重精度（FP16 或 BF16）；
- greedy decoding；
- KV Cache、Paged KV Cache、Continuous Batching；
- 基础 Prefix Cache；
- PyTorch、Triton 和 CUDA Kernel 对照；
- 可复现 Benchmark 与 Profiling 报告。

V1 明确不做：Tensor/Pipeline Parallel、Multi-node、MoE、FP8、量化、Speculative Decoding、CUDA Graph、PD Disaggregation、分布式 KV Cache。新增能力必须先完成当前周验收，且不得破坏基线可复现性。

## 执行约定

- 默认周期 20 周，每周 12–20 小时。
- 每周按“学习 20% → 实现 50% → 测试与测量 20% → 复盘 10%”分配时间。
- 所有优化必须遵循 `Baseline → Profile → Hypothesis → Change → Re-profile → Conclusion`。
- Correctness 未通过前不比较性能；没有工作负载、硬件和基线信息的 speedup 不进入结论。
- 每周结束复制一次[每周复盘模板](./weekly-review-template.md)，保存实测证据与未完成项。
- 进度以验收标准为准，不以“看完教程”或“代码能启动”为准；若未通过验收，下周前 20% 时间优先补齐。

## 最终交付物

1. 可运行的 `mini-llm/` 推理主链路。
2. PyTorch/Triton/CUDA 三类 Kernel 实现及 correctness tests。
3. Kernel 与 Serving Benchmark 一键执行入口。
4. Nsight Systems 端到端时间线和 Nsight Compute Kernel 分析。
5. 至少 7 张核心结果图及对应原始数据。
6. 架构说明、优化日志、复现实验步骤和边界说明。

