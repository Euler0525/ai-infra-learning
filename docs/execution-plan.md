# AI Infra 20 周详细执行方案

## 1. 计划基线

### 1.1 最终能力

完成本计划后，应能从代码和实测数据解释完整推理链路：

```text
tokenizer → request → scheduler → prefill/decode
          → KV Cache → block table → continuous batching
          → model executor → RMSNorm/Softmax/Attention kernels → GPU
```

面对“为什么更快”时，答案必须包含：工作负载、对照基线、瓶颈假设、Profiler 证据、变更内容、优化前后指标，以及退化区间。

### 1.2 前置条件

开始 Week 1 前完成一次环境盘点，并把结果记录到首份周报：

- Python、PyTorch、CUDA Toolkit、NVIDIA Driver 可用；
- `torch.cuda.is_available()` 为真；
- `nvidia-smi`、`nvcc --version` 可执行；
- 已记录 GPU 型号、显存、驱动、CUDA、PyTorch、Triton 版本；
- 现有 `gpu-kernel-lab/00_pytorch_gpu.py` 可运行并得到同步后的 matmul 延迟；
- 确定一个显存可容纳的小型 Llama/Qwen 类模型，V1 期间不更换架构。

若本机暂时没有 NVIDIA GPU，仍可推进模型与 Runtime 的 CPU correctness 工作，但 Week 4–10、18–20 的性能验收标记为阻塞，不得用 CPU 数据代替 GPU 结论。

### 1.3 建议目录

目录按需逐周创建，不要求第一天搭空架子：

```text
.
├── gpu-kernel-lab/
│   ├── common/                 # 测量、数据导出、环境记录
│   ├── cuda/                   # vector/reduction/RMSNorm/Softmax
│   ├── triton/                 # vector/RMSNorm/Softmax/Attention
│   ├── benchmarks/
│   └── tests/
├── mini-llm/
│   ├── model/                  # config/loader/layers/attention
│   ├── engine/                 # request/sequence/engine
│   ├── memory/                 # kv_cache/block/block_manager
│   ├── scheduler/              # scheduler/policy
│   ├── sampling/               # greedy
│   ├── server/                 # V1 可选的薄 API 层
│   ├── benchmarks/             # workload/load generator/plots
│   └── tests/
├── benchmark-results/          # 原始数据、环境元数据、图表
└── docs/
```

## 2. 阶段依赖与里程碑

| 阶段 | 周次 | 里程碑 | 进入下一阶段的硬条件 |
| --- | --- | --- | --- |
| M0 模型基础 | 1–3 | 最小 Decoder 可生成 token | reference logits/生成结果正确，prefill/decode 边界明确 |
| M1 GPU 基础 | 4–5 | PyTorch 基线与 CUDA 基础算子 | 测量无异步计时错误，CUDA vector/reduction 正确 |
| M2 Kernel 工程 | 6–10 | RMSNorm、Softmax、Attention 与 Nsight 报告 | 多 shape correctness 通过，性能结论有 profiler 证据 |
| M3 KV 内存 | 11–13 | KV Cache、Pool、Paged Block Manager | append/get/free 正确，无泄漏，能量化显存收益 |
| M4 调度系统 | 14–16 | Static/Continuous Batching 与内存预算联动 | 动态准入正确，可处理不同输出长度并回收 block |
| M5 Prefix 与评测 | 17–19 | Prefix Cache、负载生成器、端到端 profiling | 四类 serving 指标可复现，三组核心对照实验完成 |
| M6 集成交付 | 20 | 自定义 Kernel 接入 Runtime，文档齐全 | 一键复现、结果可追溯、已知限制明确 |

遇到硬条件未满足时，不把未验证实现继续堆到下一层。优先缩小 shape、batch 或功能范围，保留最小可验证闭环。

## 3. 每周固定节奏

每周建议执行四个工作块：

1. **概念与设计（2–4h）**：只学习本周实现直接需要的概念，写下关键公式、数据流和未知点。
2. **实现与测试（6–10h）**：先 reference，再优化版本；为关键状态转换补单元测试。
3. **测量与分析（2–4h）**：执行统一 Benchmark，保留原始数据、环境、命令和图。
4. **复盘与提交（1–2h）**：完成周报，列出证据、失败尝试、技术债和下一周入口条件。

每个工作块都应产生仓库内可追踪的产物。阅读笔记不能替代代码、测试或测量数据。

## 4. 逐周执行

### Week 1：建立 Transformer Inference 主链路

**目标**：理解 Decoder-only Transformer 从 token 到 next-token 的数据流，跑通最小 PyTorch 推理。

**学习重点**：Embedding、TransformerBlock、LM Head、logits、sampling；batch/sequence/context length；TTFT、ITL/TPOT、throughput、latency。

**任务**：

- 固定目标模型、dtype、device 和 greedy decoding。
- 画出模型一次 forward 的 tensor shape 流转，至少包含 RMSNorm、Q/K/V、RoPE、Attention、MLP。
- 加载配置和权重，完成 tokenizer → model → logits → next token 的最小路径。
- 为固定 prompt 保存 reference token IDs、首步 logits 摘要和生成 token，作为后续回归样本。

**产物**：`mini-llm/model/` 的最小实现、固定样例、shape 说明。

**验收**：

- 相同权重、prompt 和 greedy 策略下重复运行结果一致。
- 能解释每个主要 tensor 的 shape 和 dtype。
- 不调用高层 `model.generate()` 隐藏推理循环。

### Week 2：拆分 Prefill 与 Decode

**目标**：手写 inference loop，并明确 prefill 与 decode 的性能差异。

**学习重点**：prefill 一次处理多 token；decode 每轮生成一个 token并读取历史状态；同步与 GPU 异步执行。

**任务**：

- 设计显式的 `prefill(tokens)` 和 `decode(token, state)` 接口。
- 实现 greedy sampling 和停止条件。
- 记录每个阶段的 token 数、调用次数和耗时，暂不追求优化。
- 用 prompt length × output length 的小矩阵验证循环边界。

**产物**：`mini-llm/engine/`、`mini-llm/sampling/greedy.py`、prefill/decode 时序图。

**验收**：

- 不依赖 `model.generate()` 完成多 token 生成。
- prefill 只在请求进入时执行，decode 每 iteration 只追加一个 token。
- 生成结果与 Week 1 reference 一致。

### Week 3：手写最小 TransformerBlock

**目标**：让模型的核心层对后续 Kernel 替换保持透明。

**学习重点**：RMSNorm 公式、RoPE、causal mask、Attention、SwiGLU/MLP、残差连接。

**任务**：

- 先实现纯 PyTorch RMSNorm，并对多个 hidden size 做 correctness test。
- 实现 Q/K/V projection、RoPE、causal Attention、O projection 和 MLP。
- 逐层对比 reference model 的中间结果，定位误差来源。
- 把可替换算子接口稳定为 PyTorch reference 实现。

**产物**：`mini-llm/model/layers.py`、`attention.py`、层级回归测试。

**验收**：

- 固定输入下关键中间 tensor 与 reference 在约定 `atol/rtol` 内一致。
- 完整 logits 和 greedy 输出回归通过。
- 测试覆盖至少 batch=1 的多种 sequence length。

### Week 4：GPU 架构与 PyTorch 基线

**目标**：建立 GPU 性能世界观，并得到可信基线。

**学习重点**：SM、warp、thread/block/grid；register/shared memory/L1/L2/HBM；coalescing、occupancy、divergence；memory-bound 与 compute-bound。

**任务**：

- 修订并复用 `gpu-kernel-lab/00_pytorch_gpu.py` 的测量方法：warmup、同步、repeat、统计量。
- 对 RMSNorm、Softmax、Attention reference 做基线测量。
- 用 Nsight Systems 或 PyTorch Profiler 观察 kernel launch 和 CPU/GPU 时间线。
- 记录测量环境与原始 CSV/JSON，不只保留终端截图。

**产物**：环境清单、PyTorch baseline 数据、首份 profiling 观察。

**验收**：

- 计时覆盖 GPU 实际完成时间，而非仅 CPU launch 时间。
- 重复测量误差可解释；报告 median 及离散程度。
- 能为三个算子分别提出初始的 bandwidth/compute/launch-overhead 假设。

### Week 5：CUDA Vector 与 Reduction

**目标**：掌握 CUDA execution model 和基础 reduction。

**学习重点**：`threadIdx`、`blockIdx`、`blockDim`、grid 配置、边界检查、内存合并访问、同步。

**任务**：

- 依次实现 vector add、ReLU、reduce sum。
- 先实现正确的 naive 版本，再优化访问与 block 配置。
- 与 PyTorch reference 比较多种尺寸和非整除尺寸。
- 对 kernel launch 参数和数据布局写清设计理由。

**产物**：`gpu-kernel-lab/cuda/` 基础算子、测试和基准。

**验收**：

- 正常尺寸、边界尺寸和随机输入全部通过。
- 无越界访问，必要时用 Compute Sanitizer 检查。
- 能解释一个元素如何映射到 thread/block/grid。

### Week 6：Triton 与 RMSNorm

**目标**：完成第一个 PyTorch/Triton/CUDA 三版本算子。

**学习重点**：Triton program id、block load/store、mask、reduction、kernel fusion。

**任务**：

- 实现 Triton vector add，建立编译和测试入口。
- 实现 `rmsnorm_torch()`、`rmsnorm_triton()`、`rmsnorm_cuda()`。
- 测试 `(batch × seq, hidden)` 的典型和边界 shape。
- 扫描 hidden=512/1024/2048/4096/8192/16384，保留退化区间。

**产物**：RMSNorm 三版本、correctness matrix、延迟原始数据与曲线。

**验收**：

- 所有约定 shape 在指定 `atol/rtol` 内正确。
- 报告包含硬件、dtype、shape、warmup、repeat 和统计量。
- 不把单个 shape 的最好结果概括为普遍 speedup。

### Week 7：稳定 Softmax 与 Reduction 优化

**目标**：理解数值稳定、warp/shared-memory reduction 和融合收益。

**学习重点**：max reduction、exp、sum reduction、稳定 Softmax、row width 对实现的影响。

**任务**：

- 实现 PyTorch、naive Triton、fused Triton、CUDA Softmax。
- 加入大正数、大负数、极差输入与非 2 次幂宽度测试。
- 扫描 row count 和 row width，分析 launch overhead、occupancy 和 bandwidth。
- 对性能退化的 shape 保留结果并给出假设。

**产物**：Softmax 实现、数值稳定测试、latency vs row width 图。

**验收**：

- 输出无意外 NaN/Inf，概率和符合误差要求。
- correctness 与 benchmark 使用同一组 shape 定义。
- 能解释 fusion 减少了哪些 global memory round trip。

### Week 8：Naive Attention 基线

**目标**：先暴露完整 Attention matrix 的显存和 IO 成本。

**学习重点**：`QKᵀ / sqrt(d)`、causal mask、Softmax、`PV`；`N×N` 中间矩阵。

**任务**：

- 实现 PyTorch reference 与 naive 自定义 Attention。
- 支持 V1 所需的 causal mask、head 维度和 dtype。
- 扫描 sequence length，记录延迟、峰值显存和中间 tensor 大小。
- 写出 naive 方案的数据流和瓶颈假设。

**产物**：Attention baseline、correctness tests、显存/延迟基线。

**验收**：

- 短序列下与 reference 一致。
- 能通过公式和实测说明 `N×N` materialization 的增长趋势。
- 在开始 tiled 版本前锁定 baseline 数据。

### Week 9：Tiled Attention 与 Online Softmax

**目标**：不 materialize 完整 Attention matrix，理解 FlashAttention 的 IO-aware 思路。

**学习重点**：Q/K/V tiling、online max/sum 更新、partial output、片上存储与 HBM traffic。

**任务**：

- 先独立写出 online softmax 公式和 CPU/PyTorch 验证版本。
- 实现 tiled Triton Attention；CUDA 版本可作为扩展，不作为硬门槛。
- 与 naive/reference 版本比较 correctness、latency 和 peak memory。
- 完成后再与 Triton 官方 fused Attention 教程对照，记录差异而非直接替换。

**产物**：tiled Attention、推导笔记、latency/memory 对照数据。

**验收**：

- 多种 sequence/head dimension 下结果正确。
- 长序列不再分配完整 `N×N` Attention matrix。
- 能解释 online softmax 如何保持数值稳定。

### Week 10：Kernel Profiling 闭环

**目标**：用 Nsight Systems 和 Nsight Compute 为优化结论提供证据。

**学习重点**：Systems 看时间线和 overlap；Compute 看 occupancy、register pressure、warp stalls、L1/L2/DRAM 和吞吐。

**任务**：

- 给 benchmark 加 NVTX 标记，区分 RMSNorm、Softmax、Attention。
- 对每个算子选一个典型 shape 和一个退化 shape。
- 按 `Baseline → Profile → Hypothesis → Change → Profile again` 完成至少一个真实优化闭环。
- 把优化前后 kernel 数、latency、DRAM 相关指标和结论写入优化日志。

**产物**：Nsight 报告、Kernel 优化日志、可重跑命令。

**验收**：

- Systems 与 Compute 的职责没有混用。
- 至少一个 speedup 可由指标变化解释。
- 也记录无收益或退化的尝试及原因。

### Week 11：连续 KV Cache

**目标**：避免 decode 重算历史 K/V，并量化收益和显存成本。

**学习重点**：每层 K/V 的 shape、append/read 生命周期、每 token KV 显存公式。

**任务**：

- 实现 `KVCache.append/get/reset` 和 decode 路径。
- 比较 No KV Cache 与 KV Cache 在 context=128/256/512/1024/2048/4096 下的 decode latency。
- 用 `2 × layers × kv_heads × head_dim × bytes` 计算每 token 显存，并与实测分配对照。
- 测试多层、多请求隔离、容量边界和 reset。

**产物**：`mini-llm/memory/kv_cache.py`、延迟曲线、显存计算说明。

**验收**：

- cached decode 与未缓存 reference 输出一致。
- 每轮只计算并追加新 token 的 K/V。
- 公式估算与实测差异有明确解释。

### Week 12：KV Pool 与分配器基线

**目标**：实现可观测的连续/预分配 KV 管理，为 Paged 版本建立反例。

**学习重点**：内部碎片、容量预算、请求生命周期、OOM 点。

**任务**：

- 实现按最大长度预分配或连续区间分配的 V0 allocator。
- 记录 allocated、used、wasted bytes 和活跃请求数。
- 构造长短请求混合 workload，观察浪费与最大并发。
- 明确 allocate/append/free 的状态机和不变量。

**产物**：KV Pool V0、分配状态测试、内存浪费基线。

**验收**：

- 请求结束后容量可回收，无跨请求读写。
- 能用数据展示“为 4096 token 预留、实际只用 200 token”的浪费。
- 形成 Paged KV 的对照 workload。

### Week 13：Paged KV 与 Block Manager

**目标**：以 block table 建立 logical token position 到 physical KV block 的映射。

**学习重点**：block size、free list、logical/physical block、内部碎片、block 回收。

**任务**：

- 实现 `allocate(request_id)`、`append_slot()`、`free()`、`get_block_table()`。
- 建立 `free_blocks`、`used_blocks`、`request_to_blocks` 不变量。
- 测试跨 block 边界、容量耗尽、重复 free、请求取消和复用。
- 对照 Week 12 workload 测显存利用率、最大并发和 OOM 点。

**产物**：`block.py`、`block_manager.py`、block table 可视化和对照结果。

**验收**：

- 任一 physical block 同一时刻只归属合法 owner/reference。
- 请求完成后 block 全部回到 free list。
- 能从 block table 还原任意 token 对应的物理位置。

### Week 14：FIFO 与 Static Batching

**目标**：先实现简单、可验证的调度基线。

**学习重点**：waiting/running 队列、admission、batch 生命周期、head-of-line blocking。

**任务**：

- 定义 Request/Sequence 状态：waiting、running、finished、cancelled。
- 实现 FIFO Scheduler V0：取前 N 个请求，整批结束后再取下一批。
- 构造输出长度 10/100/20/200 的请求，记录空闲 batch slot。
- 记录 queueing、prefill、decode、finish 时间戳。

**产物**：Scheduler V0、状态转换测试、Static Batching 基线。

**验收**：

- 队列顺序和请求状态转换确定且可测试。
- 不丢请求、不重复生成、不提前释放 KV。
- 能量化短请求结束后的 slot 浪费。

### Week 15：Continuous Batching

**目标**：把调度单位从 batch 生命周期改成 iteration 生命周期。

**学习重点**：reclaim、admit、build batch、forward、sample 的 iteration 顺序。

**任务**：

- 每轮回收 finished 请求，再准入 waiting 请求填补空位。
- 支持不同 prompt/output length，维护各请求 token 与 KV 状态。
- 对照 Static Batching 扫描 concurrency=1/2/4/8/16/32（显存允许时再到 64）。
- 同时记录 tokens/s、requests/s、p50/p95 TTFT、p50/p95 ITL。

**产物**：动态 Scheduler、请求时间线、第一版核心 serving 对照图。

**验收**：

- 请求完成后的下一个 iteration 可以接纳新请求。
- 生成结果与逐请求 reference 一致。
- 吞吐收益与 tail latency 代价都被报告，不只展示有利指标。

### Week 16：Scheduler 与内存预算集成

**目标**：形成最小 mini-vLLM 闭环，让调度决策受 token/block budget 约束。

**学习重点**：`max_num_seqs`、`max_num_batched_tokens`、memory budget、prefill/decode 竞争。

**任务**：

- 设计 `SchedulerOutput`，显式返回本轮 prefill/decode 请求和 block 操作。
- 准入前检查 token 与 KV block 预算，容量不足时等待而非 OOM。
- 处理请求完成、取消、容量不足和异常回滚。
- 此时再阅读 Mini-SGLang、vLLM Scheduler/KV Cache Manager，对比接口与不变量。

**产物**：Engine + Scheduler + Block Manager 集成路径、架构图、压力测试。

**验收**：

- 混合请求压力下不超出配置预算。
- 请求终止后资源可完整回收。
- Runtime 与内存层边界清晰，能说明每 iteration 的完整状态变化。

### Week 17：Hash-based Prefix Cache

**目标**：复用共享 prompt 的 KV，降低重复 prefill 和 TTFT。

**学习重点**：block token hash、命中粒度、引用与回收、cache hit rate；Radix Tree 仅做理解或扩展。

**任务**：

- 先实现 block 级 hash → KV block 映射。
- 处理相同 prefix、部分 prefix、不命中和 hash 校验。
- 构造 shared prefix=0%/25%/50%/75%/90% 的 workload。
- 测 TTFT、prefill tokens computed、tokens/s 和 cache hit rate。

**产物**：Prefix Cache V1、命中测试、prefix reuse vs TTFT 图。

**验收**：

- 命中时不重复计算对应 prefill token。
- 不同 token 序列不会错误复用 KV。
- cache block 的引用与释放不会导致悬空访问或泄漏。

### Week 18：统一 Benchmark Suite

**目标**：让所有关键结论可由一条稳定入口复现。

**学习重点**：workload 设计、warmup/repeat、percentile、控制变量、原始数据与图表分离。

**任务**：

- 统一 Kernel Benchmark：PyTorch eager、Triton、CUDA，可选 `torch.compile`。
- 实现 Serving load generator，支持并发、prompt/output length 分布、shared prefix ratio 和 seed。
- 输出结构化原始数据、环境元数据和图表。
- 完成 Static vs Continuous、Contiguous vs Paged、No Prefix vs Prefix 三组实验。

**产物**：`benchmarks/`、可重跑命令、原始数据 schema、初版 7 张核心图。

**验收**：

- 同一配置重复执行趋势稳定，异常值处理规则预先定义。
- TTFT、ITL、tokens/s、requests/s、peak memory 均可计算。
- 图上标明 workload、硬件、dtype 和 baseline。

### Week 19：端到端 Nsight Profiling

**目标**：解释时间到底花在哪里，并定位 Runtime 与 Kernel 的交界瓶颈。

**学习重点**：CPU/GPU overlap、CUDA API、memcpy、stream、GPU idle gap、调度开销、prefill/decode 分段。

**任务**：

- 用 NVTX 标注 request queue、schedule、prefill、decode、attention、sampling。
- 选择低并发、高并发、长 prompt 三个代表 workload。
- 用 Nsight Systems 找 GPU idle、launch gap、同步和 CPU bottleneck。
- 对关键慢 kernel 下钻 Nsight Compute，更新优化日志。

**产物**：端到端 timeline、瓶颈清单、证据截图/报告和优化优先级。

**验收**：

- 每个主要时间区间能映射回代码阶段。
- 至少一个端到端瓶颈有数据支持的改进或“不值得优化”结论。
- profiling workload 可由命令和配置复现。

### Week 20：Kernel 接入、回归与交付

**目标**：完成 Model → Runtime → Kernel → GPU → Benchmark 闭环。

**任务**：

- 通过稳定接口把自定义 RMSNorm、Softmax/Attention 中成熟实现接回 Runtime。
- 对接入前后执行完整 correctness regression 和 serving benchmark。
- 区分 microbenchmark speedup 与 end-to-end speedup，解释差距。
- 整理架构、优化日志、环境、复现命令、结果、已知限制和后续路线。

**产物**：最终代码、测试、Benchmark、Profiler 报告、项目 README 与技术报告。

**验收**：

- 新 Kernel 可配置启停，reference 路径仍可用于回归。
- 一条命令能跑 smoke test，一条命令能跑精简 benchmark。
- 所有公开性能数字能追溯到原始数据与环境。
- 未实现和不支持项明确列出，不把论文或其他硬件结果当作本项目结果。

## 5. 毕业验收

### 5.1 Runtime 验收

- [ ] 能解释并在代码中定位 prefill 和 decode。
- [ ] 能计算 KV Cache 每 token/每请求/总显存。
- [ ] Block table 能正确映射 logical block 与 physical block。
- [ ] 请求结束、取消和异常后 KV block 都会回收。
- [ ] Static 与 Continuous Batching 有同 workload 对照。
- [ ] Scheduler 每 iteration 的 reclaim/admit/build/execute 顺序明确。
- [ ] token、sequence 和 memory budget 会约束调度。
- [ ] Prefix Cache 的命中、校验、引用和回收正确。
- [ ] 能用实测说明 throughput/latency trade-off。

### 5.2 Kernel 验收

- [ ] 能解释 warp、coalescing、register、shared memory 和 occupancy。
- [ ] 能区分 memory-bound、compute-bound 与 launch-bound。
- [ ] RMSNorm、Softmax、Attention 有 reference 和优化版本。
- [ ] 能解释 fusion、reduction、tiling、online softmax 的收益与代价。
- [ ] 对一个慢 kernel 完成 benchmark → profiler → hypothesis → change → re-benchmark。
- [ ] 报告包含退化 shape，不选择性隐藏结果。

### 5.3 工程验收

- [ ] 环境、随机种子、模型、dtype、shape/workload 均被记录。
- [ ] correctness test 与性能 test 分离且都能自动运行。
- [ ] 原始数据与图表生成逻辑可追溯。
- [ ] README 包含架构、运行方法、结果、限制和优化历程。
- [ ] 新环境按文档可以完成 smoke test 和代表性 benchmark。

## 6. 风险与回退策略

| 风险 | 早期信号 | 回退动作 |
| --- | --- | --- |
| 模型过大或显存不足 | Week 1 即 OOM，后续无法留 KV 空间 | 换更小配置/更短 context，但固定架构与 dtype |
| 同时学太多主题 | 连续两周只有笔记、无可运行产物 | 冻结扩展项，只完成当周最小验收 |
| Kernel 正确但结果漂移 | 长序列或极值输入失败 | 回到 FP32 reference、缩小 shape、逐步检查 reduction |
| Benchmark 波动大 | 重跑排序变化、离散度过高 | 隔离后台负载、固定 clocks（若可行）、增加 warmup/repeat 并报告分布 |
| microbenchmark 快但端到端无收益 | Kernel 快，tokens/s 不变 | 用 Systems 检查占比、launch 和调度瓶颈，保留“不值得接入”结论 |
| Scheduler 状态复杂 | 丢请求、block 泄漏、死循环 | 缩回 FIFO/单请求，给状态机和不变量补测试后逐项恢复 |
| 过早进入高级功能 | V1 主链仍不稳定却开始 TP/MoE | 将高级功能移入 backlog，Week 20 后再评估 |

## 7. 后续路线（V1 完成后）

按瓶颈和兴趣一次只选一个方向：chunked prefill、CUDA Graph、quantization、speculative decoding、Tensor Parallel/NCCL、MoE/Expert Parallel、PD disaggregation 或 multi-node。每个方向继续沿用相同的“基线—证据—改动—复测”方法，不作为本 20 周计划的完成条件。

