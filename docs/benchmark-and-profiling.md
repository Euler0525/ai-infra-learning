# Benchmark 与 Profiling 执行方案

本文定义项目统一的测量口径。所有性能结论均须遵守本方案；若某次实验例外，应在结果中说明原因。

## 1. 基本原则

1. Correctness 先于性能：优化实现必须先与 reference 对齐。
2. 控制变量：一次实验只改变一个主要因素，其他环境和 workload 保持一致。
3. 保留原始数据：图表不是数据源，必须能由 CSV/JSON 重新生成。
4. 报告完整上下文：硬件、软件版本、dtype、shape/workload、warmup、repeat、统计方法、baseline。
5. 不只展示最佳点：扫描完整范围，报告收益、持平与退化区域。
6. 不预设 speedup 目标：先得到基线和 profiler 证据，再提出优化假设。
7. 区分微基准和端到端：Kernel latency 降低不等价于 TTFT/ITL 或 tokens/s 改善。

## 2. 实验元数据

每批原始数据至少附带以下字段：

```yaml
run_id: <timestamp-or-commit>
git_commit: <commit-sha-or-dirty>
gpu_name: <model>
gpu_memory_mib: <value>
driver_version: <value>
cuda_version: <value>
pytorch_version: <value>
triton_version: <value>
python_version: <value>
model: <model-or-config>
dtype: fp16-or-bf16
seed: <integer>
warmup: <count>
repeat: <count>
command: <reproduction-command>
notes: <clock/background-load/known-exception>
```

若工作区有未提交修改，必须记录 `dirty`，避免结果无法对应到具体实现。

## 3. Correctness 协议

### 3.1 通用检查

- 固定 seed 后构造随机输入，同时保留极值和边界输入。
- reference 默认使用 PyTorch 实现；必要时用 FP32 计算 reference，再转换输出比较。
- 用 `torch.testing.assert_close`，按算子、dtype 明确 `atol` 与 `rtol`，不得为了通过测试临时放宽而不记录原因。
- 检查 shape、dtype、device、NaN/Inf、越界和确定性要求。
- 每个 benchmark shape 必须属于 correctness matrix；未经验证的 shape 不进入性能报告。

### 3.2 最低 Shape Matrix

| 算子 | 最低覆盖 |
| --- | --- |
| RMSNorm | hidden=512/1024/2048/4096/8192/16384；rows=1/32/128；至少一个非理想边界尺寸 |
| Softmax | rows=1/32/128；width=128/512/1024/2048/4096；至少一个非 2 次幂宽度 |
| Attention | batch=1；多种 heads/head_dim；seq=128/256/512/1024，显存允许时扩展到 2048/4096 |
| KV Cache | 跨 block 边界、容量恰好用完、超容量、reset/free、两请求隔离 |
| Scheduler | 不同 prompt/output length、同轮完成、多轮陆续完成、取消、无容量 |
| Prefix Cache | 0/部分/完整命中、相似但不同 token、引用释放、重复访问 |

## 4. Kernel Benchmark

### 4.1 对照实现

- PyTorch eager：必选基线。
- Triton：必选优化实现。
- CUDA：RMSNorm/Softmax 必选，Attention 可按计划能力实现。
- `torch.compile`：可选，必须注明编译模式和是否包含首次编译时间。

### 4.2 计时流程

```text
allocate deterministic inputs
  ↓
correctness check
  ↓
warmup (排除 JIT、加载和缓存初始化)
  ↓
synchronize
  ↓
repeat timed execution
  ↓
synchronize
  ↓
保存每轮或分位统计
```

优先使用 CUDA Event、`torch.utils.benchmark` 或 Triton benchmark 工具。若使用 CPU wall clock，计时区间前后必须正确同步 GPU。内存分配是否包含在计时内必须明确，并在对照实现中保持一致。

### 4.3 统计与输出

每个配置至少输出：

- median latency（μs 或 ms）；
- p20/p80 或标准差/四分位距之一；
- baseline 与 optimized speedup；
- 关键 size/shape、dtype；
- 对 Attention 可附加有效 TFLOPS 与峰值显存，但不能用它们替代 latency。

统一定义：

```text
speedup = baseline_latency / optimized_latency
```

因此 `speedup > 1` 表示优化版本更快。列名必须明确 baseline，避免分子分母歧义。

### 4.4 必做图表

1. RMSNorm latency vs hidden size（按 rows 分组）。
2. Softmax latency vs row width（按 rows 分组）。
3. Attention latency vs sequence length。

每张图同时展示 PyTorch/Triton/CUDA 中可用的实现，图注注明 GPU、dtype、统计值和误差表达。

## 5. Serving Benchmark

### 5.1 指标定义

对请求 `r`：

```text
TTFT(r) = first_token_time(r) - arrival_time(r)
ITL_i(r) = token_time_i(r) - token_time_(i-1)(r)
request_latency(r) = finish_time(r) - arrival_time(r)
token_throughput = total_generated_tokens / wall_clock_time
request_throughput = completed_requests / wall_clock_time
```

至少报告：

- TTFT p50/p95；
- ITL p50/p95；
- request latency p50/p95；
- generated tokens/s；
- requests/s；
- peak GPU memory；
- KV Cache allocated/used/utilization；
- queueing/prefill/decode 时间（可获得时）；
- 实际 batch size/active sequence 数随时间变化。

明确 throughput 的 token 口径是 output token，还是 input+output token；本项目默认使用 generated/output tokens，并可额外报告总处理 token。

### 5.2 Workload 维度

负载生成器至少可控制：

- concurrency；
- request count 或持续时间；
- prompt length 分布；
- output length 分布；
- arrival pattern（首版可用 closed-loop，报告中必须说明）；
- shared prefix ratio；
- random seed。

首轮实验使用小而固定的 deterministic workload 验证正确性，再运行压力 workload。若某 concurrency OOM，记录 OOM 点，不静默降低参数。

## 6. 三组核心对照实验

### 6.1 Static vs Continuous Batching

**不变量**：模型、权重、dtype、prompt/output 分布、请求数、GPU、memory limit。

**自变量**：scheduler policy；concurrency=1/2/4/8/16/32，资源允许时加入 64。

**输出**：

- concurrency vs generated tokens/s；
- concurrency vs p50/p95 TTFT；
- concurrency vs p50/p95 ITL；
- requests/s；
- 平均/峰值 active sequences 和空闲 slot。

**结论要求**：同时解释吞吐变化与 tail latency 变化。若 Continuous Batching 在低并发无收益或更慢，保留并解释结果。

### 6.2 Contiguous/Preallocated vs Paged KV Cache

**不变量**：模型、请求 token 序列、scheduler policy、显存预算。

**自变量**：allocator；sequence length 分布（短、长、长短混合）；block size 可作为第二阶段 sweep。

**输出**：

- allocated vs used KV bytes；
- KV utilization；
- 最大并发；
- OOM point；
- tokens/s 和 TTFT/ITL，确认内存收益是否引入性能代价。

**结论要求**：证明 Block Manager 解决的具体浪费/碎片问题，而不仅是“功能能运行”。

### 6.3 No Prefix Cache vs Prefix Cache

**不变量**：请求数、prompt/output 总长度分布、model、scheduler、memory budget。

**自变量**：shared prefix=0%/25%/50%/75%/90%，cache enabled/disabled。

**输出**：

- prefix ratio vs p50/p95 TTFT；
- 实际 cache hit rate；
- computed vs reused prefill tokens；
- tokens/s；
- cache 占用与淘汰/回收情况。

**结论要求**：0% 命中时的额外开销也必须报告；区分配置的 shared ratio 和实际 cache hit rate。

## 7. Profiling 流程

### 7.1 Nsight Systems：端到端时间去哪了

在代码中使用 NVTX 标记以下区间：

```text
request_queue
schedule
block_allocate / block_free
prefill
decode_iteration
rmsnorm / qkv / attention / mlp
sampling
```

观察：

- CPU 调度与 CUDA API 开销；
- kernel launch gap 和 GPU idle gap；
- H2D/D2H/memcpy；
- 不必要的同步；
- stream 和 CPU/GPU overlap；
- prefill/decode 时间占比随 workload 的变化。

Systems 结论必须对应一个具体 workload 和 timeline 区间。

### 7.2 Nsight Compute：单个 Kernel 为什么慢

只对 Systems 或 microbenchmark 确认的关键 kernel 下钻，重点按假设选择指标，避免无目的收集全部 metric：

- achieved occupancy；
- register usage / register pressure；
- warp stall reasons；
- DRAM/L2/L1 throughput 与 hit rate；
- memory transactions 与访问合并；
- 指令与 Tensor/CUDA Core 利用；
- launch configuration。

一次优化闭环记录：

| 字段 | 内容 |
| --- | --- |
| Baseline | 版本、shape、latency、kernel 数 |
| Evidence | Systems/Compute 指标和瓶颈位置 |
| Hypothesis | 哪个资源限制性能，为什么 |
| Change | fusion/tiling/vectorization/block size 等具体改动 |
| Result | 优化后 latency、speedup、关键指标变化 |
| Range | 哪些 shape 有效，哪些退化 |
| Decision | 保留、回滚或继续实验 |

## 8. 最终结果清单

至少交付以下 7 张图及其原始数据：

- [ ] RMSNorm latency vs hidden size。
- [ ] Softmax latency vs sequence length/row width。
- [ ] Attention latency vs sequence length。
- [ ] Static vs Continuous Batching throughput。
- [ ] Concurrency vs p95 latency（至少 TTFT，可同时展示 ITL）。
- [ ] Paged KV Cache vs baseline GPU memory/utilization。
- [ ] Prefix Cache hit ratio vs TTFT。

每张图回答三个问题：比较了什么、在什么条件下、为什么出现这个结果。不能仅用图标题代替结论。

## 9. 结果目录建议

```text
benchmark-results/
└── <run-id>/
    ├── environment.yaml
    ├── config.yaml
    ├── raw/
    │   ├── kernel.csv
    │   ├── requests.csv
    │   └── timeline-summary.csv
    ├── figures/
    ├── profiles/              # 报告或获取报告的说明
    └── summary.md
```

大型 `.nsys-rep`/`.ncu-rep` 是否入库根据仓库策略决定；即使不入库，也应保存生成命令、关键截图/导出指标和文件校验信息。

## 10. 参考资料

以下资源来自原始学习路径，使用时以本项目实际版本和实测结果为准：

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Triton Vector Add Tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [PyTorch Benchmark Utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [Mini-SGLang](https://github.com/sgl-project/mini-sglang)

