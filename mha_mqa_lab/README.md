# MHA 与 MQA

本目录用最小 PyTorch 代码比较 Multi-Head Attention（MHA）和 Multi-Query Attention（MQA），重点是数学结构、张量维度、KV Cache 和 decode 显存带宽。主要参考 [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)。

## 数学与维度

令 batch size 为 `B`，序列长度为 `T`，query head 数为 `H`，head dimension 为 `Dh`，模型宽度为 `D=H·Dh`。

MHA 的每个 head 都有独立 Q/K/V：

$$
Q_i=XW_{Q,i},\quad K_i=XW_{K,i},\quad V_i=XW_{V,i},
$$

$$
O_i=\operatorname{softmax}\left(\frac{Q_iK_i^\mathsf{T}}{\sqrt{D_h}}+M\right)V_i.
$$

MQA 保留 `H` 个 query head，但所有 query head 共享同一组 K/V：

$$
Q_i=XW_{Q,i},\quad K=XW_K,\quad V=XW_V,
$$

$$
O_i=\operatorname{softmax}\left(\frac{Q_iK^\mathsf{T}}{\sqrt{D_h}}+M\right)V.
$$

| 张量 | MHA | MQA |
| --- | --- | --- |
| `Q` | `[B,H,T,Dh]` | `[B,H,T,Dh]` |
| `K/V` | `[B,H,T,Dh]` | `[B,1,T,Dh]` |
| attention scores | `[B,H,Q_len,T]` | `[B,H,Q_len,T]` |
| K/V 投影参数 | `2D²` | `2DDh=2D²/H` |
| Q/K/V/O 总参数 | `4D²` | `2D²+2D²/H` |

如果 MHA 的所有 K/V head 完全相同，它就与 MQA 数值等价。因此 MQA 可以理解为对 MHA 加上 `K₁=...=K_H`、`V₁=...=V_H` 的约束。

## KV Cache 与显存带宽

设 Transformer 有 `L` 层，每个元素占 `s` 字节。KV Cache 容量为

$$
\operatorname{KVBytes}_{MHA}=2LBTHD_hs=2LBTDs,
$$

$$
\operatorname{KVBytes}_{MQA}=2LBTD_hs.
$$

MQA 的 KV Cache、每 token 新增写入量，以及 decode 时最低历史 K/V 读取量，理论上都缩小 `H` 倍。以 `B=1,H=32,Dh=128,T=4096,L=32` 和 FP16/BF16 为例，MHA 需要 `2 GiB`，MQA 只需 `64 MiB`。

两者单 token attention 主计算量却相同：

$$
\operatorname{FLOPs}\approx4LBHTD_h=4LBTD.
$$

原因是 MQA 仍然计算 `H` 组 query 和 attention weights；它减少的是 K/V 存储与加载，而不是 query head。论文给出的增量推理内存访问/计算比从 MHA 的

$$
\Theta\left(\frac{n}{d}+\frac{1}{b}\right)
$$

变为 MQA 的

$$
\Theta\left(\frac{1}{d}+\frac{n}{dh}+\frac{1}{b}\right).
$$

## MQA 的代价

- 所有 head 共享 K/V，表达自由度低于 MHA；不同 query head 仍能产生不同权重，但不能使用独立的上下文 K/V 表示。
- 任意训练完成的 MHA 权重不能无损变成 MQA，通常需要重新训练或继续适配。
- 质量可能轻微下降，幅度取决于模型和任务。
- 主要收益位于 KV 带宽受限的逐 token decode；prefill/training 的 attention FLOPs 不会缩小 `H` 倍。
- 实现若物化 `repeat(K/V)`，或 kernel 不能复用共享 K/V，实际速度收益会远低于理论缓存缩减。

GQA 使用 `1 < Hkv < H`，是 MHA 与 MQA 之间的折中，本目录不额外实现。

## 论文结果

以下数据来自论文，不是本机复现。论文通过加宽 MQA 的 FFN 保持模型总参数量一致。

| 任务/指标 | MHA | MQA |
| --- | ---: | ---: |
| WMT14 dev ln(PPL) | 1.424 | 1.439 |
| WMT14 dev BLEU | 26.7 | 26.5 |
| WMT14 test BLEU，beam 1 / 4 | 27.7 / 28.4 | 27.5 / 28.5 |
| Billion Word dev PPL | 29.9 | 30.2 |
| TPUv2 greedy decoder μs/token | 46 | 3.8 |
| TPUv2 beam-4 decoder μs/token | 203 | 32 |

结果表明 MQA 在这些任务上的质量接近但并非严格无损，decode 提速则很明显。TPUv2 数据不能直接等同于 RTX 4060 上的 PyTorch eager 结果。

## 运行

数学、维度和等价性演示：

```powershell
python -m mha_mqa_lab.demo
```

RTX 4060 Laptop 安全微基准：

```powershell
python -m mha_mqa_lab.benchmark
```

默认使用 FP16、`B=1,H=32,Dh=128`，扫描 `T=128/512/2048/8192`。单个 case 最多使用 512 MiB 且不超过当前空闲显存的 20%，32 层缓存只在 `demo.py` 中按公式计算，不会实际分配。可缩小 workload：

```powershell
python -m mha_mqa_lab.benchmark --contexts 128,512,2048 --repeats 30 --max-memory-mib 256
```

基准输出 median、p20/p80、KV Cache、张量峰值和 `logical_kv_gb_s`，并将 CSV 保存到 `benchmark-results/mha-mqa/`。`logical_kv_gb_s` 是理论最低 KV 载荷除以延迟，不是硬件计数器测得的 DRAM 带宽；短上下文还容易被 kernel launch 和通用 `einsum` 开销主导。
