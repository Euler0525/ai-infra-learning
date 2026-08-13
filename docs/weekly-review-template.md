# Week XX 复盘

> 将本文件复制为独立周报后填写；不要直接覆盖模板。

## 1. 本周目标

- 计划里程碑：
- 本周硬验收：
- 实际投入时间：
- 代码版本/commit：

## 2. 完成情况

| 任务 | 状态（完成/部分/阻塞） | 证据（代码、测试、数据、图） |
| --- | --- | --- |
|  |  |  |

## 3. Correctness

- Reference：
- 测试 shape/workload：
- dtype 与误差阈值：
- 测试命令：
- 结果：
- 未覆盖边界：

## 4. Benchmark

- 硬件与软件环境：
- Baseline：
- Workload/shape：
- warmup/repeat/统计值：
- 执行命令：
- 原始数据路径：

| 配置 | Median | 离散度/p95 | Speedup | 备注 |
| --- | ---: | ---: | ---: | --- |
|  |  |  |  |  |

## 5. Profiling 与推理

```text
Baseline
  ↓
Profiler evidence
  ↓
Hypothesis
  ↓
Change
  ↓
Re-profile / Re-benchmark
  ↓
Conclusion
```

- 观察到的瓶颈：
- 支持证据：
- 本周假设：
- 修改内容：
- 优化前后差异：
- 有效 shape/workload：
- 退化 shape/workload：
- 决策（保留/回滚/继续）：

## 6. 失败尝试与问题

| 尝试/问题 | 现象 | 已排查 | 结论或下一步 |
| --- | --- | --- | --- |
|  |  |  |  |

## 7. 本周验收

- [ ] 计划要求的功能已完成。
- [ ] Correctness tests 通过。
- [ ] 性能数据包含环境、baseline 和 workload。
- [ ] 原始数据已保存，图表可重建。
- [ ] 失败与退化结果已记录。
- [ ] 文档与实际实现一致。

**验收结论**：通过 / 有条件通过 / 未通过

**若未通过，补齐动作与截止点**：

## 8. 下周入口

- 已满足的前置条件：
- 尚未满足的硬条件：
- 下周第一项可执行任务：
- 暂不处理的 backlog：

