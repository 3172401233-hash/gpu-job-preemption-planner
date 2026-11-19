# gpu-job-preemption-planner
Python GPU 作业抢占规划库，处理显存碎片、检查点成本、配额和公平性

## 概述

`gpu_planner.py` 是一个纯规划器（Python 3.8.2+）：给定集群 GPU 拓扑
（节点、型号、单卡显存）、当前任务的占用集合、预计释放时间、检查点成本、
配额与截止时间，为即将超时的新任务计算一份腾出 GPU 的计划。

**规划器只输出计划，不连接真实 GPU，不执行 kill，也不执行 checkpoint。**
实际动作由外部执行器按计划落实。

## 能力

- 三种动作：`MIGRATE`（迁到有空闲显存的卡）、`PAUSE`（checkpoint 后挂起）、
  `PREEMPT`（直接抢占，丢失未检查点的工作）
- 迁移链：为迁走目标任务可递归迁走挡路任务；`visited` 集合 +
  `MAX_CHAIN_DEPTH` 深度上限保证链必然终止，成环时退化为暂停/抢占
- 公平配额：超出公平份额的用户优先成为受害者；同等情况优先丢失工作量小的任务
- 显存形状检查：数量够但单卡显存形状不足（碎片）时明确失败，不会假装分配成功
- 截止时间：checkpoint 耗时超过截止时间且不可抢占的任务不会被选为受害者；
  截止前会自然释放的任务不被惊动

## 输入 / 输出

```python
from gpu_planner import GPU, Job, Planner, Request

planner = Planner(
    gpus=[GPU("g0", "node0", "A100", memory_total_mb=40000, memory_used_mb=40000)],
    jobs=[Job("j1", "alice", ("g0",), memory_per_gpu_mb=40000,
              expected_release_s=None, checkpoint_cost_s=30,
              lost_work_s=300, quota=1, preemptible=True)],
)
plan = planner.plan(Request("r1", "bob", num_gpus=1,
                            memory_per_gpu_mb=40000, model="A100",
                            deadline_s=600))
```

`Plan` 字段：

- `success`：是否能满足请求
- `victims`：受害者有序列表（任务、动作、恢复成本、释放的卡、可用时间）
- `freed_gpus`：释放后可用的卡集合
- `reasons`：无法满足时的原因（碎片 / 总量不足 / 型号不足 / 截止时间）
- `recovery_costs`：每个受害任务的预计恢复成本（秒）

输入校验：重复的 GPU 键、重复的任务键、引用未知 GPU、显存账目不一致
都会抛出 `ValueError`。

## 测试

```bash
python -m unittest
```

覆盖：单卡碎片、跨型号资源、抢占/迁移链（含成环终止）、公平配额、
截止时间已无法满足、重复键。
