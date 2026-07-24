# Composed AVIGATE T2V 943 全量实验报告

- **日期**: 2025-05-08
- **commit**: `8f18f60`
- **脚本**: `logs/run_composed_avigate_full_bidir_20260508_202945.sh`
- **入口**: `app.eval avigate-agent-partial-eval --mode t2v`
- **Gallery**: 943 个视频（全量）

---

## 1. 任务定义

**Text-to-Video Composed Retrieval**：给定参考视频的文字描述和编辑指令，从 943 个候选视频中检索出目标视频。

- **输入**：`reference_caption + " Edit: " + edit_text` → 纯文本 query
- **输出**：按相关性排序的视频列表
- **评估**：R@1 / R@5 / R@10（target 是否出现在 top-K 中）

> 注意：这是 CVR 的简化版。真正的 CVR 应以 reference video（视觉）+ edit text（文本）联合输入，当前方案用文字 caption 替代了参考视频。

---

## 2. 数据集

Single-source video pair 数据集，943 条样本。

| 项目 | 数值 |
|------|------|
| 总样本 | 943 |
| daily_omni | 419 |
| worldsense | 524 |
| 源视频数 | 193 |
| 每个源视频切成 5 个 6s segment | 同源 5-10 个候选视频 |

差异类型分布：

| 类型 | 数量 | 占比 |
|------|------|------|
| scene | 513 | 54.4% |
| object_presence | 276 | 29.3% |
| action | 101 | 10.7% |
| attribute | 45 | 4.8% |
| object_count | 7 | 0.7% |
| audio_event | 1 | 0.1% |

---

## 3. 方法

### AVIGATE baseline

纯文本→视频检索，无 Omni 介入。

### AVIGATE+Qwen2.5-Omni Agent

```
query text
    │
    ▼
Step 1: Query Understanding（Omni 理解 query，提取 main_events/objects/scene）
    │
    ▼
Step 2: AVIGATE 初检 → top-5 候选视频（即 baseline）
    │
    ▼
Step 3: Omni Rerank（逐个观看 top-5 视频，与 query 比对，重排序）
    │
    ▼
最终排序结果
```

每个 query 平均 Omni 调用 6.9 次。

### 环境

- **Qwen2.5-Omni 7B**：GPU 0/1，端口 8092
- **AVIGATE**：GPU 4
- Python 3.10，`omni_src` conda 环境

---

## 4. 结果

### 4.1 主表

| Method | R@1 | R@5 | R@10 |
|--------|----:|----:|-----:|
| AVIGATE baseline | 0.1400 | 0.5928 | 0.7837 |
| **AVIGATE+Qwen2.5-Omni Agent** | **0.1707** | 0.6076 | 0.7911 |

### 4.2 提升/退步分析

| 变化类型 | 数量 |
|----------|------|
| Baseline 和 Agent 都正确（R@1） | 122 |
| Agent 提升到 R@1 | 39 |
| Agent 从 R@1 退步 | 15 |
| 两者都不对（R@1） | 767 |

- 净提升：**+24 条** R@1（39 提升 - 15 退步）
- Baseline R@1 = 137/943 → Agent R@1 = 161/943
- 退步率：15/137 = **10.9%**（高于 400-gallery 的 4.9%）

### 4.3 Agent 统计

| 指标 | 值 |
|------|-----|
| 平均 Omni 调用 | 6.9 |
| query rewrite 率 | 14.0% |
| fallback 率 | 6.15% |

---

## 5. 分析

### 5.1 全量 vs 400 子集对比

| Gallery | N | Baseline R@1 | Agent R@1 | 提升 |
|---------|---|----:|----:|----:|
| 400 子集 | 400 | 0.1550 | 0.2050 | +5.0pp |
| **943 全量** | **943** | **0.1400** | **0.1707** | **+3.1pp** |

全量 gallery 下：
- Baseline R@1 更低（0.14 vs 0.155）——候选池翻倍后，干扰项更多
- Agent 提升幅度缩小（+3.1pp vs +5.0pp）——Omni 在更大候选池中 rerank 效果减弱
- 退步率上升（10.9% vs 4.9%）——Omni 误判的概率增加

### 5.2 根因分析

1. **同源干扰是核心瓶颈** — 193 个源视频切成 943 个 6s segment，同源视频内容极相似。R@5=0.59 说明 AVIGATE 能大致定位到同源区域，但 top1 区分非常困难（R@1=0.14）。

2. **Omni rerank 的收益递减** — 全量 gallery 下干扰项更多，Omni 需要在更嘈杂的 top-5 中做判断，误判率上升。

3. **Rerank window=5 太小** — 约 40% 的 target 不在 top-5 内，rerank 无法触及。R@5 仅从 0.5928 提升到 0.6076（+1.5pp），说明 rerank 对 top-5 内排序改善有限。

4. **不是真正的 CVR** — 文本 caption 丢失了视频的视觉细节，AVIGATE 只能靠文字描述做匹配。

### 5.3 结论

全量 943 gallery 下，Omni Agent 的 R@1 提升从 +5pp 缩小到 +3.1pp，退步率翻倍（4.9% → 10.9%）。**当前方法（t2v + Omni rerank）的收益不足以支撑独立贡献**，需要更根本的改进方向：

- 实现真正的 CVR（video+text fusion）
- 在标准 CVR benchmark（CIRR、FashionIQ）上跟 SOTA 对比
- 扩大 rerank window 或做多轮 rerank

---

## 6. 输出文件

```
runs/composed_avigate_full_bidir_943_20260508_202945/
├── baseline_both.json          # T2V + V2T baseline 结果
├── t2v/
│   └── agent/
│       ├── traces.jsonl        # 943 条 T2V agent trace
│       └── summary.json        # T2V agent 汇总
├── staged/                     # 943 条 staging 目录
│   ├── split.csv
│   ├── data.json
│   ├── video_root/
│   └── audio_root/
└── (v2t/ 稍后完成)
```
