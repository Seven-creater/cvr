# Composed AVIGATE 全量双向检索实验报告

- **日期**: 2025-05-08 ~ 2025-05-09
- **commit**: `8f18f60`
- **Gallery**: 943 个视频/文本（全量）
- **脚本**: `logs/run_composed_avigate_full_bidir_20260508_202945.sh`

---

## 1. 任务定义

我们的最终目标是 **Composed Video Retrieval (CVR)**：给定参考视频 + 编辑指令，检索目标视频。

当前 AVIGATE 模型只支持 t2v 和 v2t，不支持 video+text 联合输入。因此我们将 CVR 拆成两个方向分别评估：

| 方向 | 输入 | 检索池 | 评估方式 |
|------|------|--------|----------|
| **T2V**（Text-to-Video） | `reference_caption + " Edit: " + edit_text` → 文本 query | 943 个候选视频 | target 视频是否在 top-K |
| **V2T**（Video-to-Text） | target 视频（视觉 query） | 943 条 caption | 对应 caption 是否在 top-K |

> 注意：两个方向都是 CVR 的简化版。真正的 CVR 应以 reference video（视觉）+ edit text（文本）联合输入。

---

## 2. 输入输出

### T2V

**输入**：
- `reference_caption`：参考视频的文字描述（如 "A person holds a basketball trading card..."）
- `edit_text`：编辑指令（如 "turn the trading card to show the back"）
- 拼接为 query：`reference_caption + " Edit: " + edit_text`

**输出**：按相关性排序的视频列表，评估 target 视频是否出现在 top-1 / top-5 / top-10

### V2T

**输入**：
- `target.mp4`：目标视频（6s segment）

**输出**：按相关性排序的文本列表，评估对应 caption 是否出现在 top-1 / top-5 / top-10

### 具体例子

以样本 `00003_daily_omni_daily_omni_-BAFzpKigw` 为例：

**T2V 模式**：
```
输入 query: "a person holds a basketball trading card featuring two players,
            David Robinson and Hakeem Olajuwon, in a clear plastic case.
            Edit: turn the trading card to show the back."
Target:    00003_daily_omni_daily_omni_-BAFzpKigw（球星卡背面视图）

AVIGATE 初检 top-5:
  rank=1  video=00001  score=43.95   ← 错误（同源不同 segment）
  rank=2  video=00003  score=40.64   ← 正确 target
  rank=3  video=00005  score=40.64
  ...

Omni Rerank 后:
  rank=1  video=00003  score=40.64   ← 正确 target 提升到 top1
  rank=2  video=00005  score=40.64
  rank=3  video=00001  score=43.95
  ...
```

**V2T 模式**：
```
输入 video: 00003 的 target.mp4（球星卡背面）
Target caption: 对应的 reference_caption + edit_text

AVIGATE 用视频特征从 943 条 caption 中检索 top-10
Omni 观看视频 → 生成描述 → 与 top-10 候选 caption 比对 → 重排序
```

---

## 3. 数据集

Single-source video pair 数据集（`/data02/usr/wangqihao/Demo/test/data/`），943 条样本。

### 构建流程

1. 从 193 个源视频（94 daily_omni + 99 worldsense）中，每个切成 5 个 6s segment
2. 同源 segment 两两配对，生成 C(5,2)=10 个候选 pair
3. Omni（qwen3-omni）对每个 pair 做差异标注 + 质量评分
4. 筛选 `final_omni_accept=True` 的 pair，得到 943 条

### 概况

| 项目 | 数值 |
|------|------|
| 总样本 | 943 |
| daily_omni | 419 |
| worldsense | 524 |
| 源视频数 | 193 |
| 正式入选（accepted=True） | 497 |
| 质量过关但被 cap 截掉 | 446 |

### 差异类型分布

| 类型 | 数量 | 占比 |
|------|------|------|
| scene | 513 | 54.4% |
| object_presence | 276 | 29.3% |
| action | 101 | 10.7% |
| attribute | 45 | 4.8% |
| object_count | 7 | 0.7% |
| audio_event | 1 | 0.1% |

### 每条样本包含

```
{序号}_{源ID}/
├── reference.mp4               # 参考视频（6s）
├── target.mp4                  # 目标视频（6s）
├── edit_text.txt               # 编辑指令
├── info.json                   # caption、差异类型、质量分等
├── reference_annotation.json   # 参考视频 Omni 标注
├── target_annotation.json      # 目标视频 Omni 标注
├── reference_omni_description.txt
└── target_omni_description.txt
```

---

## 4. 方法

### 4.1 AVIGATE baseline

AVIGATE 是预训练的视觉-文本检索模型，支持 t2v 和 v2t 两种模式，无需训练，直接推理。

### 4.2 AVIGATE+Qwen2.5-Omni Agent

在 baseline 之上加 Qwen2.5-Omni 做 rerank。

#### T2V 流程

```
query text (reference_caption + " Edit: " + edit_text)
    │
    ▼
Step 1: Query Understanding（Qwen2.5-Omni 理解 query）
    │  提取：main_events、objects、scene、audio_cues
    │  可选改写 query（rewrite 率 14%）
    │
    ▼
Step 2: AVIGATE 初检（GPU 4，文本→视频检索）
    │  从 943 个候选视频中检索 top-5
    │  → 即 baseline 结果
    │
    ▼
Step 3: Omni Rerank（Qwen2.5-Omni，GPU 0/1）
    │  对 top-5 每个视频：Omni 观看 → 生成描述 → 与 query 比对
    │  综合打分重排序
    │
    ▼
最终排序结果 → 评估 R@1/R@5/R@10
```

每个 query 平均 Omni 调用 **6.9 次**（1 次 understanding + 5 次视频描述 + 1 次 rerank）。

#### V2T 流程

```
target video (6s segment)
    │
    ▼
Step 1: AVIGATE 初检（GPU 4，视频→文本检索）
    │  从 943 条 caption 中检索 top-10
    │  → 即 baseline 结果
    │
    ▼
Step 2: Omni Rerank（Qwen2.5-Omni，GPU 0/1）
    │  Omni 观看视频 → 生成描述
    │  比对描述与 top-10 候选 caption 的相关性
    │  重排序
    │
    ▼
最终排序结果 → 评估 R@1/R@5/R@10
```

每个 query 平均 Omni 调用 **2.0 次**（1 次视频描述 + 1 次 rerank），无需 query understanding。

### 4.3 使用的模型和硬件

| 组件 | 模型 | GPU | 用途 |
|------|------|-----|------|
| AVIGATE | ckpt_msrvtt_paper_like_4gpu_stable | GPU 4 | 视觉-文本检索 |
| Qwen2.5-Omni | qwen2.5-omni-7B | GPU 0/1 (tensor-parallel 2) | 视频理解 + rerank |

- AVIGATE 端口：本地推理（加载 checkpoint）
- Qwen2.5-Omni 端口：8092（vLLM serving）
- Python 环境：`omni_src` conda（Python 3.10 + PyTorch + vLLM）

---

## 5. 实验执行

### 5.1 运行命令

```bash
# 后台全量运行（setsid + nohup，断开 SSH 不影响）
setsid nohup bash logs/run_composed_avigate_full_bidir_20260508_202945.sh \
  > logs/composed_avigate_full_bidir_20260508_202945.log 2>&1 < /dev/null &
```

脚本依次执行：
1. **Staging**：将 943 条数据准备为 AVIGATE 格式（split.csv、video_root、audio_root）
2. **Baseline**：AVIGATE 对 943 条同时计算 t2v 和 v2t 基线
3. **T2V Agent**：对 943 条跑 T2V Omni rerank
4. **V2T Agent**：对 943 条跑 V2T Omni rerank
5. **汇总**：生成 comparison_final_only.md

### 5.2 运行时间

| 阶段 | 耗时 |
|------|------|
| Staging + Baseline | ~30 分钟 |
| T2V Agent（943 条） | ~2.5 小时 |
| V2T Agent（943 条） | ~1.5 小时 |
| **总计** | **~4.5 小时** |

---

## 6. 结果

### 6.1 主表

| Mode | Method | R@1 | R@5 | R@10 |
|------|--------|----:|----:|-----:|
| T2V | AVIGATE baseline | 0.1400 | 0.5928 | 0.7837 |
| T2V | **AVIGATE+Omni Agent** | **0.1707** | 0.6076 | 0.7911 |
| V2T | AVIGATE baseline | 0.1569 | 0.5917 | 0.7815 |
| V2T | **AVIGATE+Omni Agent** | **0.1760** | 0.6002 | 0.7815 |

### 6.2 Agent 对排名的影响

| 指标 | T2V | V2T |
|------|----:|----:|
| Baseline 和 Agent 都正确（R@1） | 122 | 144 |
| Agent 提升到 R@1 | 39 | 22 |
| Agent 从 R@1 退步 | 15 | 4 |
| 两者都不对（R@1） | 767 | 773 |
| **净提升** | **+24** | **+18** |
| 退步率 | 10.9% | 2.7% |

### 6.3 Agent 统计

| 指标 | T2V | V2T |
|------|----:|----:|
| 样本数 | 943 | 943 |
| 平均 Omni 调用 | 6.9 | 2.0 |
| query rewrite 率 | 14.0% | — |
| fallback 率 | 6.15% | 2.55% |

---

## 7. 分析

### 7.1 T2V vs V2T 对比

| 指标 | T2V | V2T |
|------|-----|-----|
| Baseline R@1 | 0.140 | 0.157 |
| Agent 提升 | +3.1pp | +1.9pp |
| 退步率 | 10.9% | 2.7% |
| Omni 调用/query | 6.9 | 2.0 |

- **V2T baseline 更高**（0.157 vs 0.140）：视频作为 query 携带视觉信息，AVIGATE 直接匹配更准
- **T2V Agent 提升更大**（+3.1pp vs +1.9pp）：文本 caption 丢失视觉细节，Omni rerank 补偿空间更大
- **V2T 退步率更低**（2.7% vs 10.9%）：V2T 的 Omni 只需比对视频描述与文本，任务更简单
- **V2T 效率更高**：每 query 仅 2 次 Omni 调用（vs T2V 的 6.9 次）

### 7.2 核心瓶颈

1. **同源干扰** — 193 个源视频切成 943 个 6s segment，同源 5-10 个视频内容极相似。AVIGATE 能定位到同源区域（R@5~0.59），但 top1 区分困难（R@1~0.14）。超过 80% 的 query baseline 和 agent 都拿不到 top1。

2. **Omni rerank 收益有限** — T2V 仅提升 +3.1pp，V2T +1.9pp。R@5/R@10 几乎不变，说明 rerank 只在 top1 精细区分上有微小帮助。

3. **不是真正的 CVR** — T2V 用文本 caption 替代视频，V2T 只用视频不用 edit_text。两个方向都没有同时利用参考视频的视觉特征和编辑指令的文本特征。

### 7.3 与 400 子集对比

| Gallery | N | T2V Baseline R@1 | T2V Agent R@1 | 提升 |
|---------|---|----:|----:|----:|
| 400 子集 | 400 | 0.1550 | 0.2050 | +5.0pp |
| **943 全量** | **943** | **0.1400** | **0.1707** | **+3.1pp** |

全量 gallery 下干扰项更多，Agent 提升幅度缩小，退步率翻倍。400 子集的结果偏乐观。

### 7.4 下一步建议

- **实现真正的 CVR**：设计 reference video + edit text 的 multimodal fusion 模型（combiner network），在标准 CVR benchmark 上跟 SOTA 对比
- **扩大 rerank window**：当前 window=5 太小，~40% 的 target 不在 rerank 范围内
- **尝试更大 Omni 模型**：Qwen3-Omni 30B 对细微视觉差异可能有更好区分力
- **按差异类型分桶评估**：scene / action / object_presence 难度不同，分桶分析 Agent 在哪类差异上最有效

---

## 8. 输出文件

```
runs/composed_avigate_full_bidir_943_20260508_202945/
├── baseline_both.json              # T2V + V2T baseline
├── comparison_final_only.md        # 最终对比表
├── comparison_final_only.json
├── t2v/
│   └── agent/
│       ├── traces.jsonl            # 943 条 T2V agent trace
│       └── summary.json
├── v2t/
│   └── agent/
│       ├── traces.jsonl            # 943 条 V2T agent trace
│       └── summary.json
└── staged/
    ├── split.csv                   # 943 条 query 列表
    ├── data.json
    ├── video_root/                 # 视频 symlink
    └── audio_root/                 # 音频提取
```
