# Composed AVIGATE T2V400 + V2T200 实验报告

- **日期**: 2025-05-08
- **commit**: `8f18f60 Add composed AVIGATE 400 runner`
- **脚本**: `scripts/run_composed_avigate_400.sh`（T2V）、`app/eval.py avigate-agent-partial-eval`（V2T）
- **T2V 运行时间**: 15:59 ~ 17:04（约 1 小时 6 分钟，400 条）
- **V2T 运行时间**: 17:50 ~ 18:15（约 25 分钟，200 条）

---

## 1. 任务定义

我们的目标是 **Composed Video Retrieval (CVR)**：给定参考视频和编辑指令，从候选池中检索出目标视频。

但当前 AVIGATE 模型只支持 t2v 和 v2t 两种模式，不支持 video+text 联合输入。因此我们将 CVR 简化为两种检索方向分别评估：

### T2V（Text-to-Video）

- **输入**：`reference_caption + " Edit: " + edit_text` → 纯文本 query
- **检索**：AVIGATE 从 943 个候选视频中检索
- **评估**：target 视频是否出现在 top-K

### V2T（Video-to-Text）

- **输入**：target 视频（视觉 query）
- **检索**：AVIGATE 从 943 条 caption 中检索
- **评估**：对应 query 文本是否出现在 top-K

### 具体例子

以样本 `00003_daily_omni_daily_omni_-BAFzpKigw` 为例（T2V 模式）：

**输入**：
- 参考视频（reference.mp4）：一个人手持篮球球星卡（David Robinson & Hakeem Olajuwon），蓝色背景
- 编辑指令："turn the trading card to show the back"（把球星卡翻到背面）
- 目标视频（target.mp4）：同一张卡的背面视图
- 构造的 query："a person holds a basketball trading card featuring two players, David Robinson and Hakeem Olajuwon, in a clear plastic case. Edit: turn the trading card to show the back."

**流程**：

```
1. Query Understanding（Omni 理解 query）
   → 提取：主要事件、关键物体、场景描述、音频线索

2. AVIGATE 初检（文本→视频检索）
   → 从 943 个候选中返回 top-5：
     rank=1  video=00001  score=43.95   ← 错误（同源不同 segment）
     rank=2  video=00003  score=40.64   ← 正确 target
     rank=3  video=00005  score=40.64
     rank=4  video=00002  score=39.95
     rank=5  video=00004  score=39.95

3. Omni Rerank（Qwen2.5-Omni 逐个观看 top-5 视频，与 query 比对）
   → 重新排序后：
     rank=1  video=00003  score=40.64   ← 正确 target 提升到 top1
     rank=2  video=00005  score=40.64
     rank=3  video=00001  score=43.95
     rank=4  video=00002  score=39.95
     rank=5  video=00004  score=39.95

4. 输出：top1 = 00003（正确）
```

在这个例子中，AVIGATE 把 target 排在第 2 位，Omni 通过实际观看视频内容判断 00003 更符合"翻到背面"的描述，将其提升到 top1。

---

## 2. 评估目标

对比两种检索方向上，AVIGATE baseline 与 AVIGATE+Qwen2.5-Omni Agent 的效果差异。

---

## 3. 数据集

使用已有的 single-source video pair 数据集（`/data02/usr/wangqihao/Demo/test/data/`），943 条样本。

### 数据集概况

| 项目 | 数值 |
|------|------|
| 总样本 | 943 |
| daily_omni | 419 |
| worldsense | 524 |
| 源视频数 | 193（94 daily_omni + 99 worldsense） |
| Omni 全部通过（final_omni_accept=True） | 943 |
| 正式入选（accepted=True） | 497 |
| 质量过关但被 cap 截掉 | 446 |

### 每条样本包含

```
{序号}_{源ID}/
├── reference.mp4          # 参考视频（6s segment）
├── target.mp4             # 目标视频（6s segment）
├── edit_text.txt          # 编辑指令
├── info.json              # 汇总信息（caption、差异类型、质量分等）
├── reference_annotation.json    # 参考视频 Omni 结构化标注
├── target_annotation.json       # 目标视频 Omni 结构化标注
├── reference_omni_description.txt
└── target_omni_description.txt
```

### 差异类型分布

| 类型 | 数量 |
|------|------|
| scene | 513 |
| object_presence | 276 |
| action | 101 |
| attribute | 45 |
| object_count | 7 |
| audio_event | 1 |

### 三元组构造

使用 `scripts/build_composed_triplets.sh`（commit `b8a8197`）将 943 条样本物化为三元组：

```
runs/composed_triplets_full_<timestamp>/
├── triplets.jsonl       # 943 行 manifest
├── triplets.csv         # CSV 格式
├── summary.json
├── triplets_media/      # 每个样本的 reference.mp4 / target.mp4 / edit_text.txt（symlink）
│   └── {sample_id}/
│       ├── reference.mp4
│       ├── target.mp4
│       └── edit_text.txt
```

### 本次实验抽样

- **T2V**：从 943 条中随机抽 400 条，query = `reference_caption + " Edit: " + edit_text`，检索池为全部 943 个 target 视频
- **V2T**：复用同一 staged 目录的前 200 条，query = target 视频，检索池为全部 943 条 caption

---

## 4. 运行流程

### 4.1 环境

- **Qwen2.5-Omni 7B**（`/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen2.5-omni`）
  - GPU 0, 1，端口 8092，tensor-parallel-size 2，gpu-memory-utilization 0.70
- **AVIGATE** 推理用 GPU 4
- Python 环境：`omni_src` conda 环境（Python 3.10 + PyTorch + vLLM）

### 4.2 启动命令

```bash
# 1. 拉代码
git pull --ff-only origin main  # 8f18f60

# 2. 启动 Qwen2.5-Omni 服务
CUDA_VISIBLE_DEVICES=0,1 nohup python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 8092 \
  --model /data02/.../qwen2.5-omni \
  --served-model-name qwen2.5-omni \
  --trust-remote-code --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.70

# 3. 跑 T2V 400 条
setsid nohup bash scripts/run_composed_avigate_400.sh > "$LOG" 2>&1 < /dev/null &

# 4. 跑 V2T 200 条（复用同一 staged 目录）
setsid nohup bash -lc "
  CUDA_VISIBLE_DEVICES=4 python3 -m app.eval avigate-agent-partial-eval \
    --mode v2t --sample-size 200 \
    --output-dir $LATEST/v2t200/agent \
    --checker-model qwen2.5-omni \
    --checker-base-url http://127.0.0.1:8092/v1 \
    ...
" > "$V2T_LOG" 2>&1 < /dev/null &
```

### 4.3 端到端流程

#### T2V 流程（文本→视频）

```
┌─────────────────────────────────────────────────────────────────┐
│  输入：reference_caption + " Edit: " + edit_text → 文本 query   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Query Understanding（Qwen2.5-Omni）                     │
│  - 理解 query 文本，提取 main_events / objects / scene           │
│  - 可选：改写 query                                              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: AVIGATE 初检（GPU 4）                                   │
│  - 从 943 个候选视频中检索 top-5                                  │
│  → 即 baseline 结果                                              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Omni Rerank（Qwen2.5-Omni，GPU 0/1）                   │
│  - 对 top-5 每个视频：Omni 观看 → 描述 → 与 query 比对           │
│  - 综合打分重排序                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  输出：重排序后的 top-K 视频列表 → 评估 R@1/R@5/R@10             │
└─────────────────────────────────────────────────────────────────┘

每个 query 平均 Omni 调用 ~7 次，耗时约 10 秒。
```

#### V2T 流程（视频→文本）

```
┌─────────────────────────────────────────────────────────────────┐
│  输入：target 视频（视觉 query）                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: AVIGATE 初检（GPU 4）                                   │
│  - 用 target 视频从 943 条 caption 中检索 top-10                 │
│  → 即 baseline 结果                                              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Omni Rerank（Qwen2.5-Omni，GPU 0/1）                   │
│  - Omni 观看 target 视频 → 生成视频描述                           │
│  - 比对视频描述与 top-10 候选 caption 的相关性                    │
│  - 重排序                                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  输出：重排序后的 top-K 文本列表 → 评估 R@1/R@5/R@10             │
└─────────────────────────────────────────────────────────────────┘

每个 query 平均 Omni 调用 ~2 次，耗时约 7 秒。
```

---

## 5. 结果

### 5.1 检索指标汇总

| Mode | N | Method | R@1 | R@5 | R@10 |
|------|---|--------|----:|----:|-----:|
| T2V | 400 | AVIGATE baseline | 0.1550 | 0.6525 | 0.8475 |
| T2V | 400 | AVIGATE round1 in agent | 0.1525 | 0.6650 | 0.8500 |
| T2V | 400 | **AVIGATE+Qwen2.5-Omni Agent** | **0.2050** | 0.6650 | 0.8500 |
| V2T | 200 | AVIGATE baseline | 0.1750 | 0.6550 | 0.8650 |
| V2T | 200 | **AVIGATE+Qwen2.5-Omni Agent** | **0.1950** | 0.6650 | 0.8650 |

### 5.2 Agent 统计

| 指标 | T2V (400条) | V2T (200条) |
|------|:-----------:|:-----------:|
| 平均 Omni 调用次数 | 6.945 | 1.975 |
| query rewrite 率 | 13.5% | — |
| fallback 率 | 6.75% | 3.5% |
| audio fallback 率 | 0% | 0% |
| 单条耗时 | ~10s | ~7s |
| 总耗时 | ~66 min | ~25 min |

### 5.3 Agent 对 T2V 排名的影响（400 条）

| 变化类型 | 数量 |
|----------|------|
| Baseline 和 Agent 都正确（R@1） | 58 |
| Agent 提升到 R@1（原先不对） | 24 |
| Agent 从 R@1 退步 | 3 |
| 两者都不对（R@1） | 315 |
| Target 排名上升 | 46 |
| Target 排名下降 | 20 |

T2V 净提升 **+21 条** R@1（24 提升 - 3 退步），退步率仅 4.9%。

---

## 6. 分析

### 6.1 T2V vs V2T 对比

- **V2T baseline 更高**（R@1: 0.175 vs 0.155）：视频作为 query 携带更多视觉信息，AVIGATE 能直接提取视觉特征匹配
- **T2V Agent 提升更大**（+5pp vs +2pp）：文本 query 信息有损（caption 丢失视觉细节），Omni rerank 补偿空间更大
- **V2T Omni 更高效**：每个 query 只需 ~2 次 Omni 调用（vs T2V 的 ~7 次），因为不需要 query understanding 步骤

### 6.2 Agent 有效但提升幅度有限

**正面信号**：
- 两个方向上 Agent 都有正向提升，且退步率极低
- 400 条规模下 T2V 提升统计稳定

**瓶颈**：
- R@1 绝对值仍然很低（T2V 0.205, V2T 0.195）
- R@5/R@10 基本不变，说明 Omni rerank 只在 top1 精细区分上有帮助

### 6.3 根因分析

1. **检索池高度同源** — 193 个源视频，每个切成 5 个 6s segment，同源的 5-10 个视频内容极为相似。AVIGATE 的文本-视觉 embedding 能把同源视频聚在一起（R@5~0.65），但 top1 区分极难。

2. **Rerank window 限制** — T2V RERANK_WINDOW=5，V2T topk-value=10，只对有限的候选做 rerank。约 35% 的 target 不在 rerank 范围内。

3. **当前方案不是真正的 CVR** — T2V 用文本 caption 代替视频，V2T 只用视频不用 edit_text。真正的 CVR 应该同时利用参考视频视觉特征和编辑指令文本特征。

4. **Qwen2.5-Omni 7B 能力上限** — 7B 模型对 6s 短视频的细微视觉差异区分力有限。

### 6.4 建议

- **实现真正的 CVR**：在 AVIGATE 之上加 composed embedding 层，融合 reference video embedding + edit text embedding
- **扩大 rerank window**：T2V 尝试 RERANK_WINDOW=10 或 20
- **尝试更大模型**：Qwen3-Omni 30B 对细微视觉差异可能有更好区分力
- **按差异类型分桶评估**：scene / action / object_presence 的难度不同，分桶看 Agent 在哪类差异上最有效

---

## 7. 输出文件

```
runs/composed_avigate_eval400_20260508_155908/
├── comparison.md                      # T2V 原始对比表
├── comparison_t2v400_v2t200.md        # T2V+V2T 合并对比表
├── comparison_t2v400_v2t200.json      # 合并对比 JSON
├── baseline_traces.jsonl              # 400 条 baseline 检索结果
├── baseline_summary.json              # baseline 汇总
├── agent/
│   ├── traces.jsonl                   # 400 条 T2V agent trace
│   └── summary.json                   # T2V agent 汇总
├── v2t200/
│   └── agent/
│       ├── traces.jsonl               # 200 条 V2T agent trace
│       └── summary.json               # V2T agent 汇总
└── staged/                            # 抽样 staging 目录（T2V/V2T 共用）
```

---

## 附录：Smoke20 对比

在正式跑 400 条之前，先跑了 20 条 T2V smoke（commit `15a3fae`），结果趋势一致：

| 方法 | R@1 | R@5 | R@10 |
|------|----:|----:|-----:|
| AVIGATE baseline (20条) | 0.20 | 0.85 | 1.00 |
| AVIGATE+Qwen2.5-Omni Agent (20条) | 0.25 | 0.85 | 1.00 |
