# Composed AVIGATE Smoke20 实验报告

- **日期**: 2025-05-08
- **commit**: `15a3fae Add composed AVIGATE smoke runner`
- **脚本**: `scripts/run_composed_avigate_smoke20.sh`
- **入口**: `app/composed_avigate_smoke.py`

---

## 1. 目标

对比三种方法在 composed video retrieval 任务上的效果：

1. **AVIGATE baseline** — 纯视觉-文本检索，无 Omni 介入
2. **AVIGATE round1 in agent** — Agent 框架第一轮（等价 baseline，验证一致性）
3. **AVIGATE+Qwen2.5-Omni Agent** — Agent 用 Qwen2.5-Omni 对 top-K 候选视频做 rerank

---

## 2. 数据集

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

### 本次实验抽样

从 943 条中随机抽 20 条作为 test query。每条样本的 query 构造为 `reference_caption + " Edit: " + edit_text`，target 为对应的 `target.mp4`。检索池为全部 943 个 target 视频。

抽样涉及 5 个源视频：`-BAFzpKigw`（5条）、`-jR3C_yA_G0`（6条）、`-oC3FVOx62g`（1条）、`-q6f1XIGuL4`（6条）、`XRKONruE`（2条）。

---

## 3. 运行流程

### 3.1 环境

- **Qwen2.5-Omni 7B**（`/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen2.5-omni`）
  - GPU 0, 1，端口 8092，tensor-parallel-size 2，gpu-memory-utilization 0.70
- **AVIGATE** 推理用 GPU 4
- Python 环境：`omni_src` conda 环境（Python 3.10 + PyTorch + vLLM）

### 3.2 启动命令

```bash
# 1. 拉代码
git pull --ff-only origin main  # 15a3fae

# 2. 启动 Qwen2.5-Omni 服务
CUDA_VISIBLE_DEVICES=0,1 nohup python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 8092 \
  --model /data02/.../qwen2.5-omni \
  --served-model-name qwen2.5-omni \
  --trust-remote-code --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.70

# 3. 跑 smoke
GPU_ID=4 \
CHECKER_BASE_URL=http://127.0.0.1:8092/v1 \
CHECKER_MODEL=qwen2.5-omni \
SAMPLE_SIZE=20 \
OMNI_CONCURRENCY=2 \
RERANK_WINDOW=5 \
bash scripts/run_composed_avigate_smoke20.sh
```

### 3.3 Agent 工作流程

对每个 query，Agent 执行：

1. **Query Understanding** — 用 Omni 理解 query 文本，提取 retrieval hints
2. **AVIGATE 初检** — 用 AVIGATE 检索 top-K（K=RERANK_WINDOW=5）候选视频
3. **Omni Rerank** — 对每个候选视频，用 Qwen2.5-Omni 观看视频内容并与 query 比对，打相关性分
4. **重排序** — 按 Omni 相关性分 + AVIGATE 原始分综合排序

每个 query 平均调用 Omni 7 次（1 次 query understanding + 5 次视频描述 + 1 次 rerank）。

---

## 4. 结果

### 4.1 检索指标

| 方法 | R@1 | R@5 | R@10 |
|------|----:|----:|-----:|
| AVIGATE baseline | 0.20 | 0.85 | 1.00 |
| AVIGATE round1 in agent | 0.20 | 0.85 | 1.00 |
| **AVIGATE+Qwen2.5-Omni Agent** | **0.25** | 0.85 | 1.00 |

### 4.2 Agent 统计

| 指标 | 值 |
|------|-----|
| 平均 Omni 调用次数 | 7.0 |
| query rewrite 率 | 5%（20 条中仅 1 条改写） |
| audio fallback 率 | 0% |
| 检索模式 | t2v（text-to-video） |

### 4.3 Agent 提升的 query

Agent 在 1 条 query 上将 target 从 rank 2 提升到 rank 1：

- **query**: "a person holds a basketball trading card featuring two players, David Robinson and Hakeem Olajuwon..."
- **target**: `00003_daily_omni_daily_omni_-BAFzpKigw`
- baseline top1 为 `00001`（同源不同 segment），Agent rerank 后将正确 target `00003` 提到 top1

---

## 5. 分析

### 5.1 提升有限

R@1 仅从 0.20 提升到 0.25（+5 个百分点），R@5 和 R@10 无变化。可能原因：

1. **检索池特性** — 同源视频高度相似（同一 30s 视频切成 5 个 6s segment），AVIGATE 的文本-视觉 embedding 已经能把同源视频聚在一起（R@5=0.85 说明 target 基本在前 5），但 top1 区分困难。Omni rerank 只帮了 1 条，说明 Qwen2.5-Omni 7B 在区分同源不同 segment 的细微差异上能力有限。

2. **query rewrite 率极低**（5%）— Agent 几乎不改写 query，说明原始 query（`reference_caption + edit_text`）对 Omni 来说已经足够理解。改写不会带来额外收益。

3. **Rerank window 限制** — RERANK_WINDOW=5，只对 AVIGATE top5 做 rerank。但 baseline R@5 已经 0.85，意味着最多只有 ~3 条被遗漏的 target 可以通过扩大 window 找回。

4. **baseline R@1 偏低**（0.20）— 20 条中只有 4 条 baseline top1 正确。根源在于检索池中同源视频有 5 个相似 candidate，AVIGATE 很难从文本描述区分具体是哪个 6s segment。Omni rerank 也面临同样的困难。

### 5.2 建议

- **扩大 rerank window**：尝试 RERANK_WINDOW=10 或 20，看能否改善 R@1
- **增大样本量**：20 条样本太少，统计波动大。跑 100+ 条才能得到可靠结论
- **考虑视频内容差异更大的检索池**：当前检索池 943 条中很多是同源视频，对 reranker 是极大挑战。混入更多异源视频可能让 rerank 优势更明显
- **尝试更大的 Omni 模型**：Qwen2.5-Omni 7B 对细微视觉差异的区分能力有限，Qwen3-Omni 30B 可能有更好表现

---

## 6. 输出文件

```
runs/composed_avigate_smoke20_20260508_150954/
├── comparison.md          # 对比表格
├── agent/
│   ├── traces.jsonl       # 20 条完整 agent trace
│   └── summary.json       # 汇总统计
└── baseline/
    └── results.jsonl      # baseline 检索结果
```
